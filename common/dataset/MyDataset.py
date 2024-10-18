from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pycocotools.coco as coco
from pycocotools.cocoeval import COCOeval
import numpy as np
import json
import os

import torch.utils.data as data
import numpy as np
import torch
import json
import cv2
import os
import math

from common.config.opts import opts
from common.utils.image import flip, color_aug
from common.utils.image import get_affine_transform, affine_transform
from common.utils.image import gaussian_radius, draw_umich_gaussian, draw_msra_gaussian
from common.utils.image import draw_dense_reg
from common.utils.augmentations import Augmentation

from natsort import natsort


class MyDataset(data.Dataset):
    # opt = opts().parse()
    num_classes = 1
    dense_wh = False
    reg_offset = True
    mean = np.array([0.49965, 0.49965, 0.49965],
                    dtype=np.float32).reshape(1, 1, 3)
    std = np.array([0.08255, 0.08255, 0.08255],
                   dtype=np.float32).reshape(1, 1, 3)

    def __init__(self, opt, split):
        super(MyDataset, self).__init__()

        self.img_dir0 = opt.data_dir

        self.img_dir = opt.data_dir

        self.resolution = None
        #  self.coco = COCO(self.annot_path)
        #  self.img_ids = self.coco.getImgIds()
        # self.transform = opt.transform  # 假设有一些数据增强或预处理操作
        self.down_ratio = opt.down_ratio
        self.max_objs = opt.K
        self.seqLen = opt.seqLen

        self.class_name = [
            '__background__', 's']
        self._valid_ids = [
            1, 2]
        self.cat_ids = {v: i for i, v in enumerate(self._valid_ids)}  # 生成对应的category dict

        self.split = split
        self.opt = opt

        print('==> initializing coco 2017 {} data.'.format(split))
        # self.coco = coco.COCO(self.annot_path)
        from glob import glob
        self.images = glob(os.path.join(self.img_dir0, '*.bmp'))
        # self.images = glob( os.path.join(self.img_dir0, '*.jpg'))
        # print(self.images)
        # self.images = glob( os.path.join(self.img_dir0, '*.bmp'))
        self.images = natsort.natsorted(self.images)
        self.num_samples = len(self.images)

        print('Loaded {} {} samples'.format(split, self.num_samples))

        if split == 'train':
            self.aug = Augmentation()
        else:
            self.aug = None

    def _to_float(self, x):
        return float("{:.2f}".format(x))

    # 遍历每一个标注文件解析写入detections. 输出结果使用
    def convert_eval_format(self, all_bboxes):
        # import pdb; pdb.set_trace()
        detections = []
        for image_id in all_bboxes:
            for cls_ind in all_bboxes[image_id]:
                category_id = self._valid_ids[cls_ind - 1]
                for bbox in all_bboxes[image_id][cls_ind]:
                    bbox[2] -= bbox[0]
                    bbox[3] -= bbox[1]
                    score = bbox[4]
                    bbox_out = list(map(self._to_float, bbox[0:4]))

                    detection = {
                        "image_id": int(image_id),
                        "category_id": int(category_id),
                        "bbox": bbox_out,
                        "score": float("{:.2f}".format(score))
                    }
                    if len(bbox) > 5:
                        extreme_points = list(map(self._to_float, bbox[5:13]))
                        detection["extreme_points"] = extreme_points
                    detections.append(detection)
        return detections

    def __len__(self):
        return self.num_samples

    def save_results(self, results, save_dir, time_str):
        json.dump(self.convert_eval_format(results),
                  open('{}/results_{}.json'.format(save_dir,time_str), 'w'))

        print('{}/results_{}.json'.format(save_dir,time_str))

    def run_eval(self, results, save_dir, time_str):
        self.save_results(results, save_dir, time_str)
        coco_dets = self.coco.loadRes('{}/results_{}.json'.format(save_dir, time_str))
        coco_eval = COCOeval(self.coco, coco_dets, "bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        stats = coco_eval.stats
        precisions = coco_eval.eval['precision']

        return stats, precisions

    def run_eval_just(self, save_dir, time_str, iouth):
        coco_dets = self.coco.loadRes('{}/{}'.format(save_dir, time_str))
        coco_eval = COCOeval(self.coco, coco_dets, "bbox", iouth = iouth)
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        stats_5 = coco_eval.stats
        precisions = coco_eval.eval['precision']

        return stats_5, precisions

    def _coco_box_to_bbox(self, box):
        bbox = np.array([box[0], box[1], box[0] + box[2], box[1] + box[3]],
                        dtype=np.float32)
        return bbox

    def _get_border(self, border, size):
        i = 1
        while size - border // i <= border // i:
            i *= 2
        return border // i
    
    def clahe(self, img):
        
        imgr = img[:, :, 0]
        imgg = img[:, :, 1]
        imgb = img[:, :, 2]

        claher = cv2.createCLAHE(clipLimit=3, tileGridSize=(8,8))
        claheg = cv2.createCLAHE(clipLimit=2, tileGridSize=(8,8))
        claheb = cv2.createCLAHE(clipLimit=1, tileGridSize=(8,8))
        cllr = claher.apply(imgr)
        cllg = claheg.apply(imgg)
        cllb = claheb.apply(imgb)

        rgb_img = np.dstack((cllr,cllg,cllb))
        return rgb_img

    def __getitem__(self, index):
        file_name = self.images[index]
        # file_name = self.coco.loadImgs(ids=[img_id])[0]['file_name']
        # ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        # anns = self.coco.loadAnns(ids=ann_ids)
        # num_objs = min(len(anns), self.max_objs)

        file_name = file_name.replace('\\', '/')
        seq_num = self.seqLen
        # imIdex = int(file_name.split('.')[0].split('/')[-1][3:])
        imIdex = int(file_name.split('/')[-1][3:-4])  # 获取影像索引：文件名000001.bmp -> 001
        imf = file_name.split(file_name.split('/')[-1])[0]
        imtype = '.'+file_name.split('.')[-1]
        if self.resolution is not None:
            img = np.zeros([self.resolution[0], self.resolution[1], 3, seq_num])
        for ii in range(seq_num):
            imIndexNew = '%06d' % max(imIdex - ii, 1)
            imName = imf + imIndexNew + imtype
            # print("1",imName)
            im = cv2.imread(imName)
            # im = cv2.convertScaleAbs(im, alpha=1.5, beta=5)#使用cv2.convertScaleAbs进行亮度对比度调整
            im = cv2.copyMakeBorder(im, 0, 8, 0, 0, cv2.BORDER_CONSTANT, value=(0,0,0))
            # im = self.clahe(im)
            if ii == 0:
                imgOri = im
            #normalize
            if self.resolution is None:
                self.resolution = imgOri.shape[:2]
                img = np.zeros([self.resolution[0], self.resolution[1], 3, seq_num])
            inp_i = (im.astype(np.float32) / 255.)
            inp_i = (inp_i - self.mean) / self.std
            img[:, :, :, ii] = inp_i

        #transpose
        inp = img.transpose(2, 3, 0, 1).astype(np.float32)

        bbox_tol = []
        cls_id_tol = []

        height, width = img.shape[0], img.shape[1]
        c = np.array([img.shape[1] / 2., img.shape[0] / 2.], dtype=np.float32)

        s = max(img.shape[0], img.shape[1]) * 1.0

        output_h = height // self.down_ratio
        output_w = width // self.down_ratio
        num_classes = self.num_classes
        trans_output = get_affine_transform(c, s, 0, [output_w, output_h])

        hm = np.zeros((num_classes, output_h, output_w), dtype=np.float32)
        wh = np.zeros((self.max_objs, 2), dtype=np.float32)
        dense_wh = np.zeros((2, output_h, output_w), dtype=np.float32)
        reg = np.zeros((self.max_objs, 2), dtype=np.float32)
        ind = np.zeros((self.max_objs), dtype=np.int64)
        reg_mask = np.zeros((self.max_objs), dtype=np.uint8)
        cat_spec_wh = np.zeros((self.max_objs, num_classes * 2), dtype=np.float32)
        cat_spec_mask = np.zeros((self.max_objs, num_classes * 2), dtype=np.uint8)

        draw_gaussian = draw_umich_gaussian

        ret = {'input': inp, 'hm': hm, 'reg_mask': reg_mask, 'ind': ind, 'wh': wh, 'imgOri': imgOri}

        if self.dense_wh:
            hm_a = hm.max(axis=0, keepdims=True)
            dense_wh_mask = np.concatenate([hm_a, hm_a], axis=0)
            ret.update({'dense_wh': dense_wh, 'dense_wh_mask': dense_wh_mask})
            del ret['wh']

        if self.reg_offset:
            ret.update({'reg': reg})

        ret['file_name'] = file_name

        return imIdex, ret
