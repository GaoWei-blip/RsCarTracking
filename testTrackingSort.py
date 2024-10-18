from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import cv2
import numpy as np
import time
import torch
import math

from model.DSFNet import DSFNet, load_model_weight
from tracker.sort import *

from common.config.opts import opts
from common.dataset.MyDataset import MyDataset
from common.external.nms import soft_nms
from common.utils.decode import ctdet_decode
from common.utils.post_process import ctdet_post_process

from progress.bar import Bar


# 置信度阈值
CONFIDENCE_threshold = 0.2
# 矩形框的颜色
COLORS = [(255, 0, 0)]
# 矩形框的线条粗细
FONT = cv2.FONT_HERSHEY_SIMPLEX


def cv2_demo(frame_img, detections):
    """
    根据给定的检测结果，在图像上绘制矩形框和置信度值，然后返回处理后的图像和对应的检测信息
    param frame: 帧图像
    param detections: 一个包含检测框和置信度的二维数组。每行表示一个检测结果，格式为 [x1, y1, x2, y2, confidence]
    """
    det = []
    for i in range(detections.shape[0]):
        if detections[i, 4] >= CONFIDENCE_threshold:
            pt = detections[i, :]
            # 绘制边界框
            cv2.rectangle(frame_img, (int(pt[0]), int(pt[1])), (int(pt[2]), int(pt[3])), (0, 255, 0), 2)
            # 显示目标ID
            cv2.putText(frame_img, str(int(pt[4])), (int(pt[0]), int(pt[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 0, 255), 2)
            det.append([int(pt[0]), int(pt[1]), int(pt[2]), int(pt[3]), detections[i, 4]])
    return frame_img, det


def process(model, image, return_time):
    with torch.no_grad():
        output = model(image)[-1]
        hm = output['hm'].sigmoid_()
        wh = output['wh']
        reg = output['reg']
        torch.cuda.synchronize()
        forward_time = time.time()
        detections = ctdet_decode(hm, wh, reg=reg, K=1000)
    if return_time:
        return output, detections, forward_time
    else:
        return output, detections


def post_process(detections, meta, num_classes=1, scale=1):
    detections = detections.detach().cpu().numpy()
    detections = detections.reshape(1, -1, detections.shape[2])
    detections = ctdet_post_process(
        detections.copy(), [meta['c']], [meta['s']],
        meta['out_height'], meta['out_width'], num_classes)
    for j in range(1, num_classes + 1):
        detections[0][j] = np.array(detections[0][j], dtype=np.float32).reshape(-1, 5)
        detections[0][j][:, :4] /= scale
    return detections[0]


def pre_process(image, scale=1):
    height, width = image.shape[2:4]
    new_height = int(height * scale)
    new_width = int(width * scale)

    inp_height, inp_width = height, width
    c = np.array([new_width / 2., new_height / 2.], dtype=np.float32)
    s = max(height, width) * 1.0

    meta = {'c': c, 's': s,
            'out_height': inp_height,
            'out_width': inp_width}
    return meta


def merge_outputs(detections, num_classes, max_per_image):
    results = {}
    for j in range(1, num_classes + 1):
        results[j] = np.concatenate(
            [detection[j] for detection in detections], axis=0).astype(np.float32)

        soft_nms(results[j], Nt=0.5, method=2)

    scores = np.hstack(
        [results[j][:, 4] for j in range(1, num_classes + 1)])
    if len(scores) > max_per_image:
        kth = len(scores) - max_per_image
        thresh = np.partition(scores, kth)[kth]
        for j in range(1, num_classes + 1):
            keep_inds = (results[j][:, 4] >= thresh)
            results[j] = results[j][keep_inds]
    return results


def test(opt_config, data_label, model_path, saveTxt):
    """
    测试函数
    """
    os.environ['CUDA_VISIBLE_DEVICES'] = opt_config.gpus_str

    print(opt_config.model_name)

    # opt_config.data_dir = './data/RsCarData/images/test/1-1/img/'

    dataset = MyDataset(opt_config, data_label)     # 数据集目录影像和传统COCO数据集有区别，修改数据集类

    # 加载数据集
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1, shuffle=False, num_workers=1, pin_memory=True)

    # 构建并加载模型
    heads = {'hm': dataset.num_classes, 'wh': 2, 'reg': 2}
    model = DSFNet(heads)
    model = load_model_weight(model, model_path)
    model = model.cuda()
    model.eval()

    results = {}
    return_time = False
    scale = 1
    num_classes = dataset.num_classes
    max_per_image = opt_config.K

    file_folder_pre = ''
    im_count = 0

    track_results_save_dir = None  # 追踪结果保存路劲
    fid = None  # 追踪结果保存文件
    mot_tracker1 = Sort()  # 初始化追踪器
    if saveTxt:
        track_results_save_dir = '/output/results'
        if not os.path.exists(track_results_save_dir):
            os.makedirs(track_results_save_dir)

    # 初始化进度条
    num_iters = len(data_loader)
    bar = Bar('processing', max=num_iters)

    for ind, (img_id, pre_processed_images) in enumerate(data_loader):  # 遍历数据加载器 data_loader 中的每一帧图像数据
        # 更新进度条信息，显示当前处理进度、总用时 (total) 和预计剩余时间 (ETA)。
        bar.suffix = '[{0}/{1}]|Tot: {total:} |ETA: {eta:} '.format(
            ind, num_iters, total=bar.elapsed_td, eta=bar.eta_td
        )

        # 每当进入一个新视频帧文件夹时（根据路径中的文件夹名称判断），需要重新初始化跟踪器 mot_tracker 、关闭之前打开的文件并创建新文件。
        file_folder_cur = pre_processed_images['file_name'][0].split('/')[-3]
        if file_folder_cur != file_folder_pre:
            if saveTxt and file_folder_pre != '':
                fid.close()
            file_folder_pre = file_folder_cur
            mot_tracker1 = Sort()
            if saveTxt:
                im_count = 0
                txt_path = os.path.join(track_results_save_dir, file_folder_cur + '.txt')
                fid = open(txt_path, 'w+')

        # read images
        detection = []
        meta = pre_process(pre_processed_images['input'], scale)  # 调整图片尺寸
        image = pre_processed_images['input'].cuda()
        img = pre_processed_images['imgOri'].squeeze().numpy()

        # detection
        output, dets = process(model, image, return_time)
        # POST PROCESS
        dets = post_process(dets, meta, num_classes)
        detection.append(dets)
        ret = merge_outputs(detection, num_classes, max_per_image)

        # update tracker
        dets_track = dets[1]
        # 从检测结果中提取需要跟踪的物体，并根据置信度筛选（超过CONFIDENCE_threshold）
        dets_track_select = np.argwhere(dets_track[:, -1] > CONFIDENCE_threshold)
        dets_track = dets_track[dets_track_select[:, 0], :]  # [左上角坐标x1, 左上角坐标y1, 右下角坐标x2, 右下角坐标y2, id]
        # 进行跟踪更新，输出跟踪结果
        track_bbs_ids = mot_tracker1.update(dets_track)

        # 生成跟踪结果并显示图像
        track = np.zeros(shape=(track_bbs_ids.shape[0], 6))
        track[:, 0] = ind  # 帧号
        track[:, 1] = track_bbs_ids[:, -1]  # id
        # track[:, 2] = (track_bbs_ids[:, 0] + track_bbs_ids[:, 2]) / 2  # 中心点坐标x = (x1+x2)/2
        # track[:, 3] = (track_bbs_ids[:, 1] + track_bbs_ids[:, 3]) / 2  # 中心点坐标y = (y1+y2)/2
        track[:, 2] = track_bbs_ids[:, 0]  # 左上角坐标x1
        track[:, 3] = track_bbs_ids[:, 1]  # 左上角坐标y1
        track[:, 4] = track_bbs_ids[:, 2] - track_bbs_ids[:, 0]  # 宽度 = x2 - x1
        track[:, 5] = track_bbs_ids[:, 3] - track_bbs_ids[:, 1]   # 高度 = y2 - y1

        # frame, det = cv2_demo(img, track_bbs_ids)
        # cv2.imshow('frame', frame)
        # cv2.waitKey(5)

        # 保存追踪结果
        if saveTxt:
            # 逆序排列，使id升序
            track = track[::-1, :]
            # 写入结果
            for it in range(track_bbs_ids.shape[0]):
                fid.write('%d,%d,%0.2f,%0.2f,%0.2f,%0.2f,1,-1,-1,-1\n' % (track[it, 0], track[it, 1], track[it, 2],
                                                                          track[it, 3], track[it, 4], track[it, 5]))
            im_count += 1
        results[img_id.numpy().astype(np.int32)[0]] = ret
        bar.next()
    bar.finish()

    # dataset.run_eval(results, opt.save_results_dir, results_name)


if __name__ == '__main__':
    # 读取参数
    opt = opts().parse()
    print(opt.data_dir)

    # 需要测试的数据集：训练集/验证集/测试集
    split = 'test'
    # 是否保存追踪结果，默认False
    # save_track_results = opt.save_track_results
    save_track_results = True
    # 创建结果保存路径
    if not os.path.exists(opt.save_results_dir):
        os.makedirs(opt.save_results_dir)

    # 如果未在命令行指定模型路径，则使用默认模型路径
    if opt.load_model != '':
        modelPath = opt.load_model
    else:
        modelPath = './checkpoints/model1/model_75.pth'
    # 打印模型路径
    print(modelPath)

    test(opt, split, modelPath, save_track_results)

