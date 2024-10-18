from __future__ import print_function

import numpy as np
import matplotlib

matplotlib.use('Agg')

from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment

np.random.seed(0)


def linear_assignment(cost_matrix):
    try:
        import lap
        _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
        return np.array([[y[i], i] for i in x if i >= 0])  #
    except ImportError:
        x, y = linear_sum_assignment(cost_matrix)
        return np.array(list(zip(x, y)))


def iou_batch(bb_test, bb_gt):
    """
  From SORT: Computes IOU between two bboxes in the form [x1,y1,x2,y2]
  """
    bb_gt = np.expand_dims(bb_gt, 0)
    bb_test = np.expand_dims(bb_test, 1)

    xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
    yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
    xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
    yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / ((bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])
              + (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1]) - wh)
    return (o)


def convert_bbox_to_z(bbox):
    """
  Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
    [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
    the aspect ratio
  """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w / 2.
    y = bbox[1] + h / 2.
    s = w * h  #scale is just area
    r = w / float(h)
    return np.array([x, y, s, r]).reshape((4, 1))


def convert_x_to_bbox(x, score=None):
    """
  Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
    [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
  """
    w = np.sqrt(x[2] * x[3])
    h = x[2] / w
    if (score == None):
        return np.array([x[0] - w / 2., x[1] - h / 2., x[0] + w / 2., x[1] + h / 2.]).reshape((1, 4))
    else:
        return np.array([x[0] - w / 2., x[1] - h / 2., x[0] + w / 2., x[1] + h / 2., score]).reshape((1, 5))


class KalmanBoxTracker(object):
    """
  This class represents the internal state of individual tracked objects observed as bbox.
  """
    count = 0

    def __init__(self, bbox, feature):
        """
    Initialises a tracker using initial bounding box.
    """
        #define constant velocity model
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array(
            [[1, 0, 0, 0, 1, 0, 0], [0, 1, 0, 0, 0, 1, 0], [0, 0, 1, 0, 0, 0, 1], [0, 0, 0, 1, 0, 0, 0],
             [0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 1]])
        self.kf.H = np.array(
            [[1, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0]])

        self.kf.R[2:, 2:] *= 10.
        self.kf.P[4:, 4:] *= 1000.  # give high uncertainty to the unobservable initial velocities
        self.kf.P *= 10.
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01

        self.kf.x[:4] = convert_bbox_to_z(bbox)
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        self.feature = feature

    def update(self, bbox, feature):
        """
        Updates the state vector with observed bbox.
        """
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(convert_bbox_to_z(bbox))
        self.feature = feature

    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(convert_x_to_bbox(self.kf.x))
        return self.history[-1]

    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        return convert_x_to_bbox(self.kf.x)


def associate_detections_to_trackers(detections, trackers, iou_threshold=0.3):
    """
    Assigns detections to tracked object (both represented as bounding boxes)

    Returns 3 lists of matches, unmatched_detections and unmatched_trackers
    """
    if len(trackers) == 0:
        return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 5), dtype=int)

    iou_matrix = iou_batch(detections, trackers)

    if min(iou_matrix.shape) > 0:
        a = (iou_matrix > iou_threshold).astype(np.int32)
        if a.sum(1).max() == 1 and a.sum(0).max() == 1:
            matched_indices = np.stack(np.where(a), axis=1)
        else:
            matched_indices = linear_assignment(-iou_matrix)
    else:
        matched_indices = np.empty(shape=(0, 2))

    unmatched_detections = []
    for d, det in enumerate(detections):
        if d not in matched_indices[:, 0]:
            unmatched_detections.append(d)
    unmatched_trackers = []
    for t, trk in enumerate(trackers):
        if t not in matched_indices[:, 1]:
            unmatched_trackers.append(t)

    #filter out matched with low IOU
    matches = []
    for m in matched_indices:
        if iou_matrix[m[0], m[1]] < iou_threshold:
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1, 2))
    if len(matches) == 0:
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


def extract_features(bbox_xyxy, ori_img):
    extractor = Extractor('./tracker/ckpt40.t7', use_cuda=True)
    im_crops = []
    for box in bbox_xyxy:
        x1, y1, x2, y2 = box
        im = ori_img[y1:y2, x1:x2]
        im_crops.append(im)
    if im_crops:
        features = extractor(im_crops)
    else:
        features = np.array([])
    return features


def compute_similarity(dets, trks, det_features, trk_features):
    """
    :param dets:
    :param trks:
    :param det_features: (N, 512)
    :param trk_features: (M, 512)
    :return:
    """
    # 计算外观特征之间的欧几里得距离
    feature_distances = np.linalg.norm(det_features[:, np.newaxis] - trk_features, axis=2)  # (N, M)

    # 计算检测框和追踪器框的中心点
    det_centers = (dets[:, :2] + dets[:, 2:]) / 2  # (N, 2)
    trk_centers = (trks[:, :2] + trks[:, 2:]) / 2  # (M, 2)

    # 计算边框中心之间的距离
    bbox_distances = np.linalg.norm(det_centers[:, np.newaxis] - trk_centers, axis=2)  # (N, M)

    # 归一化
    feature_distances = feature_distances / np.max(feature_distances)
    bbox_distances = bbox_distances / np.max(bbox_distances)

    # 使用 DeepSORT 的加权方法计算组合相似度
    # 通常情况下，DeepSORT 使用的加权方法为：
    # combined_similarity = alpha * feature_distances + (1 - alpha) * bbox_distances
    alpha = 0.2  # 可以根据实际情况调整
    combined_similarity = alpha * feature_distances + (1 - alpha) * bbox_distances

    return combined_similarity


def cascade_matching(detections_indices, trackers_indices, detections, trackers, max_frame=40):
    if len(trackers_indices) == 0:
        return np.empty((0, 2), dtype=int), np.array(detections_indices), np.empty((0, 5), dtype=int)

    matches = []
    unmatched_detections = detections_indices
    for level in range(max_frame):

        # 如果没有剩余的检测要匹配，则提前退出循环
        if len(unmatched_detections) == 0:
            break

        # 获取该级别（即未更新的帧数为 1+level）中需要匹配的轨迹索引
        track_indices_l = [
            k for k in trackers_indices
            if trackers[k].time_since_update == 1 + level
        ]

        # 如果该级别没有轨迹需要匹配，则跳过这一层
        if len(track_indices_l) == 0:
            continue

        matches_l, _, unmatched_detections = match(trackers, detections, track_indices_l, unmatched_detections)

        matches += matches_l

    # 找出那些未匹配的轨迹（总的轨迹集合减去已匹配的轨迹）
    unmatched_tracks = list(set(trackers_indices) - set(k for _, k in matches))

    return np.array(matches), np.array(unmatched_detections), np.array(unmatched_tracks)


def match(trackers, detections, track_indices, detection_indices, max_distance=0.2):
    # 计算相似度矩阵
    level_trackers = np.array([trackers[i].get_state()[0] for i in track_indices])
    level_detections = np.array([detections[i].bbox[:4] for i in detection_indices])
    level_trackers_features = np.array([trackers[i].feature for i in track_indices])
    level_detections_features = np.array([detections[i].feature for i in detection_indices])

    cost_matrix = compute_similarity(level_detections, level_trackers, level_detections_features, level_trackers_features)

    # 使用匈牙利算法进行最优匹配
    row_indices, col_indices = linear_sum_assignment(cost_matrix)

    # 初始化匹配结果、未匹配轨迹和未匹配检测的列表
    matches, unmatched_tracks, unmatched_detections = [], [], []

    # 遍历所有的检测，如果某个检测没有被匹配，则加入未匹配检测列表
    for row, detection_idx in enumerate(detection_indices):
        if row not in row_indices:
            unmatched_detections.append(detection_idx)

    # 遍历所有的轨迹，如果某个轨迹没有被匹配，则加入未匹配轨迹列表
    for col, track_idx in enumerate(track_indices):
        if col not in col_indices:
            unmatched_tracks.append(track_idx)

    # 遍历匹配结果，将符合条件的配对加入 matches，否则加入未匹配列表
    for row, col in zip(row_indices, col_indices):
        detection_idx = detection_indices[row]
        track_idx = track_indices[col]
        # 如果某个匹配的代价超过最大距离阈值，视为未匹配
        if cost_matrix[row, col] > max_distance:
            unmatched_tracks.append(track_idx)
            unmatched_detections.append(detection_idx)
        else:
            # 匹配成功的轨迹和检测对
            matches.append((detection_idx, track_idx))

    # 返回匹配结果、未匹配的轨迹和未匹配的检测
    return matches, unmatched_tracks, unmatched_detections


class Detection(object):
    def __init__(self, bbox, feature):
        self.bbox = np.asarray(bbox, dtype=np.float32)
        self.feature = np.asarray(feature, dtype=np.float32)

class Sort(object):
    def __init__(self, max_age=40, min_hits=3, iou_threshold=0.001):
        """
    Sets key parameters for SORT
    """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0
        KalmanBoxTracker.count = 0

    def update(self, ori_img, dets=np.empty((0, 5))):
        """
        Params:
          dets - a numpy array of detections in the format [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
        Requires: this method must be called once for each frame even with empty detections (use np.empty((0, 5)) for frames without detections).
        Returns a similar array, where the last column is the object ID.

        NOTE: The number of objects returned may differ from the number of detections provided.
        """

        # 去除无效对象
        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]
        valid_indices = (y2 < ori_img.shape[0]) & (x2 < ori_img.shape[1]) & (y1 >= 0) & (x1 >= 0)
        dets = dets[valid_indices]

        detections_features = extract_features(dets[:, :4].astype(int), ori_img)
        detections = [Detection(det, detections_features[i]) for i, det in enumerate(dets)]

        # 预测追踪器
        self.frame_count += 1
        # get predicted locations from existing trackers.
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        ret = []
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()[0]
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)

        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)

        # 返回匹配、未匹配的索引
        matched_a, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets, trks, self.iou_threshold)
        # 尝试二次匹配
        matched_b, unmatched_dets, unmatched_trks = cascade_matching(unmatched_dets, unmatched_trks, detections, self.trackers)
        # 结果拼接
        if len(matched_b) > 0:
            matched = np.concatenate((matched_a, matched_b))
        else:
            matched = matched_a

        # update matched trackers with assigned detections
        for m in matched:
            # feature = extract_features([dets[m[0], :4].astype(int)], ori_img)
            self.trackers[m[1]].update(dets[m[0], :], detections[m[0]].feature)

        # create and initialise new trackers for unmatched detections
        for i in unmatched_dets:
            # feature = extract_features([dets[i, :4].astype(int)], ori_img)
            trk = KalmanBoxTracker(dets[i, :], detections[i].feature)
            self.trackers.append(trk)
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.get_state()[0]
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                ret.append(np.concatenate((d, [trk.id + 1])).reshape(1, -1))  # +1 as MOT benchmark requires positive
            i -= 1
            # remove dead tracklet
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)
        if len(ret) > 0:
            return np.concatenate(ret)
        return np.empty((0, 5))


# =====================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
import cv2
import logging


class Extractor(object):
    def __init__(self, model_path, use_cuda=True):
        self.net = Net(reid=True)
        self.device = "cuda" if torch.cuda.is_available() and use_cuda else "cpu"
        state_dict = torch.load(model_path, map_location=lambda storage, loc: storage)['net_dict']
        self.net.load_state_dict(state_dict)
        logger = logging.getLogger("root.tracker")
        logger.info("Loading weights from {}... Done!".format(model_path))
        self.net.to(self.device)
        self.size = (64, 128)
        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def _preprocess(self, im_crops):
        def _resize(im, size):
            return cv2.resize(im.astype(np.float32) / 255., size)

        im_batch = torch.cat([self.norm(_resize(im, self.size)).unsqueeze(0) for im in im_crops], dim=0).float()
        return im_batch

    def __call__(self, im_crops):
        im_batch = self._preprocess(im_crops)
        with torch.no_grad():
            im_batch = im_batch.to(self.device)
            features = self.net(im_batch)
        return features.cpu().numpy()


class BasicBlock(nn.Module):
    def __init__(self, c_in, c_out, is_downsample=False):
        super(BasicBlock, self).__init__()
        self.is_downsample = is_downsample
        if is_downsample:
            self.conv1 = nn.Conv2d(c_in, c_out, 3, stride=2, padding=1, bias=False)
        else:
            self.conv1 = nn.Conv2d(c_in, c_out, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(c_out)
        self.relu = nn.ReLU(True)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(c_out)
        if is_downsample:
            self.downsample = nn.Sequential(
                nn.Conv2d(c_in, c_out, 1, stride=2, bias=False),
                nn.BatchNorm2d(c_out)
            )
        elif c_in != c_out:
            self.downsample = nn.Sequential(
                nn.Conv2d(c_in, c_out, 1, stride=1, bias=False),
                nn.BatchNorm2d(c_out)
            )
            self.is_downsample = True

    def forward(self, x):
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.relu(y)
        y = self.conv2(y)
        y = self.bn2(y)
        if self.is_downsample:
            x = self.downsample(x)
        return F.relu(x.add(y), True)


def make_layers(c_in, c_out, repeat_times, is_downsample=False):
    blocks = []
    for i in range(repeat_times):
        if i == 0:
            blocks += [BasicBlock(c_in, c_out, is_downsample=is_downsample), ]
        else:
            blocks += [BasicBlock(c_out, c_out), ]
    return nn.Sequential(*blocks)


class Net(nn.Module):
    def __init__(self, num_classes=2679, reid=False):
        super(Net, self).__init__()
        # 3 128 64
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # nn.Conv2d(32,32,3,stride=1,padding=1),
            # nn.BatchNorm2d(32),
            # nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, padding=1),
        )
        # 32 64 32
        self.layer1 = make_layers(64, 64, 2, False)
        # 32 64 32
        self.layer2 = make_layers(64, 128, 2, True)
        # 64 32 16
        self.layer3 = make_layers(128, 256, 2, True)
        # 128 16 8
        self.layer4 = make_layers(256, 512, 2, True)
        # 256 8 4
        self.avgpool = nn.AvgPool2d((8, 4), 1)
        # 256 1 1
        self.reid = reid
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        # B x 128
        if self.reid:
            x = x.div(x.norm(p=2, dim=1, keepdim=True))
            return x
        # classifier
        x = self.classifier(x)
        return x
