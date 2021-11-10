import torch
import math
import torch.nn.functional as F
from torch import nn
import numpy as np
from utils.utils import xywh2xyxy
# from torch2trt import TRTModule
''' This file aims to provide how to calculate iou and its variations '''

# IoU Loss
# 输入xyxy
def iou(bboxes1, bboxes2):
    rows = bboxes1.shape[0]  # bboxes1.shape[0]，张量bboxes1中有几个box
    cols = bboxes2.shape[0]
    ious = torch.zeros((rows, cols))
    if rows * cols == 0:
        return ious
    exchange = False
    if bboxes1.shape[0] > bboxes2.shape[0]:
        bboxes1, bboxes2 = bboxes2, bboxes1
        ious = torch.zeros((cols, rows))
        exchange = True

    # 计算每个box的面积
    area1 = (bboxes1[:, 2] - bboxes1[:, 0]) * (
        bboxes1[:, 3] - bboxes1[:, 1])
    area2 = (bboxes2[:, 2] - bboxes2[:, 0]) * (
        bboxes2[:, 3] - bboxes2[:, 1])

    inter_max_xy = torch.min(bboxes1[:, 2:],bboxes2[:, 2:])  # x1，y1中的最大值
    inter_min_xy = torch.max(bboxes1[:, :2],bboxes2[:, :2])  # x2，y2中的最小值

    inter = torch.clamp((inter_max_xy - inter_min_xy), min=0)
    inter_area = inter[:, 0] * inter[:, 1]
    union = area1+area2-inter_area
    ious = inter_area / union
    ious = torch.clamp(ious,min=0,max = 1.0)
    if exchange:
        ious = ious.T
    return torch.sum(1-ious)

# GIoU Loss
# 输入xyxy
def giou(bboxes1, bboxes2):
    rows = bboxes1.shape[0]
    cols = bboxes2.shape[0]
    ious = torch.zeros((rows, cols))
    if rows * cols == 0:
        return ious
    exchange = False
    if bboxes1.shape[0] > bboxes2.shape[0]:
        bboxes1, bboxes2 = bboxes2, bboxes1
        ious = torch.zeros((cols, rows))
        exchange = True
    area1 = (bboxes1[:, 2] - bboxes1[:, 0]) * (
        bboxes1[:, 3] - bboxes1[:, 1])
    area2 = (bboxes2[:, 2] - bboxes2[:, 0]) * (
        bboxes2[:, 3] - bboxes2[:, 1])

    inter_max_xy = torch.min(bboxes1[:, 2:],bboxes2[:, 2:])

    inter_min_xy = torch.max(bboxes1[:, :2],bboxes2[:, :2])

    out_max_xy = torch.max(bboxes1[:, 2:],bboxes2[:, 2:])

    out_min_xy = torch.min(bboxes1[:, :2],bboxes2[:, :2])

    inter = torch.clamp((inter_max_xy - inter_min_xy), min=0)
    inter_area = inter[:, 0] * inter[:, 1]
    outer = torch.clamp((out_max_xy - out_min_xy), min=0)
    outer_area = outer[:, 0] * outer[:, 1]
    union = area1+area2-inter_area
    closure = outer_area

    ious = inter_area / union - (closure - union) / closure
    ious = torch.clamp(ious,min=-1.0,max = 1.0)
    if exchange:
        ious = ious.T
    return torch.sum(1-ious)

# DIoU Loss
# 输入xywh
def diou(bboxes1, bboxes2):
    # this is from official website:
    # https://github.com/Zzh-tju/CIoU/blob/master/layers/modules/multibox_loss.py
    bboxes1 = torch.sigmoid(bboxes1)        # make sure the input belongs to [0, 1]
    bboxes2 = torch.sigmoid(bboxes2)
    rows = bboxes1.shape[0]
    cols = bboxes2.shape[0]
    cious = torch.zeros((rows, cols))
    if rows * cols == 0:
        return cious
    exchange = False
    if bboxes1.shape[0] > bboxes2.shape[0]:
        bboxes1, bboxes2 = bboxes2, bboxes1
        cious = torch.zeros((cols, rows))
        exchange = True
    w1 = torch.exp(bboxes1[:, 2])       # this means this bbox has been encoded by log
    h1 = torch.exp(bboxes1[:, 3])       # you needn't do this if your bboxes are not encoded
    w2 = torch.exp(bboxes2[:, 2])
    h2 = torch.exp(bboxes2[:, 3])
    area1 = w1 * h1
    area2 = w2 * h2
    center_x1 = bboxes1[:, 0]
    center_y1 = bboxes1[:, 1]
    center_x2 = bboxes2[:, 0]
    center_y2 = bboxes2[:, 1]

    inter_l = torch.max(center_x1 - w1 / 2, center_x2 - w2 / 2)
    inter_r = torch.min(center_x1 + w1 / 2, center_x2 + w2 / 2)
    inter_t = torch.max(center_y1 - h1 / 2, center_y2 - h2 / 2)
    inter_b = torch.min(center_y1 + h1 / 2, center_y2 + h2 / 2)
    inter_area = torch.clamp((inter_r - inter_l),min=0) * torch.clamp((inter_b - inter_t),min=0)

    c_l = torch.min(center_x1 - w1 / 2, center_x2 - w2 / 2)
    c_r = torch.max(center_x1 + w1 / 2, center_x2 + w2 / 2)
    c_t = torch.min(center_y1 - h1 / 2, center_y2 - h2 / 2)
    c_b = torch.max(center_y1 + h1 / 2, center_y2 + h2 / 2)

    inter_diag = (center_x2 - center_x1)**2 + (center_y2 - center_y1)**2
    c_diag = torch.clamp((c_r - c_l), min=0)**2 + torch.clamp((c_b - c_t), min=0)**2

    union = area1+area2-inter_area
    u = (inter_diag) / c_diag
    iou = inter_area / union
    dious = iou - u
    dious = torch.clamp(dious, min=-1.0, max=1.0)
    if exchange:
        dious = dious.T
    return torch.sum(1 - dious)


# CIoU Loss
# 输入xywh
def ciou(bboxes1, bboxes2):
    # this is from official website:
    # https://github.com/Zzh-tju/CIoU/blob/master/layers/modules/multibox_loss.py
    # bboxes2 is ground truth
    bboxes1 = torch.sigmoid(bboxes1)
    bboxes2 = torch.sigmoid(bboxes2)
    rows = bboxes1.shape[0]
    cols = bboxes2.shape[0]
    cious = torch.zeros((rows, cols))
    if rows * cols == 0:
        return cious
    exchange = False
    if bboxes1.shape[0] > bboxes2.shape[0]:
        bboxes1, bboxes2 = bboxes2, bboxes1
        cious = torch.zeros((cols, rows))
        exchange = True
    w1 = torch.exp(bboxes1[:, 2])
    h1 = torch.exp(bboxes1[:, 3])
    w2 = torch.exp(bboxes2[:, 2])
    h2 = torch.exp(bboxes2[:, 3])
    area1 = w1 * h1
    area2 = w2 * h2
    center_x1 = bboxes1[:, 0]
    center_y1 = bboxes1[:, 1]
    center_x2 = bboxes2[:, 0]
    center_y2 = bboxes2[:, 1]

    inter_l = torch.max(center_x1 - w1 / 2, center_x2 - w2 / 2)
    inter_r = torch.min(center_x1 + w1 / 2, center_x2 + w2 / 2)
    inter_t = torch.max(center_y1 - h1 / 2, center_y2 - h2 / 2)
    inter_b = torch.min(center_y1 + h1 / 2, center_y2 + h2 / 2)
    inter_area = torch.clamp((inter_r - inter_l), min=0) * torch.clamp((inter_b - inter_t), min=0)

    c_l = torch.min(center_x1 - w1 / 2, center_x2 - w2 / 2)
    c_r = torch.max(center_x1 + w1 / 2, center_x2 + w2 / 2)
    c_t = torch.min(center_y1 - h1 / 2, center_y2 - h2 / 2)
    c_b = torch.max(center_y1 + h1 / 2, center_y2 + h2 / 2)

    inter_diag = (center_x2 - center_x1) ** 2 + (center_y2 - center_y1) ** 2
    c_diag = torch.clamp((c_r - c_l), min=0) ** 2 + torch.clamp((c_b - c_t), min=0) ** 2

    union = area1+area2-inter_area
    u = (inter_diag) / c_diag
    iou = inter_area / union
    v = (4 / (math.pi ** 2)) * torch.pow((torch.atan(w2 / h2) - torch.atan(w1 / h1)), 2)
    with torch.no_grad():
        S = (iou > 0.5).float()     # if iou < 0.5, the effect of aspect ratio is neglected
        alpha = S * v / (1 - iou + v)
    cious = iou - u - alpha * v
    cious = torch.clamp(cious, min=-1.0, max=1.0)
    if exchange:
        cious = cious.T
    return torch.sum(1 - cious)

# advanced iou-based nms:
# it'll be too slow if we use diou-nms directly. for advanced nms, please check
# https://arxiv.org/pdf/2005.03572.pdf


class LabelSmoothCELoss(nn.Module):
    def __init__(self):
        super(LabelSmoothCELoss, self).__init__()

    def forward(self, pred, label, num_classes, smoothing=0.1):
        pred = F.softmax(pred, dim=1)
        smoothed_one_hot_label = (1.0 - smoothing) * label + smoothing / num_classes
        loss = (-torch.log(pred + 1e-10)) * smoothed_one_hot_label
        loss = loss.sum(axis=1,  keepdim=False)
        loss = loss.mean()
        return loss


class IoULoss(nn.Module):
    def __init__(self):
        super(IoULoss, self).__init__()

    def forward(self, bboxes1, bboxes2, mode=1, xyxy=True):
        # mode=1 iou, mode=2 giou, mode=3 diou, mode=4 ciou
        rows = bboxes1.shape[0]  # bboxes1.shape[0]，张量bboxes1中有几个box
        cols = bboxes2.shape[0]
        ious = torch.zeros((rows, cols))
        if rows * cols == 0:
            return ious
        if xyxy == False:
            bboxes1 = xywh2xyxy(bboxes1)
            bboxes2 = xywh2xyxy(bboxes2)
        area1 = (bboxes1[:, 2] - bboxes1[:, 0]) * (bboxes1[:, 3] - bboxes1[:, 1])
        area2 = (bboxes2[:, 2] - bboxes2[:, 0]) * (bboxes2[:, 3] - bboxes2[:, 1])
        inter_max_xy = torch.min(bboxes1[:, None, 2:], bboxes2[:, 2:])
        inter_min_xy = torch.max(bboxes1[:, None, :2], bboxes2[:, :2])
        inter = (inter_max_xy - inter_min_xy).clamp(min=0).prod(2)
        union = area1[:, None] + area2 - inter
        ious = inter / union
        ious = torch.clamp(ious, min=0, max=1.0)

        if mode == 1:
            return torch.sum(1-ious) / (rows * cols)
        else:
            out_max_xy = torch.max(bboxes1[:, None, 2:], bboxes2[:, 2:])
            out_min_xy = torch.min(bboxes1[:, None, :2], bboxes2[:, :2])
            outer = (out_max_xy - out_min_xy).clamp(min=0).prod(2)
            if mode == 2:
                gious = ious - (outer - union) / outer
                gious = torch.clamp(gious, min=-1.0, max=1.0)
                return torch.sum(1 - gious) / (rows * cols)
            else:
                inter_diag = ((bboxes1[:, None, 0] + bboxes1[:, None, 2]) / 2 - (bboxes2[:, 0] + bboxes2[:, 2]) / 2) ** 2 \
                             + ((bboxes1[:, None, 1] + bboxes1[:, None, 3]) / 2 - (bboxes2[:, 1] + bboxes2[:, 3]) / 2) ** 2
                outer_diag = torch.clamp((out_max_xy[..., 0] - out_min_xy[..., 0]), min=0) ** 2 + \
                             torch.clamp((out_max_xy[..., 1] - out_min_xy[..., 1]), min=0) ** 2
                u = inter_diag / outer_diag
                if mode == 3:
                    dious = ious - u
                    dious = torch.clamp(dious, min=-1.0, max=1.0)
                    return torch.sum(1 - dious) / (rows * cols)
                if mode == 4:
                    w2 = bboxes2[:, 2] - bboxes2[:, 0]
                    h2 = bboxes2[:, 3] - bboxes2[:, 1]
                    w1 = bboxes1[:, None, 2] - bboxes1[:, None, 0]
                    h1 = bboxes1[:, None, 3] - bboxes1[:, None, 1]
                    v = (4 / (math.pi ** 2)) * torch.pow((torch.atan(w2 / h2) - torch.atan(w1 / h1)), 2)
                    with torch.no_grad():
                        S = (ious > 0.5).float()  # if iou < 0.5, the effect of aspect ratio is neglected
                        alpha = S * v / (1 - ious + v)
                    cious = ious - u - alpha * v
                    cious = torch.clamp(cious, min=-1.0, max=1.0)
                    return torch.sum(1 - cious) / (rows * cols)
