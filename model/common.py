import math
from collections import OrderedDict
import torch
import torch.nn as nn

from utils import *


def make_conv_bn_relu_layers(in_channels, out_channels, kernel_size, stride=1,
                        padding=0, dilation=1, batch_norm=True, relu=True):
    layers = OrderedDict()
    layers['conv'] = nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                               padding, dilation)
    if batch_norm:
        layers['bn'] = nn.BatchNorm2d(out_channels)
    if relu:
        layers['relu'] = nn.ReLU(inplace=True)
    return nn.Sequential(layers)


# This function is derived from torchvision VGG make_layers()
# https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
def vgg_make_layers(cfg, in_channels, batch_norm=False):
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'C':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [nn.Sequential(OrderedDict([('conv', conv2d),
                                                      ('bn', nn.BatchNorm2d(v)),
                                                      ('relu', nn.ReLU(inplace=True))]))]
            else:
                layers += [nn.Sequential(OrderedDict([('conv', conv2d),
                                                      ('relu', nn.ReLU(inplace=True))]))]
            in_channels = v
    return nn.ModuleList(layers)


def vary_boxes_by_aspect_ratios(box, next_box, aspect_ratios):
    """
    generate prior boxes by different aspect ratios from ssd
    :param box: [w, h]
    :param next_box: [w, h]
    :param aspect_ratios: eg: [2, 3]
    :return: [[w, h], ...]
    """
    prior_boxes = []
    prior_boxes.append(box)
    prior_boxes.append([math.sqrt(box[0] * next_box[0]), math.sqrt(box[1] * next_box[1])])
    for ratio in aspect_ratios:
        sqrt_ratio = math.sqrt(ratio)
        prior_boxes.append([box[0] * sqrt_ratio, box[1] / sqrt_ratio])
        prior_boxes.append([box[0] / sqrt_ratio, box[1] * sqrt_ratio])
    return prior_boxes


def generate_prior_boxes_grid(box_sizes, grid_size, clip=True):
    """
    :param box_sizes: a list of prior box size, eg: [[0.1, 0.1], [0.2, 0.2], [w, h]]
    :param grid_size: feature map size, [h, w]
    :param clip: whether clip output tensor in [0, 1]
    :return: tensor : grid_h * grid_w * prior_box_num * [cx, cy, w, h]
    """
    cy, cx = torch.meshgrid([torch.arange(grid_size[0], dtype=torch.float32),
                             torch.arange(grid_size[1], dtype=torch.float32)])
    cy = cy.add(0.5).div(grid_size[0])
    cx = cx.add(0.5).div(grid_size[1])
    cy = cy.unsqueeze(2)
    cx = cx.unsqueeze(2)
    grids = []
    for box_size in box_sizes:
        box_size = torch.tensor(box_size, dtype=torch.float32)
        box_size = box_size.repeat(grid_size[0], grid_size[1], 1)
        grid = torch.cat([cx, cy, box_size], dim=2)
        grids.append(grid)
    grids = torch.cat(grids, dim=2).view(grid_size[0], grid_size[1], -1, 4)
    if clip:
        grids.clamp_(0, 1)
    return grids


def encode_boxes(gt, priors, variances=None):
    """
    encode locations from predictions using priors to undo the encoding we did for offset
    regression at training time, see https://github.com/rykov8/ssd_keras/issues/53 for more about variance
    :param gt: tensor, num_priors * [x1, y1, x2, y2]
    :param priors: tensor, num_priors * [cx, cy, w, h]
    :param variances: list[float], shape: 2
    :return: tensor, num_priors * [cx, cy, w, h]
    """
    if variances is None:
        ghat_cxcy = ((gt[:, :2] + gt[:, 2:]) / 2 - priors[:, :2]) / priors[:, 2:]
        ghat_wh = torch.log((gt[:, 2:] - gt[:, :2]) / priors[:, 2:])
    else:
        ghat_cxcy = ((gt[:, :2] + gt[:, 2:]) / 2 - priors[:, :2]) / (variances[0] * priors[:, 2:])
        ghat_wh = torch.log((gt[:, 2:] - gt[:, :2]) / priors[:, 2:]) / variances[1]
    return torch.cat([ghat_cxcy, ghat_wh], 1)


def decode_boxes(loc, priors, variances=None):
    """
    decode locations from predictions using priors to undo the encoding we did for offset
    regression at training time, see https://github.com/rykov8/ssd_keras/issues/53 for more about variance
    :param loc: tensor, shape: num_priors * [cx, cy, w, h]
    :param priors: tensor, shape: num_priors * [cx, cy, w, h]
    :param variances: list[float], shape: 2
    :return: tensor, num_priors * [x1, y1, x2, y2]
    """
    if variances is None:
        boxes = torch.cat([priors[:, :2] + loc[:, :2] * priors[:, 2:],
                           priors[:, 2:] * torch.exp(loc[:, 2:])], dim=1)
    else:
        boxes = torch.cat([priors[:, :2] + variances[0] * loc[:, :2] * priors[:, 2:],
                           priors[:, 2:] * torch.exp(variances[1] * loc[:, 2:])], dim=1)
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes


def transform_truths(truths, priors, variance=None, match_thresh=0.5):
    """
    transform ground truth to loc_gt, conf_gt
    :param truths: shape: batch_size * num_objects * [x1, y1, x2, y2, cls_id]
    :param priors: num_priors * [cx, cy, w, h]
    :param variance:
    :param match_thresh:
    :return: loc_gt: batch_size * num_priors * [cx, cy, w, h]
             conf_gt: batch_size * num_priors * num_classes
             matched_masks: batch_size * num_priors
    """
    batch_size = len(truths)
    num_priors = priors.size(0)
    loc_gt = torch.zeros(batch_size, num_priors, 4, device=priors.device, dtype=priors.dtype, requires_grad=False)
    conf_gt = torch.zeros(batch_size, num_priors, device=priors.device, dtype=torch.long, requires_grad=False)
    matched_masks = []
    for i in range(batch_size):
        box_gt = truths[i][:, :4]
        cls_gt = truths[i][:, 4].long()
        overlaps = compute_overlaps(box_gt, format_bbox(priors))
        best_truth_overlap, best_truth_idx = overlaps.max(dim=0)
        best_prior_idx = overlaps.argmax(dim=1)
        best_truth_idx[best_truth_overlap <= match_thresh] = -1
        best_truth_idx[best_prior_idx] = torch.arange(len(box_gt), dtype=best_truth_idx.dtype,
                                                      device=best_truth_idx.device)
        matched_mask = best_truth_idx > -1
        matched_value = best_truth_idx[matched_mask]
        conf_gt[i, matched_mask] = cls_gt[matched_value]
        loc_gt[i, matched_mask] = encode_boxes(box_gt[matched_value], priors[matched_mask], variance)
        matched_masks.append(matched_mask)
    return loc_gt, conf_gt, torch.cat(matched_masks)


def generate_detections(loc_data, conf_data, prior_data, variance=None, confidence=0.5, nms_threshold=0.5):
    """
    generate detections([x1, y1, x2, y2, cls, score]) from predictions
    :param loc_data: tensor: batch_size * num_priors * [cx, cy, w, h]
    :param conf_data: tensor: batch_size * num_priors * num_classes
    :param prior_data: tensor: num_priors * [cx, cy, w, h]
    :param variance:
    :param confidence:
    :param nms_threshold:
    :return: list of tensor: batch_size * num_detection * [x1, y1, x2, y2, cls, score]
    """
    batch_size = loc_data.size(0)

    detections = [torch.tensor([], dtype=loc_data.dtype, device=loc_data.device) for _ in range(batch_size)]
    for i in range(batch_size):
        # transform bbox into x1, y1, x2, y2
        decoded_boxes = decode_boxes(loc_data[i], prior_data, variance)
        # transform num of class scores into class index, and class score
        cls_score, cls_index = torch.max(conf_data[i], 1)
        cls_score = cls_score.unsqueeze(1)
        cls_index = cls_index.unsqueeze(1).float()
        detection = torch.cat([decoded_boxes, cls_index, cls_score], dim=1)
        # filter by confidence
        detection = detection[detection[:, 5] > confidence, :]
        if detection.shape[0] == 0:
            detections[i] = detection
            continue
        # get classes detected in the image
        img_classes = torch.unique(detection[:, 4])
        # do nms for each class
        for cls in img_classes:
            if cls == 0:
                continue
            detection_cls = detection[detection[:, 4] == cls, :]
            detection_filtered = run_nums(detection_cls, 5, nms_threshold)
            if detections[i].size(0) == 0:
                detections[i] = detection_filtered
            else:
                detections[i] = torch.cat([detections[i], detection_filtered])
    return detections
