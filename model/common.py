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


def generate_prior_boxes_grid(box_sizes, grid_size, device=torch.device("cpu"), clip=True):
    """
    :param box_sizes: a list of prior box size, eg: [[0.1, 0.1], [0.2, 0.2]]
    :param grid_size: feature map size, [h, w]
    :param device: torch device
    :return: tensor with shape: [grid_h, grid_w, prior box num, 4]
    """
    cy, cx = torch.meshgrid([torch.arange(grid_size[0], dtype=torch.float32, device=device),
                             torch.arange(grid_size[1], dtype=torch.float32, device=device)])
    cy = cy.add(0.5).div(grid_size[0])
    cx = cx.add(0.5).div(grid_size[1])
    cy = cy.unsqueeze(2)
    cx = cx.unsqueeze(2)
    grids = []
    for box_size in box_sizes:
        box_size = torch.tensor(box_size, device=device, dtype=torch.float32)
        box_size = box_size.repeat(grid_size[0], grid_size[1], 1)
        grid = torch.cat([cx, cy, box_size], dim=2)
        grids.append(grid)
    grids = torch.cat(grids, dim=2).view(grid_size[0], grid_size[1], -1, 4)
    if clip:
        grids.clamp_(0, 1)
    return grids


def decode_boxes(loc, priors, variances):
    """
    decode locations from predictions using priors to undo the encoding we did for offset
    regression at training time, see https://github.com/rykov8/ssd_keras/issues/53 for more about variance
    :param loc: tensor, shape: [num_priors, 4]
    :param priors: tensor, shape: [num_priors, 4]
    :param variances: list[float], shape: 2
    :return:
    """
    boxes = torch.cat([priors[:, :2] + variances[0] * loc[:, :2] * priors[:, 2:],
                       priors[:, 2:] * torch.exp(variances[1] * loc[:, 2:])], dim=1)
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes


def generate_detections(loc_data, conf_data, prior_data, variance, confidence, iou_threshold=0.5):
    """
    generate detections([y1, x1, y2, x2, cls, score]) from predictions
    :param loc_data: tensor, shape: [batch_size, num_priors, 4]
    :param conf_data: tensor, shape: [batch_size, num_priors, num_classes]
    :param prior_data: tensor shape: [num_priors, 4]
    :return:
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
            detection_filtered = run_nums(detection_cls, 5, iou_threshold)
            if detections[i].size(0) == 0:
                detections[i] = detection_filtered
            else:
                detections[i] = torch.cat([detections[i], detection_filtered])
    return detections
