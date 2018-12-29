import math
from collections import OrderedDict
import torch
import torch.nn as nn


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
    :param box: [x, y]
    :param next_box: [x, y]
    :param aspect_ratios: eg: [2, 3]
    :return: [[x, y], ...]
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
    :param grid_size: feature map size, [x, y]
    :param device: torch device
    :return: tensor with shape: [prior box num, grid_x, grid_y, 4]
    """
    prior_box_num = len(box_sizes)
    cx, cy = torch.meshgrid([torch.arange(grid_size[0], dtype=torch.float32, device=device),
                             torch.arange(grid_size[1], dtype=torch.float32, device=device)])
    cx.add_(0.5).div_(grid_size[0])
    cy.add_(0.5).div_(grid_size[0])
    cx = cx.repeat(prior_box_num, 1, 1).unsqueeze(3)
    cy = cy.repeat(prior_box_num, 1, 1).unsqueeze(3)
    box_sizes = torch.tensor(box_sizes, device=device, dtype=torch.float32)
    box_sizes = box_sizes.repeat(grid_size[0], grid_size[1], 1, 1).permute(2, 0, 1, 3)
    grid = torch.cat([cx, cy, box_sizes], dim=3)
    if clip:
        grid.clamp_(0, 1)
    return grid

def detection_layer(loc_data, conf_data, prior_data):
    """

    :param loc_data:
    :param conf_data:
    :param prior_data:
    :return:
    """
    batch_size = loc_data.size(0)