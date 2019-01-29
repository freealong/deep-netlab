from __future__ import division

import torch
import torch.nn as nn
import numpy as np
from utils.util import *

from base import BaseModel
from model.common import *


def parse_cfg(cfgfile):
    """
    Takes a configuration file

    Returns a list of blocks. Each blocks describes a block in the neural
    network to be built. Block is represented as a dictionary in the list

    """
    file = open(cfgfile, 'r')
    lines = file.read().split('\n')  # store the lines in a list
    lines = [x for x in lines if len(x) > 0]  # get read of the empty lines
    lines = [x for x in lines if x[0] != '#']
    lines = [x.rstrip().lstrip() for x in lines]

    block = {}
    blocks = []

    for line in lines:
        if line[0] == "[":  # This marks the start of a new block
            if len(block) != 0:
                blocks.append(block)
                block = {}
            block["type"] = line[1:-1].rstrip()
        else:
            key, value = line.split("=")
            block[key.rstrip()] = value.lstrip()
    blocks.append(block)

    return blocks


def create_modules(blocks):
    net_info = blocks[0]
    input_size = np.array([int(net_info["height"]), int(net_info["width"])])

    prev_filters = int(net_info['channels'])
    output_filters = []

    module_list = nn.ModuleList()
    index = 0
    zoom = 1

    for block in blocks:
        module = nn.Sequential()
        filters = prev_filters

        if block['type'] == 'net':
            continue

        elif block['type'] == 'convolutional':
            # load params from block
            kernel_size = int(block['size'])
            stride = int(block['stride'])
            zoom *= stride
            padding = int(block['pad'])
            filters = int(block['filters'])
            activation = block['activation']
            if padding:
                pad = (kernel_size - 1) // 2
            else:
                pad = 0
            if 'batch_normalize' in block:
                batch_normalize = int(block['batch_normalize'])
                bias = False
            else:
                batch_normalize = 0
                bias = True

            # add conv layer
            conv = nn.Conv2d(prev_filters, filters, kernel_size, stride, pad, bias=bias)
            module.add_module("conv_{0}".format(index), conv)

            # add batch normalize layer
            if batch_normalize:
                bn = nn.BatchNorm2d(filters)
                module.add_module("batch_norm_{0}".format(index), bn)

            # add activation layer
            if activation == "leaky":
                activn = nn.LeakyReLU(0.1, inplace=True)
                module.add_module("leaky_{0}".format(index), activn)

        elif block['type'] == "upsample":
            stride = int(block['stride'])
            zoom //= stride
            upsample = nn.Upsample(scale_factor=stride)
            module.add_module("upsample_{0}".format(index), upsample)

        elif block['type'] == "route":
            layers = [int(x) for x in block['layers'].split(',')]
            filters = 0
            for i in range(len(layers)):
                if layers[i] < 0:
                    layers[i] += index
                filters += output_filters[layers[i]]

            route = EmptyLayer(layers)
            module.add_module("route_{0}".format(index), route)

        elif block['type'] == "shortcut":
            shortcut = EmptyLayer([index - 1, int(block['from']) + index])
            module.add_module("shortcut_{0}".format(index), shortcut)

        elif block['type'] == 'yolo':
            mask = [int(x) for x in block['mask'].split(',')]
            anchors = [int(x) for x in block['anchors'].split(',')]
            anchors = [(anchors[i] / input_size[1], anchors[i + 1] / input_size[0]) for i in range(0, len(anchors), 2)]
            anchors = [anchors[i] for i in mask]

            detection = YoloLayer(anchors, input_size // zoom)
            module.add_module("yolo_{0}".format(index), detection)

        index += 1
        module_list.append(module)
        prev_filters = filters
        output_filters.append(filters)

    return net_info, module_list


class EmptyLayer(nn.Module):
    def __init__(self, input=None):
        super(EmptyLayer, self).__init__()
        self.input = input


class YoloLayer(nn.Module):
    def __init__(self, anchors, grid_size):
        super(YoloLayer, self).__init__()
        self.prior_boxes = generate_prior_boxes_grid(anchors, grid_size, True)
        self.num_boxes = len(anchors) * grid_size[0] * grid_size[1]

    def forward(self, x):
        b = x.shape[0]
        x = x.permute(0, 2, 3, 1).contiguous().view(b, self.num_boxes, -1)
        # Sigmoid the  centre_X, centre_Y, object confidencce and class scores
        x[..., 0:2] = torch.sigmoid(x[..., 0:2])
        x[..., 4:] = torch.sigmoid(x[..., 4:])
        return x


class YoloV3(BaseModel):
    def __init__(self, params_cfg):
        super(YoloV3, self).__init__()
        self.blocks = parse_cfg(params_cfg)
        self.net_info, self.module_list = create_modules(self.blocks)
        self.prior_boxes = self._cat_prior_boxes()

        self.ignore_threshold = 0.5
        self.lambda_cood = 5.0
        self.lambda_noobj = 0.5

        self.mse_loss = nn.MSELoss(reduction='elementwise_mean')
        self.bce_loss = nn.BCELoss(reduction='elementwise_mean')

    def _cat_prior_boxes(self):
        prior_boxes_list = []
        for i, module in enumerate(self.blocks[1:]):
            module_type = module['type']
            if module_type == "yolo":
                prior_boxes_list.append(self.module_list[i][0].prior_boxes)
        return torch.cat([o.view(-1, 4) for o in prior_boxes_list])

    def forward(self, x):
        module_outputs = {}
        output = []

        for i, module in enumerate(self.blocks[1:]):
            module_type = module['type']
            if module_type == 'convolutional' or module_type == 'upsample':
                x = self.module_list[i](x)
                # print(i, x.shape)
            elif module_type == 'route':
                layers = self.module_list[i][0].input
                if len(layers) == 1:
                    x = module_outputs[layers[0]]
                else:
                    x = torch.cat((module_outputs[layers[0]], module_outputs[layers[1]]), 1)
            elif module_type == 'shortcut':
                layers = self.module_list[i][0].input
                x = module_outputs[layers[0]] + module_outputs[layers[1]]
            elif module_type == 'yolo':
                x = self.module_list[i][0](x)
                output.append(x)
            module_outputs[i] = x

        output = torch.cat(output, 1)
        self.prior_boxes = self.prior_boxes.to(x.device)
        return output

    def postprocess(self, output, objectness_thresh=0.5, conf_thresh=0.5, nms_thresh=0.45):
        batch_detections = []
        for x in output:
            mask = x[:, 4] > objectness_thresh
            x = x[mask] # filter by objectness threshold
            if len(x) == 0:
                batch_detections.append(x)
                continue
            loc, conf = x[:, :4], x[:, 5:]
            detections = generate_detections(loc, conf, self.prior_boxes[mask], confidence=conf_thresh)
            detections[:, 4] += 1 # cls + 1 to map class names with background
            detections = run_nms_cls(detections, nms_thresh)
            batch_detections.append(detections)
        return batch_detections

    def calculate_metric(self, output, target):
        batch_size = len(target)
        metric_avg = None
        valid_target = 0
        for i in range(batch_size):
            if len(target[i]) == 0:
                continue
            metric = compute_detection_metrics(output[i], target[i])
            if metric_avg is None:
                metric_avg = list(metric)
            else:
                metric_avg[1] += metric[1]
            valid_target += 1
        metric_avg[1] /= valid_target
        return metric_avg

    def calculate_loss(self, output, target):
        loc, obj, conf = output[:, :, :4], output[:, :, 4], output[:, :, 5:]
        loc_t, conf_t, best_mask, obj_mask = self._build_target(target, self.prior_boxes, self.ignore_threshold)
        obj_t = obj_mask.to(dtype=obj.dtype)
        noobj_mask = ~obj_mask
        one_hot_conf_t = torch.zeros_like(conf)
        one_hot_conf_t[best_mask, conf_t[best_mask] - 1] = 1
        x_loss = self.lambda_cood * self.bce_loss(loc[best_mask][:, 0], loc_t[best_mask][:, 0])
        y_loss = self.lambda_cood * self.bce_loss(loc[best_mask][:, 1], loc_t[best_mask][:, 1])
        w_loss = self.lambda_cood * self.mse_loss(loc[best_mask][:, 2], loc_t[best_mask][:, 2])
        h_loss = self.lambda_cood * self.mse_loss(loc[best_mask][:, 3], loc_t[best_mask][:, 3])
        conf_loss = self.bce_loss(obj[obj_mask], obj_t[obj_mask]) +\
                    self.bce_loss(obj[noobj_mask], obj_t[noobj_mask]) * self.lambda_noobj
        cls_loss = self.bce_loss(conf[best_mask], one_hot_conf_t[best_mask])
        loss = x_loss + y_loss + w_loss + h_loss + conf_loss + cls_loss
        return loss

    @staticmethod
    def _build_target(truth, priors, match_thresh):
        batch_size = len(truth)
        num_priors = priors.size(0)
        loc_gt = torch.zeros(batch_size, num_priors, 4, device=priors.device, dtype=priors.dtype, requires_grad=False)
        conf_gt = torch.zeros(batch_size, num_priors, device=priors.device, dtype=torch.long, requires_grad=False)
        matched_masks = torch.zeros(batch_size, num_priors, device=priors.device, dtype=torch.uint8, requires_grad=False)
        best_matched_masks = torch.zeros(batch_size, num_priors, device=priors.device, dtype=torch.uint8, requires_grad=False)
        for i in range(batch_size):
            box_gt = truth[i][:, :4]
            cls_gt = truth[i][:, 4].long()
            overlaps = compute_overlaps(box_gt, format_bbox(priors))
            # for each prior box select the best truth
            best_truth_overlap, best_truth_idx = overlaps.max(dim=0)
            # for each truth select the best prior box
            best_prior_idx = overlaps.argmax(dim=1)
            # for each prior box if its iou with any truth < thresh, set it's best truth id to -1
            best_truth_idx[best_truth_overlap <= match_thresh] = -1
            # for each prior box if its iou max than any other truth, set it's best truth id
            best_truth_idx[best_prior_idx] = torch.arange(len(box_gt), dtype=best_truth_idx.dtype,
                                                          device=best_truth_idx.device)
            matched_mask = best_truth_idx > -1
            matched_masks[i, matched_mask] = 1
            best_matched_masks[i, best_prior_idx] = 1
            best_matched_mask = best_matched_masks[i]

            best_matched_value = best_truth_idx[best_matched_mask]
            conf_gt[i, best_matched_mask] = cls_gt[best_matched_value]
            loc_gt[i, best_matched_mask] = encode_boxes(box_gt[best_matched_value], priors[best_matched_mask])
        return loc_gt, conf_gt, best_matched_masks, matched_masks

    def load_darknet_weights(self, weightfile, conv_num=10000):
        # Open the weights file
        fp = open(weightfile, "rb")

        # The first 4 values are header information
        # 1. Major version number
        # 2. Minor Version Number
        # 3. Subversion number
        # 4. IMages seen
        header = np.fromfile(fp, dtype=np.int32, count=5)
        self.header = torch.from_numpy(header)
        self.seen = self.header[3]

        # The rest of the values are the weights
        # Let's load them up
        weights = np.fromfile(fp, dtype=np.float32)

        conv_count = 0
        ptr = 0
        for i in range(len(self.module_list)):
            module_type = self.blocks[i + 1]["type"]

            if module_type == "convolutional":
                conv_count += 1
                if conv_count >= conv_num:
                    break
                model = self.module_list[i]
                try:
                    batch_normalize = int(self.blocks[i + 1]["batch_normalize"])
                except:
                    batch_normalize = 0

                conv = model[0]

                if (batch_normalize):
                    bn = model[1]

                    # Get the number of weights of Batch Norm Layer
                    num_bn_biases = bn.bias.numel()

                    # Load the weights
                    bn_biases = torch.from_numpy(weights[ptr:ptr + num_bn_biases])
                    ptr += num_bn_biases

                    bn_weights = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr += num_bn_biases

                    bn_running_mean = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr += num_bn_biases

                    bn_running_var = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr += num_bn_biases

                    # Cast the loaded weights into dims of model weights.
                    bn_biases = bn_biases.view_as(bn.bias.data)
                    bn_weights = bn_weights.view_as(bn.weight.data)
                    bn_running_mean = bn_running_mean.view_as(bn.running_mean)
                    bn_running_var = bn_running_var.view_as(bn.running_var)

                    # Copy the data to model
                    bn.bias.data.copy_(bn_biases)
                    bn.weight.data.copy_(bn_weights)
                    bn.running_mean.copy_(bn_running_mean)
                    bn.running_var.copy_(bn_running_var)

                else:
                    # Number of biases
                    num_biases = conv.bias.numel()

                    # Load the weights
                    conv_biases = torch.from_numpy(weights[ptr: ptr + num_biases])
                    ptr = ptr + num_biases

                    # reshape the loaded weights according to the dims of the model weights
                    conv_biases = conv_biases.view_as(conv.bias.data)

                    # Finally copy the data
                    conv.bias.data.copy_(conv_biases)

                # Let us load the weights for the Convolutional layers
                num_weights = conv.weight.numel()

                # Do the same as above for weights
                conv_weights = torch.from_numpy(weights[ptr:ptr + num_weights])
                ptr = ptr + num_weights

                conv_weights = conv_weights.view_as(conv.weight.data)

                conv.weight.data.copy_(conv_weights)


if __name__ == "__main__":
    from PIL import Image
    from utils.visualization import *
    from torchvision import transforms
    import cv2

    cfg_file = "../config/yolov3-voc.cfg"
    img_file = "../images/dog-cycle-car.png"
    weights_file = "../data/darknet53.conv.74"
    # weights_file = "../data/yolov3.weights"
    input_size = (416, 416)

    model = YoloV3(cfg_file)
    model.load_darknet_weights(weights_file, 53)
    torch.save(model.state_dict(), weights_file + ".pth")

    # input size should be divisible by 32
    img = cv2.imread(img_file)
    x = cv2.resize(img, input_size).astype(np.float32)
    x -= (104., 117., 123.)
    x = x.astype(np.float32)
    x = x[:, :, ::-1].copy()
    x = torch.from_numpy(x).permute(2, 0, 1)
    input = x.unsqueeze(0)

    # target
    target = [torch.tensor([[0.1, 0.1, 0.2, 0.2, 1], [0.2, 0.2, 0.3, 0.3, 20]])]

    if torch.cuda.is_available():
        input = input.cuda()
        model = model.cuda()
        target = [x.cuda() for x in target]
    output = model(input)

    # loss
    loss = model.calculate_loss(output, target)
    print(loss)

    try:
        output = model.postprocess(output)
    except AttributeError:
        pass
    print(output)
