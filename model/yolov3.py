from __future__ import division

import torch
import torch.nn as nn
import numpy as np
from utils.util import *


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

    prev_filters = int(net_info['channels'])
    output_filters = []

    module_list = nn.ModuleList()
    index = 0

    for block in blocks:
        module = nn.Sequential()
        filters = prev_filters

        if block['type'] == 'net':
            continue

        elif block['type'] == 'convolutional':
            # load params from block
            kernel_size = int(block['size'])
            stride = int(block['stride'])
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
            anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]
            anchors = [anchors[i] for i in mask]

            detection = YoloLayer(anchors)
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
    def __init__(self, anchors):
        super(YoloLayer, self).__init__()
        self.anchors = anchors
        self.num_anchors = len(self.anchors)

        self.ignore_threshold = 0.5
        self.lambda_cood = 5.0
        self.lambda_noobj = 0.5

        self.mse_loss = nn.MSELoss(reduction='elementwise_mean')
        self.bce_loss = nn.BCELoss(reduction='elementwise_mean')

    def forward(self, x, img_dim, target=None):
        '''
        transform yolo layer output into prediction
        :param x: yolo layer output in B*C*H*W shape
        :param anchors:
        :param img_dim: network input dim: [h, w]
        :param target: None or B*[cls, x, y, w, h]
        :return: prediction in B*num_bbox*bbox_attrs shape
        '''
        # Tensors for cuda support
        device = x.device
        batch_size = x.shape[0]
        stride = [img_dim[0] // x.shape[2], img_dim[1] // x.shape[3]]
        grid_size = [img_dim[0] // stride[0], img_dim[1] // stride[1]]
        anchors = [(a[0] / stride[1], a[1] / stride[0]) for a in self.anchors]
        anchors = torch.tensor(anchors, device=device)
        # bbox_attrs = len([x, y, w, h, objectness_conf) + num_classes, -1 = num_anchors * bbox_attrs
        x = x.view(batch_size, self.num_anchors, -1, grid_size[0], grid_size[1])
        x = x.permute(0, 1, 3, 4, 2).contiguous()  # batch_size * num_anchors * grid_height * grid_width * bbox_attrs
        # Sigmoid the  centre_X, centre_Y, object confidencce and class scores
        x[..., 0:2] = torch.sigmoid(x[..., 0:2])
        x[..., 4:] = torch.sigmoid(x[..., 4:])
        if target is not None:
            transformed_target = self.transform_target(x, target, anchors, device)
            losses = self.calculate_loss(x, transformed_target)
            # add dim to loss for muti-gpu training, details see https://github.com/NVIDIA/pix2pixHD/issues/42
            return [loss.unsqueeze(0) for loss in losses]
        # Add center offsets
        y_offset, x_offset = torch.meshgrid([torch.arange(grid_size[0], dtype=torch.float32, device=device),
                                             torch.arange(grid_size[1], dtype=torch.float32, device=device)])
        y_offset = y_offset.repeat(batch_size, self.num_anchors, 1, 1)
        x_offset = x_offset.repeat(batch_size, self.num_anchors, 1, 1)
        x[..., 0] += x_offset
        x[..., 1] += y_offset
        # log space transform height and the width
        grid_anchors = anchors.repeat(grid_size[0], grid_size[1], 1, 1).permute(2, 0, 1, 3).contiguous()
        x[..., 2] = torch.exp(x[..., 2]) * grid_anchors[..., 0]
        x[..., 3] = torch.exp(x[..., 3]) * grid_anchors[..., 1]
        # scale x, y, w, h
        x[..., 0] *= stride[1]
        x[..., 1] *= stride[0]
        x[..., 2] *= stride[1]
        x[..., 3] *= stride[0]
        return x.view(batch_size, self.num_anchors * grid_size[0] * grid_size[1], -1)

    def transform_target(self, x, target, anchors, device):
        mask = torch.zeros(x.shape[:-1], device=device)
        noobj_mask = torch.ones_like(mask, device=device)
        tx = torch.zeros_like(mask, device=device)
        ty = torch.zeros_like(mask, device=device)
        tw = torch.zeros_like(mask, device=device)
        th = torch.zeros_like(mask, device=device)
        tcls_shape = list(x.shape)
        tcls_shape[-1] -= 5
        tcls = torch.zeros(tcls_shape, device=device)

        for b in range(x.shape[0]):
            for t in range(len(target[b])):
                # Convert to position relative to box
                gx = target[b][t][1] * x.shape[3]
                gy = target[b][t][2] * x.shape[2]
                gw = target[b][t][3] * x.shape[3]
                gh = target[b][t][4] * x.shape[2]
                # Get grid box indices
                gi = int(gx)
                gj = int(gy)
                # Get shape of gt box
                gt_box = torch.tensor([0, 0, gw, gh], device=device)
                # Get shape of anchor box
                anchor_shapes = torch.cat((torch.zeros_like(anchors, device=device), anchors), 1)
                # Calculate iou between gt and anchor shapes
                anchor_ious = calculate_bbox_iou(gt_box, anchor_shapes, False)
                # Where the overlap is larger than threshold set mask to zero (ignore)
                noobj_mask[b, anchor_ious > self.ignore_threshold, gj, gi] = 0
                # Find the best matching anchor box
                best_n = torch.argmax(anchor_ious)
                # Masks
                mask[b, best_n, gj, gi] = 1
                # Coordinates
                tx[b, best_n, gj, gi] = gx - gi
                ty[b, best_n, gj, gi] = gy - gj
                # Width and height
                tw[b, best_n, gj, gi] = torch.log(gw / anchors[best_n][0] + 1e-16)
                th[b, best_n, gj, gi] = torch.log(gh / anchors[best_n][1] + 1e-16)
                # One-hot encoding of label
                tcls[b, best_n, gj, gi, int(target[b][t][0])] = 1

        return mask, noobj_mask, tx, ty, tw, th, tcls

    def calculate_loss(self, x, transformed_target):
        mask, noobj_mask, tx, ty, tw, th, tcls = transformed_target
        x_loss = self.lambda_cood * self.bce_loss(x[..., 0] * mask, tx * mask)
        y_loss = self.lambda_cood * self.bce_loss(x[..., 1] * mask, ty * mask)
        w_loss = self.lambda_cood * self.mse_loss(x[..., 2] * mask, tw * mask) / 2
        h_loss = self.lambda_cood * self.mse_loss(x[..., 3] * mask, th * mask) / 2
        conf_loss = self.bce_loss(x[..., 4] * mask, mask) + \
                    self.lambda_noobj * self.bce_loss(x[..., 4] * noobj_mask, noobj_mask * 0)
        cls_loss = self.bce_loss(x[..., 5:][mask == 1], tcls[mask == 1])
        loss = x_loss + y_loss + w_loss + h_loss + conf_loss + cls_loss
        return loss, x_loss, y_loss, w_loss, h_loss, conf_loss, cls_loss


class YoloNet(nn.Module):
    def __init__(self, cfgfile):
        super(YoloNet, self).__init__()
        self.blocks = parse_cfg(cfgfile)
        self.net_info, self.module_list = create_modules(self.blocks)

    def forward(self, x, target=None):
        is_training = target is not None
        img_dim = (x.shape[2], x.shape[3])
        outputs = {}
        predictions = []
        losses = []

        for i, module in enumerate(self.blocks[1:]):
            module_type = module['type']
            if module_type == 'convolutional' or module_type == 'upsample':
                x = self.module_list[i](x)
                # print(i, x.shape)
            elif module_type == 'route':
                layers = self.module_list[i][0].input
                if len(layers) == 1:
                    x = outputs[layers[0]]
                else:
                    x = torch.cat((outputs[layers[0]], outputs[layers[1]]), 1)
            elif module_type == 'shortcut':
                layers = self.module_list[i][0].input
                x = outputs[layers[0]] + outputs[layers[1]]
            elif module_type == 'yolo':
                x = self.module_list[i][0](x, img_dim, target)
                if is_training:
                    if len(losses) == 0:
                        losses = x
                    else:
                        losses = [a + b for a, b in zip(losses, x)]
                else:
                    if len(predictions) == 0:
                        predictions = x
                    else:
                        predictions = torch.cat((predictions, x), 1)
            outputs[i] = x

        # print([loss.item() for loss in losses])
        return losses if is_training else predictions

    def load_weights(self, weights):
        checkpoint = torch.load(weights)
        self.load_state_dict(checkpoint['state_dict'])

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
    # blocks = parse_cfg('cfg/yolov3-coco.cfg')
    # print('\n\n'.join([repr(x) for x in blocks]))
    #
    # net_info, module_list = create_modules(blocks)
    # print(net_info)
    # print(module_list)

    model = YoloNet("cfg/yolov3-coco.cfg")
    model.load_darknet_weights('darknet53.conv.74', 53)
    # model.load_darknet_weights('yolov3.weights')
    # input size should be divisible by 32
    input_size = [416, 608]
    using_cuda = torch.cuda.is_available()

    input_tensor, input_np = load_input_image("images/dog-cycle-car.png", input_size)
    if using_cuda:
        input_tensor = input_tensor.cuda()
        model = model.cuda()
    target_tensor = (torch.randn(1, 3, 5) + 10) / 20
    loss = model(input_tensor)
    print(loss)
