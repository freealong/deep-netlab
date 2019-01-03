import json
import numpy as np
import torch
import torch.nn as nn

from model.common import *


class SSD(nn.Module):
    def __init__(self, phase, params_cfg):
        super(SSD, self).__init__()
        # load params
        self.phase = phase
        with open(params_cfg) as f:
            self.cfg = json.loads(f.read())
        self.num_classes = self.cfg['num_classes']
        self.image_dim = self.cfg['image_dim']
        self.aspect_ratios = self.cfg['aspect_ratios']
        self.feature_maps_dims = self.cfg['feature_maps_dim']
        self.min_sizes = self.cfg['min_sizes']
        self.max_sizes = self.cfg['max_sizes']
        self.variance = self.cfg['variance']
        self.clip = self.cfg['clip']
        # create net
        self.base = self.create_base(self.image_dim[-1])
        self.extra = self.create_extra(self.base[-1].conv.out_channels)
        self.loc, self.conf = self.create_head(self.base, self.extra, self.num_classes)
        self.L2Norm = L2Norm(512, 20)
        self.prior_boxes = self.create_prior_boxes(self.min_sizes, self.max_sizes, self.image_dim[:2],
                                                   self.aspect_ratios, self.feature_maps_dims, self.clip)
        if self.phase == "test":
            self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, conf_thresh=0.5, nms_thresh=0.5):
        batch_size = x.shape[0]
        detection_sources = list()
        loc = list()
        conf = list()
        # forward base to conv4_3
        for i in range(13):
            x = self.base[i](x)
        detection_sources.append(self.L2Norm(x))
        # forward base to fc7
        for i in range(13, len(self.base)):
            x = self.base[i](x)
        detection_sources.append(x)
        # forward extra
        for i, v in enumerate(self.extra):
            x = v(x)
            if i % 2 == 1:
                detection_sources.append(x)
        # forward head to detection_sources
        for s, l, c in zip(detection_sources, self.loc, self.conf):
            t = l(s)
            loc.append(l(s).permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 4))
            conf.append(c(s).permute(0, 2, 3, 1).contiguous().view(batch_size, -1, self.num_classes))
        loc = torch.cat(loc, 1)
        conf = torch.cat(conf, 1)
        if self.prior_boxes.device != x.device:
            self.prior_boxes = self.prior_boxes.to(x.device)
        # loc shape: batch_size * all_feature_size * 4
        # conf shape: batch_size * all_feature_size * num_classes
        # prior_boxes shape: all_feature_size * 4
        if self.phase == "test":
            conf = self.softmax(conf)
            output = generate_detections(loc, conf, self.prior_boxes, self.variance, conf_thresh, nms_thresh)
        else:
            output = (loc, conf, self.prior_boxes)
        return output

    @staticmethod
    def create_base(in_channels=3, batch_norm=False):
        cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
               512, 512, 512]
        layers = vgg_make_layers(cfg, in_channels=in_channels, batch_norm=batch_norm)
        pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        fc_relu6 = make_conv_bn_relu_layers(512, 1024, kernel_size=3, padding=6, dilation=6, batch_norm=False)
        fc_relu7 = make_conv_bn_relu_layers(1024, 1024, kernel_size=1, batch_norm=False)
        layers.extend([pool5, fc_relu6, fc_relu7])
        return layers

    @staticmethod
    def create_extra(in_channels, batch_norm=False):
        layers = []
        layers += [make_conv_bn_relu_layers(in_channels, 256, kernel_size=1, padding=0, stride=1, batch_norm=batch_norm)]
        layers += [make_conv_bn_relu_layers(256, 512, kernel_size=3, padding=1, stride=2, batch_norm=batch_norm)]
        layers += [make_conv_bn_relu_layers(512, 128, kernel_size=1, padding=0, stride=1, batch_norm=batch_norm)]
        layers += [make_conv_bn_relu_layers(128, 256, kernel_size=3, padding=1, stride=2, batch_norm=batch_norm)]
        layers += [make_conv_bn_relu_layers(256, 128, kernel_size=1, padding=0, stride=1, batch_norm=batch_norm)]
        layers += [make_conv_bn_relu_layers(128, 256, kernel_size=3, padding=0, stride=1, batch_norm=batch_norm)]
        layers += [make_conv_bn_relu_layers(256, 128, kernel_size=1, padding=0, stride=1, batch_norm=batch_norm)]
        layers += [make_conv_bn_relu_layers(128, 256, kernel_size=3, padding=0, stride=1, batch_norm=batch_norm)]
        return nn.ModuleList(layers)

    @staticmethod
    def create_head(base, extra, num_classes):
        loc_layers, conf_layers = [], []
        # base conv4_3
        loc_layers += [nn.Conv2d(base[12].conv.out_channels, 4 * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(base[12].conv.out_channels, 4 * num_classes, kernel_size=3, padding=1)]
        # base fc7
        loc_layers += [nn.Conv2d(base[-1].conv.out_channels, 6 * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(base[-1].conv.out_channels, 6 * num_classes, kernel_size=3, padding=1)]
        # extra conv8_2
        loc_layers += [nn.Conv2d(extra[1].conv.out_channels, 6 * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(extra[1].conv.out_channels, 6 * num_classes, kernel_size=3, padding=1)]
        # extra conv9_2
        loc_layers += [nn.Conv2d(extra[3].conv.out_channels, 6 * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(extra[3].conv.out_channels, 6 * num_classes, kernel_size=3, padding=1)]
        # extra conv10_2
        loc_layers += [nn.Conv2d(extra[5].conv.out_channels, 4 * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(extra[5].conv.out_channels, 4 * num_classes, kernel_size=3, padding=1)]
        # extra conv11_2
        loc_layers += [nn.Conv2d(extra[7].conv.out_channels, 4 * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(extra[7].conv.out_channels, 4 * num_classes, kernel_size=3, padding=1)]
        return nn.ModuleList(loc_layers), nn.ModuleList(conf_layers)

    @staticmethod
    def create_prior_boxes(min_sizes, max_sizes, image_dim, aspect_ratios, feature_maps_dims, clip):
        min_sizes = np.array([[s, s] for s in min_sizes])
        max_sizes = np.array([[s, s] for s in max_sizes])
        image_dim = np.array(image_dim)
        min_sizes = min_sizes / image_dim
        max_sizes =  max_sizes / image_dim
        prior_boxes = []
        for k, v in enumerate(feature_maps_dims):
            box_sizes = vary_boxes_by_aspect_ratios(min_sizes[k], max_sizes[k], aspect_ratios[k])
            prior_boxes.append(generate_prior_boxes_grid(box_sizes, v, clip=clip))
        prior_boxes = torch.cat([o.view(-1, 4) for o in prior_boxes])
        return prior_boxes


class L2Norm(nn.Module):
    def __init__(self,n_channels, scale):
        super(L2Norm,self).__init__()
        self.n_channels = n_channels
        self.gamma = scale or None
        self.eps = 1e-10
        self.weight = nn.Parameter(torch.Tensor(self.n_channels))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.constant(self.weight,self.gamma)

    def forward(self, x):
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt()+self.eps
        x = torch.div(x,norm)
        out = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x) * x
        return out


if __name__ == "__main__":
    from PIL import Image
    from utils.visualization import *
    from torchvision import transforms
    import cv2

    cfg_file = "../config/ssd300.json"
    img_file = "../images/dog-cycle-car.png"
    weights_file = "../weights.pth"

    # prepare model
    model = SSD('test', cfg_file)
    weights = torch.load(weights_file)
    model.load_state_dict(weights)


    # prepare input
    trans = transforms.Compose([
        transforms.Resize([300, 300]),
        transforms.Normalize((0.5, 0.5, 0.5), (1., 1., 1.)),
        transforms.ToTensor()
        ])
    # img = Image.open(img_file)
    # input = trans(img)
    img = cv2.imread(img_file)
    x = cv2.resize(img, (300, 300)).astype(np.float32)
    x -= (104., 117., 123.)
    x = x.astype(np.float32)
    x = x[:, :, ::-1].copy()
    x = torch.from_numpy(x).permute(2, 0, 1)
    input = x.unsqueeze(0)

    if torch.cuda.is_available():
        input = input.cuda()
        model = model.cuda()
    output = model.forward(input, 0.6)[0]

    print(output)
    output[:, 0] *= 602
    output[:, 1] *= 452
    output[:, 2] *= 602
    output[:, 3] *= 452

    draw_detections(img, output)
    cv2.imshow("im", img)
    cv2.waitKey()

