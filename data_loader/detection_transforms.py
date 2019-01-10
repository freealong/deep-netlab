import numbers
import collections
from numpy import random
import torch
from torchvision import transforms
from torchvision.transforms import functional as F

from utils.util import calculate_bbox_iou


class Compose(transforms.Compose):
    def __call__(self, img, tgt=None):
        for t in self.transforms:
            if t.__call__.__code__.co_argcount > 2:
                img, tgt = t(img, tgt)
            else:
                img = t(img)
        return img, tgt


class RandomPad(object):
    def __init__(self, ratio, p=0.5, fill=0, padding_mode='constant'):
        assert isinstance(ratio, (numbers.Number, tuple))
        assert isinstance(fill, (numbers.Number, str, tuple))
        assert padding_mode in ['constant', 'edge', 'reflect', 'symmetric']
        if isinstance(ratio, collections.Sequence) and len(ratio) not in [2, 4]:
            raise ValueError("Padding must be an int or a 2, or 4 element tuple, not a " +
                             "{} element tuple".format(len(ratio)))
        self.ratio = ratio
        self.p = p
        self.fill = fill
        self.padding_mode = padding_mode

    def __call__(self, image, target):
        if self.p < random.random():
            return image, target
        # generate padding from ratio
        width, height = image.size
        if isinstance(self.ratio, numbers.Number):
            padding = (random.randint(0, int(self.ratio * width)), random.randint(0, int(self.ratio * height)))
        elif len(self.ratio) == 2:
            padding = (random.randint(0, int(self.ratio[0] * width)), random.randint(0, int(self.ratio[1] * height)))
        else:
            padding = (random.randint(0, int(self.ratio[0] * width)), random.randint(0, int(self.ratio[1] * height)),
                       random.randint(0, int(self.ratio[2] * width)), random.randint(0, int(self.ratio[3] * height)))
        padding_image = F.pad(image, padding, self.fill, self.padding_mode)
        padding_width, padding_height = padding_image.size
        # move target
        target = target.clone()
        left, top = padding[0], padding[1]
        target[:, 0] = (target[:, 0] * width + left) / padding_width
        target[:, 1] = (target[:, 1] * height + top) / padding_height
        target[:, 2] = (target[:, 2] * width + left) / padding_width
        target[:, 3] = (target[:, 3] * height + top) / padding_height
        return F.pad(image, padding, self.fill, self.padding_mode), target


class RandomHorizontalFlip(transforms.RandomHorizontalFlip):
   def __call__(self, img, tgt):
       if random.random() < self.p:
           flip_tgt = tgt.clone()
           flip_tgt[:, 0] = 1. - tgt[:, 2]
           flip_tgt[:, 2] = 1. - tgt[:, 0]
           return F.hflip(img), flip_tgt
       else:
           return img, tgt


class RandomSampleCrop(object):
    def __init__(self):
        self.sample_options = (
            # using entire original input image
            None,
            # sample a patch s.t. MIN jaccard w/ obj in .1,.3,.4,.7,.9
            (0.1, 1),
            (0.3, 1),
            (0.7, 1),
            (0.9, 1),
            # randomly sample a patch
            (0, 1),
        )

    def __call__(self, img, tgt):
        mode = random.choice(self.sample_options)
        if mode is None:
            return img, tgt
        width, height = img.size
        # max trails 50
        for _ in range(50):
            crop_width = int(random.uniform(0.3 * width, width))
            crop_height = int(random.uniform(0.3 * height, height))
            if crop_height / crop_width < 0.5 or crop_height / crop_width > 2:
                continue
            i, j, h, w = transforms.RandomCrop.get_params(img, (crop_height, crop_width))
            rect = torch.tensor([j / width, i / height, (j + w) / width, (i + h) / height])
            overlap = calculate_bbox_iou(rect, tgt[:, :4])
            if overlap.min() < mode[0] or overlap.max() > mode[1]:
                continue
            centers = (tgt[:, :2] + tgt[:, 2:4]) / 2.
            mask = (centers[:, 0] > rect[0]) & (centers[:, 1] > rect[1]) &\
                   (centers[:, 0] < rect[2]) & (centers[:, 1] < rect[3])
            if not mask.any():
                continue
            sample_tgt = tgt[mask, :].clone()
            sample_tgt[:, 0] = torch.max(sample_tgt[:, 0], rect[0])
            sample_tgt[:, 1] = torch.max(sample_tgt[:, 1], rect[1])
            sample_tgt[:, 2] = torch.min(sample_tgt[:, 2], rect[2])
            sample_tgt[:, 3] = torch.min(sample_tgt[:, 3], rect[3])
            sample_tgt[:, 0] -= rect[0]
            sample_tgt[:, 1] -= rect[1]
            sample_tgt[:, 2] -= rect[0]
            sample_tgt[:, 3] -= rect[1]
            sample_tgt[:, 0] *= width / w
            sample_tgt[:, 1] *= height / h
            sample_tgt[:, 2] *= width / w
            sample_tgt[:, 3] *= height / h
            return F.crop(img, i, j, h, w), sample_tgt
        return img, tgt


class SSDAugmentation(object):
    def __init__(self, size=(300, 300), mean=(104, 117, 123)):
        self.normal_mean = [x / 255. for x in mean]
        self.augment = Compose([
            RandomPad(ratio=0.5, fill=mean, p=0.5),
            RandomSampleCrop(),
            transforms.Resize(size=size),
            transforms.ColorJitter(brightness=0.125, contrast=0.5, saturation=0.5, hue=0.05),
            RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(self.normal_mean, (1., 1., 1.))
        ])

    def __call__(self, img, tgt):
        return self.augment(img, tgt)