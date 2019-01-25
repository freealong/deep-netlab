import argparse
import torch
from model.metric import *

parser = argparse.ArgumentParser(description='Eval Detection')

parser.add_argument('-t', '--type', default='voc', type=str, required=True,
                    help='detection metric type: voc, coco')
parser.add_argument('-d', '--data', default=None, type=str, required=True,
                    help='preds and gts pth file')

args = parser.parse_args()

metric_func = None
if args.type == "voc":
    metric_func = eval_detection_voc
else:
    raise ValueError("invalid detection metric type {}".format(args.type))

data = torch.load(args.data)
preds, gts = data['preds'], data['gts']
metric = metric_func(preds, gts, use_07_metric=True)
print("{} detection metric:\n{}".format(args.type, metric.__str__()))
