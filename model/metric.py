import torch
import numpy as np
from chainercv import evaluations


def my_metric(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)

def my_metric2(output, target, k=3):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)


def eval_detection_voc(pred, gt, iou_thresh=0.5, use_07_metric=False):
    empty_array = np.empty([0,], dtype=np.float32)
    pred_bboxes = [x[:, :4] if len(x) > 0 else empty_array for x in pred]
    pred_labels = [x[:, 4] if len(x) > 0 else empty_array for x in pred]
    pred_scores = [x[:, 5] if len(x) > 0 else empty_array for x in pred]
    gt_bboxes = [x[:, :4] if len(x) > 0 else empty_array for x in gt]
    gt_labels = [x[:, 4] if len(x) > 0 else empty_array for x in gt]
    return evaluations.eval_detection_voc(pred_bboxes, pred_labels, pred_scores, gt_bboxes, gt_labels,
                                          iou_thresh=iou_thresh, use_07_metric=use_07_metric)