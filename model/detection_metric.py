import numpy as np
from chainercv import evaluations


def eval_detection_voc(pred, gt, iou_thresh=0.5, use_07_metric=False):
    empty_array = np.empty([0,], dtype=np.float32)
    pred_bboxes = [x[:, :4] if len(x) > 0 else empty_array for x in pred]
    pred_labels = [x[:, 4] if len(x) > 0 else empty_array for x in pred]
    pred_scores = [x[:, 5] if len(x) > 0 else empty_array for x in pred]
    gt_bboxes = [x[:, :4] if len(x) > 0 else empty_array for x in gt]
    gt_labels = [x[:, 4] if len(x) > 0 else empty_array for x in gt]
    return evaluations.eval_detection_voc(pred_bboxes, pred_labels, pred_scores, gt_bboxes, gt_labels,
                                          iou_thresh=iou_thresh, use_07_metric=use_07_metric)
