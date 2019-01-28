import os
import numpy as np
import torch


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def format_bbox(bboxes):
    return torch.cat((bboxes[:, :2] - bboxes[:, 2:] / 2,     # xmin, ymin
                     bboxes[:, :2] + bboxes[:, 2:] / 2), 1)  # xmax, ymax


def calculate_bbox_iou(bbox, bbox_list):
    '''
    calculate iou between a bbox and a list of bbox
    :param bbox: [x1, y1, x2, y2]
    :param bbox_list: [[x1, y1, x2, y2], ...]
    :return: a list of iou
    '''
    x1, y1, x2, y2 = bbox
    x1_list, y1_list, x2_list, y2_list = bbox_list[:, 0], bbox_list[:, 1], bbox_list[:, 2], bbox_list[:, 3]
    inter_x1 = torch.max(x1, x1_list)
    inter_y1 = torch.max(y1, y1_list)
    inter_x2 = torch.min(x2, x2_list)
    inter_y2 = torch.min(y2, y2_list)
    inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
    bbox_area = (x2 - x1) * (y2 - y1)
    bbox_list_area = (x2_list - x1_list) * (y2_list - y1_list)
    return inter_area.to(torch.float) / (bbox_area + bbox_list_area - inter_area).to(torch.float)


def compute_overlaps(boxes1, boxes2):
    """
    compute IoU overlaps between two sets of boxes.
    :param boxes1: [[x1, y1, x2, y2], [x1, y1, x2, y2], ...], shape: N*4
    :param boxes2: [[x1, y1, x2, y2], [x1, y1, x2, y2], ...], shape: M*4
    :return: shape: N*M
    """
    overlaps = torch.zeros([boxes1.shape[0], boxes2.shape[0]], device=boxes2.device)
    for i in range(boxes1.shape[0]):
        box1 = boxes1[i]
        overlaps[i, :] = calculate_bbox_iou(box1, boxes2)
    return overlaps


def run_nms(detections, thresh, sort_index=5):
    '''
    do non maximum suppresion
    :param detections: detections result [[x1,y1,x2,y2,...]]
    :param sort_index: which col in detections should we sort
    :param thresh: nms threshold
    :return: detections after nms
    '''
    index = torch.sort(detections[:, sort_index], descending=True)[1]
    detections = detections[index]
    for i in range(detections.shape[0]):
        try:
            ious = calculate_bbox_iou(detections[i][:4], detections[i + 1:, :4])
        except Exception:
            break
        detections = torch.cat((detections[:i + 1], detections[i + 1:][ious < thresh, :]))

    return detections


def run_nms_cls(detections, thresh, sort_index=5):
    if len(detections) == 0:
        return detections
    filtered_detections =[]
    # get classes detected in the image
    classes = torch.unique(detections[:, 4])
    # do nms for each class
    for cls in classes:
        detection_cls = detections[detections[:, 4] == cls, :]
        filtered_detections.append(run_nms(detection_cls, thresh, sort_index))
    return torch.cat(filtered_detections)


def compute_detection_metrics(pred, gt, iou_threshold=0.5):
    """Compute Average Precision at a set IoU threshold (default 0.5).
    Returns:
    mAP: Mean Average Precision
    precisions: List of precisions at different class score thresholds.
    recalls: List of recall values at different class score thresholds.
    overlaps: [pred_boxes, gt_boxes] IoU overlaps.
    """
    assert len(gt) > 0
    if len(pred) == 0:
        return ['precision', 'recall'], np.array([0., 0.])
    pred_boxes, pred_class_ids = pred[:, :4], pred[:, 4]
    gt_boxes, gt_class_ids = gt[:, :4], gt[:, 4]
    # Compute IoU overlaps [pred_boxes, gt_boxes]
    overlaps = compute_overlaps(pred_boxes, gt_boxes)
    correct_ids = pred_class_ids.view(overlaps.shape[0], -1).expand(overlaps.shape) ==\
                  gt_class_ids.expand(overlaps.shape)
    tp = ((overlaps > iou_threshold) & correct_ids).sum().to(torch.float)
    precision = tp / len(pred_boxes)
    recall = tp / len(gt_boxes)
    return ['precision', 'recall'], np.array([precision.item(), recall.item()])


if __name__ == "__main__":
    pred = torch.tensor([[100, 50, 200, 150, 0], [50, 100, 100, 200, 1]])
    gt = torch.tensor([[98, 50, 200, 150, 0], [55, 100, 100, 200, 1]])
    metrics = compute_detection_metrics(pred, gt)
    print(metrics)
