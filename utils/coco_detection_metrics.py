from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


def box_iou(box1: Sequence[float], box2: Sequence[float]) -> float:
    x1 = max(float(box1[0]), float(box2[0]))
    y1 = max(float(box1[1]), float(box2[1]))
    x2 = min(float(box1[2]), float(box2[2]))
    y2 = min(float(box1[3]), float(box2[3]))

    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area1 = max(0.0, float(box1[2]) - float(box1[0])) * max(0.0, float(box1[3]) - float(box1[1]))
    area2 = max(0.0, float(box2[2]) - float(box2[0])) * max(0.0, float(box2[3]) - float(box2[1]))
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def _ap_from_pr(recalls: np.ndarray, precisions: np.ndarray) -> float:
    if recalls.size == 0:
        return 0.0

    recall_thresholds = np.linspace(0.0, 1.0, 101)
    ap = 0.0
    for threshold in recall_thresholds:
        mask = recalls >= threshold
        ap += float(np.max(precisions[mask])) if np.any(mask) else 0.0
    return ap / len(recall_thresholds)


def _class_ap(
    all_preds: List[List[List[float]]],
    all_gts: List[List[List[float]]],
    class_id: int,
    iou_thresh: float,
) -> float:
    detections: List[Tuple[int, List[float]]] = []
    gt_by_image: Dict[int, List[List[float]]] = {}

    total_gt = 0
    for image_idx, (preds, gts) in enumerate(zip(all_preds, all_gts)):
        class_gts = [gt for gt in gts if int(gt[4]) == class_id]
        gt_by_image[image_idx] = class_gts
        total_gt += len(class_gts)

        for pred in preds:
            if int(pred[5]) == class_id:
                detections.append((image_idx, pred))

    if total_gt == 0:
        return float("nan")
    if not detections:
        return 0.0

    detections.sort(key=lambda item: item[1][4], reverse=True)
    matched = {image_idx: np.zeros(len(gts), dtype=bool) for image_idx, gts in gt_by_image.items()}
    tp = np.zeros(len(detections), dtype=np.float32)
    fp = np.zeros(len(detections), dtype=np.float32)

    for det_idx, (image_idx, pred) in enumerate(detections):
        best_iou = 0.0
        best_gt_idx = -1
        for gt_idx, gt in enumerate(gt_by_image[image_idx]):
            iou = box_iou(pred[:4], gt[:4])
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        if best_iou >= iou_thresh and best_gt_idx >= 0 and not matched[image_idx][best_gt_idx]:
            tp[det_idx] = 1.0
            matched[image_idx][best_gt_idx] = True
        else:
            fp[det_idx] = 1.0

    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    recalls = tp_cum / max(total_gt, 1)
    precisions = tp_cum / np.maximum(tp_cum + fp_cum, 1e-8)
    return _ap_from_pr(recalls, precisions)


def _precision_recall_at_score(
    all_preds: List[List[List[float]]],
    all_gts: List[List[List[float]]],
    class_ids: Iterable[int],
    iou_thresh: float,
    score_thresh: float,
) -> Tuple[float, float]:
    total_tp = 0
    total_fp = 0
    total_gt = 0

    for class_id in class_ids:
        for preds, gts in zip(all_preds, all_gts):
            class_gts = [gt for gt in gts if int(gt[4]) == class_id]
            class_preds = [pred for pred in preds if int(pred[5]) == class_id and float(pred[4]) >= score_thresh]
            class_preds.sort(key=lambda pred: pred[4], reverse=True)

            total_gt += len(class_gts)
            matched = np.zeros(len(class_gts), dtype=bool)

            for pred in class_preds:
                best_iou = 0.0
                best_gt_idx = -1
                for gt_idx, gt in enumerate(class_gts):
                    iou = box_iou(pred[:4], gt[:4])
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx

                if best_iou >= iou_thresh and best_gt_idx >= 0 and not matched[best_gt_idx]:
                    total_tp += 1
                    matched[best_gt_idx] = True
                else:
                    total_fp += 1

    precision = total_tp / max(total_tp + total_fp, 1)
    recall = total_tp / max(total_gt, 1)
    return precision, recall


def compute_coco_detection_metrics(
    all_preds: List[List[List[float]]],
    all_gts: List[List[List[float]]],
    num_classes: int,
    score_thresh: float = 0.50,
) -> Dict[str, float]:
    """Compute paper-comparable detection metrics.

    Prediction format per image: [x1, y1, x2, y2, score, label].
    Ground-truth format per image: [x1, y1, x2, y2, label].
    Class 0 is background and is ignored.
    """
    class_ids = list(range(1, num_classes))
    thresholds = [round(x, 2) for x in np.arange(0.50, 1.00, 0.05)]

    ap_by_threshold: Dict[float, float] = {}
    per_class_ap50: Dict[int, float] = {}

    for threshold in thresholds:
        class_aps = []
        for class_id in class_ids:
            ap = _class_ap(all_preds, all_gts, class_id, threshold)
            if threshold == 0.50:
                per_class_ap50[class_id] = ap
            if not np.isnan(ap):
                class_aps.append(ap)

        ap_by_threshold[threshold] = float(np.mean(class_aps)) if class_aps else 0.0

    precision, recall = _precision_recall_at_score(
        all_preds=all_preds,
        all_gts=all_gts,
        class_ids=class_ids,
        iou_thresh=0.50,
        score_thresh=score_thresh,
    )

    metrics = {
        "ap50": round(ap_by_threshold[0.50] * 100.0, 4),
        "ap75": round(ap_by_threshold[0.75] * 100.0, 4),
        "map_coco": round(float(np.mean(list(ap_by_threshold.values()))) * 100.0, 4),
        "precision": round(precision * 100.0, 4),
        "recall": round(recall * 100.0, 4),
    }

    for class_id, ap in per_class_ap50.items():
        if not np.isnan(ap):
            metrics[f"ap50_class_{class_id}"] = round(ap * 100.0, 4)

    return metrics
