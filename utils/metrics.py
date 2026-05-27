"""
utils/metrics.py — Unified Detection Metrics

Computes:
  - AP@0.50         (PASCAL VOC style, 11-point interpolation)
  - AP@0.75
  - AP@0.90
  - mAP@0.50:0.95   (COCO style, mean over 10 thresholds)
  - Precision @ IoU=0.50
  - Recall    @ IoU=0.50

Also handles CSV logging of per-epoch metrics history.
"""

import os
import csv
import numpy as np
from typing import List, Dict, Tuple


# ---------------------------------------------------------------------------
# IoU / matching helpers
# ---------------------------------------------------------------------------

def compute_iou(box1: list, box2: list) -> float:
    """Compute Intersection-over-Union between two [x1,y1,x2,y2] boxes."""
    x1 = max(box1[0], box2[0]);  y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2]);  y2 = min(box1[3], box2[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    a1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    a2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = a1 + a2 - inter
    return float(inter / union) if union > 0 else 0.0


def compute_ap_at_threshold(
    all_preds: List[List],
    all_gts:   List[List],
    iou_thresh: float,
) -> Tuple[float, float, float]:
    """
    Compute AP at a single IoU threshold using 11-point interpolation.

    Args:
        all_preds : list of lists; each inner list contains [x1,y1,x2,y2,score]
                    for one image (sorted descending by score expected after merge)
        all_gts   : list of lists; each inner list contains [x1,y1,x2,y2]
        iou_thresh: IoU threshold for a detection to be counted as TP

    Returns:
        (ap, precision, recall)  — all in [0, 1]
    """
    # Flatten all detections across images, keeping image index
    detections = []
    total_gt = 0
    for img_id, (preds, gts) in enumerate(zip(all_preds, all_gts)):
        total_gt += len(gts)
        for p in preds:
            detections.append((img_id, p))

    if len(detections) == 0 or total_gt == 0:
        return 0.0, 0.0, 0.0

    # Sort all detections by confidence (descending)
    detections.sort(key=lambda x: x[1][4], reverse=True)

    tp = np.zeros(len(detections))
    fp = np.zeros(len(detections))
    matched = {i: np.zeros(len(all_gts[i])) for i in range(len(all_gts))}

    for idx, (img_id, pred) in enumerate(detections):
        best_iou, best_j = 0.0, -1
        for j, gt in enumerate(all_gts[img_id]):
            iou = compute_iou(pred[:4], gt)
            if iou > best_iou:
                best_iou, best_j = iou, j
        if best_iou >= iou_thresh and best_j >= 0 and matched[img_id][best_j] == 0:
            tp[idx] = 1
            matched[img_id][best_j] = 1
        else:
            fp[idx] = 1

    tp_cum  = np.cumsum(tp)
    fp_cum  = np.cumsum(fp)
    recalls    = tp_cum / max(total_gt, 1)
    precisions = tp_cum / np.maximum(tp_cum + fp_cum, 1e-8)

    # 11-point interpolated AP (PASCAL VOC)
    ap = 0.0
    for t in np.linspace(0, 1, 11):
        if np.any(recalls >= t):
            ap += np.max(precisions[recalls >= t])
    ap /= 11.0

    final_prec = float(precisions[-1]) if len(precisions) > 0 else 0.0
    final_rec  = float(recalls[-1])    if len(recalls)    > 0 else 0.0
    return float(ap), final_prec, final_rec


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_all_metrics(
    all_preds: List[List],
    all_gts:   List[List],
) -> Dict[str, float]:
    """
    Compute the full suite of detection metrics.

    Returns a dict with keys:
        ap50, ap75, ap90, map_coco (mAP@0.5:0.95), precision, recall
    """
    ap50,  prec50,  rec50  = compute_ap_at_threshold(all_preds, all_gts, 0.50)
    ap75,  _,       _      = compute_ap_at_threshold(all_preds, all_gts, 0.75)
    ap90,  _,       _      = compute_ap_at_threshold(all_preds, all_gts, 0.90)

    # mAP COCO-style: mean over [0.50, 0.55, ..., 0.95]
    coco_aps = []
    for t in np.arange(0.50, 1.00, 0.05):
        a, _, _ = compute_ap_at_threshold(all_preds, all_gts, round(t, 2))
        coco_aps.append(a)
    map_coco = float(np.mean(coco_aps))

    return {
        "ap50":      round(ap50  * 100, 4),
        "ap75":      round(ap75  * 100, 4),
        "ap90":      round(ap90  * 100, 4),
        "map_coco":  round(map_coco * 100, 4),
        "precision": round(prec50 * 100, 4),
        "recall":    round(rec50  * 100, 4),
    }


# ---------------------------------------------------------------------------
# CSV Logging
# ---------------------------------------------------------------------------

METRICS_COLUMNS = [
    "epoch", "split",
    "loss",
    "ap50", "ap75", "ap90", "map_coco",
    "precision", "recall",
    "lr",
]


def init_metrics_csv(csv_path: str) -> None:
    """Create CSV file with header row (call once at start of training)."""
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=METRICS_COLUMNS)
        writer.writeheader()


def append_metrics_csv(csv_path: str, row: dict) -> None:
    """Append one row (epoch metrics) to the CSV file."""
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=METRICS_COLUMNS)
        # Fill any missing fields with empty string
        full_row = {k: row.get(k, "") for k in METRICS_COLUMNS}
        writer.writerow(full_row)


class MetricsTracker:
    """
    Convenience class that wraps CSV init + append + in-memory history.

    Usage:
        tracker = MetricsTracker("outputs/exp1/metrics/metrics.csv")
        tracker.log(epoch=5, split="test", loss=0.12, ap50=89.3, ...)
    """
    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self.history: List[dict] = []
        init_metrics_csv(csv_path)

    def log(self, **kwargs) -> None:
        """Log one epoch; kwargs must contain at least 'epoch' and 'split'."""
        self.history.append(kwargs)
        append_metrics_csv(self.csv_path, kwargs)

    def best(self, key: str = "ap50", split: str = None) -> dict:
        """Return the epoch row with the highest value of `key`, optionally for one split."""
        rows = self.history
        if split is not None:
            rows = [row for row in rows if row.get("split") == split]
        if not rows:
            return {}
        return max(rows, key=lambda r: r.get(key, 0))
