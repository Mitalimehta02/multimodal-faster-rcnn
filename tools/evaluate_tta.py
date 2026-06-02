"""
evaluate_flir_tta.py  —  Test-Time Augmentation evaluation for FLIR aligned.

Strategy
--------
For every test image we run N forward passes:
  1. Original scale
  2. Horizontally flipped (boxes are un-flipped after inference)
  3. Scaled × 0.90  (smaller context)
  4. Scaled × 1.10  (larger context)

All predictions are pooled and de-duplicated with Soft-NMS / standard NMS.
This typically adds +1–3 AP50 and +1–2 AP75 at no training cost.
"""

import argparse
import json
import os
import sys

import torch
import torch.nn.functional as F
import torchvision.ops as tv_ops
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from dataset.flir_aligned import FlirAlignedDataset
from utils.coco_detection_metrics import compute_coco_detection_metrics

PAPER_TARGETS = {"ap50": 79.20, "ap75": 37.40, "map_coco": 41.30}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="TTA evaluation for FLIR aligned")
    parser.add_argument("--config",      required=True,  help="FLIR YAML config")
    parser.add_argument("--checkpoint",  required=True,  help="Path to .pth checkpoint")
    parser.add_argument("--gpu",         default="0")
    parser.add_argument("--data_root",   default=None)
    parser.add_argument("--eval_score_thresh", type=float, default=0.05)
    parser.add_argument("--nms_thresh",  type=float, default=0.50,
                        help="NMS threshold when merging TTA predictions")
    parser.add_argument("--scales",      type=float, nargs="+",
                        default=[0.90, 1.00, 1.10],
                        help="Scale factors relative to config image_size")
    parser.add_argument("--hflip",       action="store_true", default=True,
                        help="Add horizontal-flip augmentation (default: True)")
    parser.add_argument("--no_hflip",    dest="hflip", action="store_false")
    parser.add_argument("--output_json", default=None)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def build_model(model_config, num_classes):
    from model.faster_rcnn import FasterRCNN
    return FasterRCNN(model_config=model_config, num_classes=num_classes)


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------
def collate_fn(batch):
    rgbs, irs, targets, ids = [], [], [], []
    for rgb, ir, target, img_id in batch:
        rgbs.append(rgb); irs.append(ir); targets.append(target); ids.append(img_id)
    return rgbs, irs, targets, ids


def build_test_dataset(ds_cfg, image_size=None):
    size = image_size or tuple(ds_cfg.get("image_size", [512, 640]))
    return FlirAlignedDataset(
        root=ds_cfg["root"],
        split_file=os.path.join(ds_cfg["root"], ds_cfg["test_split_file"])
                   if not os.path.isabs(ds_cfg["test_split_file"])
                   else ds_cfg["test_split_file"],
        rgb_template=ds_cfg.get("rgb_template", "JPEGImages/FLIR_{id}_RGB.jpg"),
        ir_template=ds_cfg.get("ir_template",   "JPEGImages/FLIR_{id}_PreviewData.jpeg"),
        ann_template=ds_cfg.get("ann_template",  "Annotations/FLIR_{id}_PreviewData.xml"),
        image_size=size,
        training=False,
        keep_empty=ds_cfg.get("keep_empty", True),
    )


# ---------------------------------------------------------------------------
# TTA helpers
# ---------------------------------------------------------------------------
def hflip_boxes(boxes, img_w):
    """Flip boxes horizontally: x1 <-> (W - x2), x2 <-> (W - x1)."""
    flipped = boxes.clone()
    flipped[:, 0] = img_w - boxes[:, 2]
    flipped[:, 2] = img_w - boxes[:, 0]
    return flipped


def resize_boxes(boxes, orig_size, new_size):
    """Scale boxes from orig_size (H,W) to new_size (H,W)."""
    sy = new_size[0] / orig_size[0]
    sx = new_size[1] / orig_size[1]
    scale = boxes.new_tensor([sx, sy, sx, sy])
    return boxes * scale


def merge_predictions(box_list, score_list, label_list, nms_thresh):
    """Concatenate all TTA predictions and apply per-class NMS."""
    if not box_list:
        return torch.zeros((0, 4)), torch.zeros(0), torch.zeros(0, dtype=torch.long)

    all_boxes  = torch.cat(box_list,  dim=0)
    all_scores = torch.cat(score_list, dim=0)
    all_labels = torch.cat(label_list, dim=0)

    keep_boxes, keep_scores, keep_labels = [], [], []
    for cls in all_labels.unique():
        mask = all_labels == cls
        b = all_boxes[mask]; s = all_scores[mask]
        keep = tv_ops.nms(b, s, nms_thresh)
        keep_boxes.append(b[keep])
        keep_scores.append(s[keep])
        keep_labels.append(torch.full((keep.shape[0],), cls.item(),
                                      dtype=torch.long, device=b.device))

    if not keep_boxes:
        return torch.zeros((0, 4)), torch.zeros(0), torch.zeros(0, dtype=torch.long)

    return torch.cat(keep_boxes), torch.cat(keep_scores), torch.cat(keep_labels)


# ---------------------------------------------------------------------------
# Single-image TTA forward
# ---------------------------------------------------------------------------
@torch.no_grad()
def tta_predict_batch(model, rgb_batch, ir_batch, base_size, scales, hflip,
                      eval_score_thresh, nms_thresh, device):
    """
    rgb_batch, ir_batch: list of tensors [C,H,W] at base_size (H,W).
    Returns list of dicts with keys 'boxes','scores','labels' (CPU tensors).
    """
    B = len(rgb_batch)
    orig_h, orig_w = base_size  # base size in (H, W)

    # Accumulate per-image predictions
    img_boxes  = [[] for _ in range(B)]
    img_scores = [[] for _ in range(B)]
    img_labels = [[] for _ in range(B)]

    # Build augmentation list: (scale, flip)
    augs = []
    for s in scales:
        augs.append((s, False))
        if hflip:
            augs.append((s, True))

    for (scale, flip) in augs:
        # Resize
        new_h = int(round(orig_h * scale))
        new_w = int(round(orig_w * scale))
        new_h = max(new_h, 32)
        new_w = max(new_w, 32)

        rgb_scaled = F.interpolate(
            torch.stack(rgb_batch).to(device),
            size=(new_h, new_w), mode="bilinear", align_corners=False
        )
        ir_scaled = F.interpolate(
            torch.stack(ir_batch).to(device),
            size=(new_h, new_w), mode="bilinear", align_corners=False
        )

        if flip:
            rgb_scaled = torch.flip(rgb_scaled, dims=[-1])
            ir_scaled  = torch.flip(ir_scaled,  dims=[-1])

        outputs = model(rgb_scaled, ir_scaled)

        for i, output in enumerate(outputs):
            boxes  = output["boxes"].cpu()
            scores = output["scores"].cpu()
            labels = output["labels"].cpu()

            mask = scores >= eval_score_thresh
            boxes, scores, labels = boxes[mask], scores[mask], labels[mask]

            if len(boxes) == 0:
                continue

            # Un-flip boxes
            if flip:
                boxes = hflip_boxes(boxes, new_w)

            # Scale boxes back to base_size
            if scale != 1.0:
                boxes = resize_boxes(boxes, (new_h, new_w), (orig_h, orig_w))

            # Clamp to image bounds
            boxes[:, 0].clamp_(0, orig_w)
            boxes[:, 1].clamp_(0, orig_h)
            boxes[:, 2].clamp_(0, orig_w)
            boxes[:, 3].clamp_(0, orig_h)

            img_boxes[i].append(boxes)
            img_scores[i].append(scores)
            img_labels[i].append(labels)

    results = []
    for i in range(B):
        boxes, scores, labels = merge_predictions(
            img_boxes[i], img_scores[i], img_labels[i], nms_thresh
        )
        results.append({"boxes": boxes, "scores": scores, "labels": labels})
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(args.config):
        print(f"\n[ERROR] Config file not found: '{args.config}'")
        sys.exit(1)

    with open(args.config, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    ds_cfg = dict(cfg["dataset_params"])
    if args.data_root:
        ds_cfg["root"] = os.path.abspath(args.data_root)

    # Path checks
    if not os.path.isdir(ds_cfg["root"]):
        print(f"\n[ERROR] Dataset root directory not found: '{ds_cfg['root']}'")
        print("Please check the 'root' path in your config YAML file or use the --data_root argument to specify the correct path.")
        sys.exit(1)

    split_file = os.path.join(ds_cfg["root"], ds_cfg["test_split_file"]) if not os.path.isabs(ds_cfg["test_split_file"]) else ds_cfg["test_split_file"]
    if not os.path.isfile(split_file):
        print(f"\n[ERROR] FLIR test split file not found: '{split_file}'")
        print("Please make sure your dataset root and split files are placed correctly.")
        sys.exit(1)

    if not os.path.isfile(args.checkpoint):
        print(f"\n[ERROR] Checkpoint file not found: '{args.checkpoint}'")
        print("Please specify a valid path to a trained model checkpoint (.pth).")
        sys.exit(1)

    base_size = tuple(ds_cfg.get("image_size", [512, 640]))   # (H, W)

    dataset = build_test_dataset(ds_cfg, image_size=base_size)
    loader  = DataLoader(
        dataset, batch_size=4, shuffle=False,
        num_workers=4, collate_fn=collate_fn,
        pin_memory=(device.type == "cuda"),
    )

    # Build & load model
    model = build_model(cfg.get("model_params", {}),
                        ds_cfg["num_classes"]).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    state_dict = ckpt.get("model_state_dict", ckpt)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[WARN] Missing keys: {len(missing)}")
    model.eval()

    # Override model score_thresh for eval
    try:
        model.detector.roi_heads.score_thresh = args.eval_score_thresh
    except AttributeError:
        pass

    print(f"\nTTA config")
    print(f"  Dataset : FLIR aligned")
    print(f"  Scales  : {args.scales}")
    print(f"  H-flip  : {args.hflip}")
    print(f"  NMS th  : {args.nms_thresh}")
    print(f"  Passes  : {len(args.scales) * (2 if args.hflip else 1)}")
    print(f"  Ckpt    : {args.checkpoint}\n")

    all_preds, all_gts = [], []

    for rgbs, irs, targets, _ in tqdm(loader, desc="TTA Eval", ncols=90):
        results = tta_predict_batch(
            model, rgbs, irs,
            base_size=base_size,
            scales=args.scales,
            hflip=args.hflip,
            eval_score_thresh=args.eval_score_thresh,
            nms_thresh=args.nms_thresh,
            device=device,
        )

        for result, target in zip(results, targets):
            preds = []
            for box, score, label in zip(result["boxes"], result["scores"], result["labels"]):
                preds.append([*box.tolist(), score.item(), int(label.item())])

            gts = []
            for box, label in zip(target["boxes"], target["labels"]):
                gts.append([*box.tolist(), int(label.item())])

            all_preds.append(preds)
            all_gts.append(gts)

    metrics = compute_coco_detection_metrics(
        all_preds=all_preds,
        all_gts=all_gts,
        num_classes=ds_cfg["num_classes"],
        score_thresh=0.50,
    )

    beats = (
        metrics["ap50"]     > PAPER_TARGETS["ap50"]
        and metrics["ap75"] > PAPER_TARGETS["ap75"]
        and metrics["map_coco"] > PAPER_TARGETS["map_coco"]
    )

    print("\n" + "=" * 72)
    print("FLIR aligned  —  TTA Evaluation")
    print("=" * 72)
    print(f"Checkpoint : {args.checkpoint}")
    print(f"Images     : {len(dataset)}")
    print(f"AP50       : {metrics['ap50']:.4f}")
    print(f"AP75       : {metrics['ap75']:.4f}")
    print(f"mAP        : {metrics['map_coco']:.4f}")
    print(f"Precision  : {metrics['precision']:.4f}")
    print(f"Recall     : {metrics['recall']:.4f}")
    print("-" * 72)
    print(f"CSSA Paper : AP50=79.20  AP75=37.40  mAP=41.30")
    print(
        f"Delta      : "
        f"AP50={metrics['ap50'] - PAPER_TARGETS['ap50']:+.2f}  "
        f"AP75={metrics['ap75'] - PAPER_TARGETS['ap75']:+.2f}  "
        f"mAP={metrics['map_coco'] - PAPER_TARGETS['map_coco']:+.2f}"
    )
    print(f"Status     : {'✅ BEATS PAPER TARGETS' if beats else '❌ DOES NOT BEAT ALL TARGETS'}")
    print("=" * 72)

    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as fh:
            json.dump({
                "checkpoint": args.checkpoint,
                "tta_scales": args.scales,
                "tta_hflip": args.hflip,
                "nms_thresh": args.nms_thresh,
                "metrics": metrics,
                "paper_targets": PAPER_TARGETS,
                "beats_paper": beats,
            }, fh, indent=2)
        print(f"JSON saved : {args.output_json}")


if __name__ == "__main__":
    main()
