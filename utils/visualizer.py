"""
utils/visualizer.py — Detection Visualization

Draws bounding boxes, class labels, and confidence scores on images.
Saves high-resolution PNG files organised by epoch.

Color scheme:
  - GT boxes  : green  (#00FF00)
  - Pred boxes: red    (#FF4444) for person
  - Each class gets a distinct color if multi-class is used in future
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive backend (safe on remote server)
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import List, Optional

import torch
import torchvision.transforms.functional as TF


# ---------------------------------------------------------------------------
# Color palette for classes (0=background, 1=person, ...)
# ---------------------------------------------------------------------------
CLASS_COLORS = {
    0: "#888888",   # background (unused in viz)
    1: "#FF4444",   # person — red
    2: "#4488FF",   # car
    3: "#FFAA00",   # bicycle
}
CLASS_NAMES = {0: "bg", 1: "person", 2: "car", 3: "bicycle"}
GT_COLOR    = "#00CC44"   # green for ground-truth boxes


def _denorm_rgb(tensor: torch.Tensor) -> np.ndarray:
    """
    Reverse ImageNet normalisation and convert CHW tensor to HWC uint8 array.
    Safe to call even if tensor is already on CPU.
    """
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img  = tensor.cpu() * std + mean
    img  = img.clamp(0, 1).permute(1, 2, 0).numpy()
    return (img * 255).astype(np.uint8)


def draw_detections(
    rgb_tensor: torch.Tensor,
    pred_boxes:   torch.Tensor,
    pred_scores:  torch.Tensor,
    pred_labels:  torch.Tensor,
    gt_boxes:     Optional[torch.Tensor] = None,
    score_thresh: float = 0.30,
    title: str = "",
) -> plt.Figure:
    """
    Create a matplotlib figure with two subplots:
      Left : Ground-truth boxes (green)
      Right : Predicted  boxes (colored by class)

    Args:
        rgb_tensor  : CHW float32 normalised tensor
        pred_boxes  : [N, 4] predicted boxes (x1,y1,x2,y2)
        pred_scores : [N]    confidence scores
        pred_labels : [N]    class indices
        gt_boxes    : [M, 4] ground-truth boxes (optional)
        score_thresh: minimum confidence to draw a prediction
        title       : suptitle string

    Returns matplotlib Figure (caller must call plt.close(fig) after saving).
    """
    img_np = _denorm_rgb(rgb_tensor)

    fig, (ax_gt, ax_pred) = plt.subplots(1, 2, figsize=(20, 8), dpi=120)
    fig.patch.set_facecolor("#1A1A2E")

    for ax, label in zip((ax_gt, ax_pred), ("Ground Truth", "Predictions")):
        ax.imshow(img_np)
        ax.set_title(label, color="white", fontsize=14, fontweight="bold", pad=8)
        ax.axis("off")
        ax.set_facecolor("#1A1A2E")

    # --- Draw GT boxes ---
    gt_labels = None
    if isinstance(gt_boxes, tuple) and len(gt_boxes) == 2:
        gt_boxes, gt_labels = gt_boxes

    if gt_boxes is not None and len(gt_boxes) > 0:
        for gt_idx, box in enumerate(gt_boxes):
            x1, y1, x2, y2 = box.tolist() if hasattr(box, "tolist") else box
            gt_label = 1
            if gt_labels is not None:
                gt_label = int(gt_labels[gt_idx].item()) if hasattr(gt_labels[gt_idx], "item") else int(gt_labels[gt_idx])
            gt_name = CLASS_NAMES.get(gt_label, f"cls{gt_label}")
            rect = patches.FancyBboxPatch(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=2, edgecolor=GT_COLOR, facecolor="none",
                boxstyle="square,pad=0",
            )
            ax_gt.add_patch(rect)
            ax_gt.text(x1, max(y1 - 4, 0), gt_name,
                       color="white", fontsize=8, fontweight="bold",
                       bbox=dict(facecolor=GT_COLOR, alpha=0.75, pad=1.5, edgecolor="none"))
        ax_gt.set_title(f"Ground Truth  ({len(gt_boxes)} boxes)",
                        color="white", fontsize=14, fontweight="bold", pad=8)

    # --- Draw Predicted boxes ---
    keep_mask = pred_scores >= score_thresh
    p_boxes  = pred_boxes[keep_mask]
    p_scores = pred_scores[keep_mask]
    p_labels = pred_labels[keep_mask]

    for box, score, label in zip(p_boxes, p_scores, p_labels):
        x1, y1, x2, y2 = box.tolist()
        lbl   = int(label.item()) if hasattr(label, "item") else int(label)
        color = CLASS_COLORS.get(lbl, "#FF4444")
        name  = CLASS_NAMES.get(lbl, f"cls{lbl}")

        rect = patches.FancyBboxPatch(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2, edgecolor=color, facecolor="none",
            boxstyle="square,pad=0",
        )
        ax_pred.add_patch(rect)
        txt = f"{name} {score:.2f}"
        ax_pred.text(x1, max(y1 - 4, 0), txt,
                     color="white", fontsize=8, fontweight="bold",
                     bbox=dict(facecolor=color, alpha=0.80, pad=1.5, edgecolor="none"))

    n_pred = int(keep_mask.sum().item())
    ax_pred.set_title(f"Predictions  ({n_pred} boxes, conf≥{score_thresh:.2f})",
                      color="white", fontsize=14, fontweight="bold", pad=8)

    if title:
        fig.suptitle(title, color="white", fontsize=15, fontweight="bold", y=1.01)

    plt.tight_layout(pad=1.5)
    return fig


def save_detection_visualizations(
    model,
    dataset,
    output_dir: str,
    epoch: int,
    device: torch.device,
    max_images: int = 12,
    score_thresh: float = 0.30,
) -> None:
    """
    Run inference on `max_images` samples from `dataset` and save detection
    visualizations organized by epoch.

    Args:
        model      : the FasterRCNN model (set to eval mode inside)
        dataset    : MultimodalVOCDataset instance
        output_dir : root path like "outputs/exp1/visualizations"
        epoch      : current epoch number (used for folder name)
        device     : torch device
        max_images : how many validation images to visualise per epoch
        score_thresh: confidence threshold for drawing predictions
    """
    epoch_dir = os.path.join(output_dir, f"epoch_{epoch:03d}")
    os.makedirs(epoch_dir, exist_ok=True)

    model.eval()

    # Limit indices to min(max_images, len(dataset))
    indices = list(range(min(max_images, len(dataset))))

    with torch.no_grad():
        for idx in indices:
            try:
                rgb, ir, target, img_id = dataset[idx]

                rgb_batch = rgb.unsqueeze(0).to(device)
                ir_batch  = ir.unsqueeze(0).to(device)

                outputs = model(rgb_batch, ir_batch)
                out = outputs[0]

                pred_boxes  = out["boxes"].cpu()
                pred_scores = out["scores"].cpu()
                pred_labels = out["labels"].cpu()
                gt_boxes    = (target["boxes"], target.get("labels"))

                title = f"Image: {img_id}  |  Epoch {epoch:03d}"
                fig = draw_detections(
                    rgb_tensor=rgb,
                    pred_boxes=pred_boxes,
                    pred_scores=pred_scores,
                    pred_labels=pred_labels,
                    gt_boxes=gt_boxes,
                    score_thresh=score_thresh,
                    title=title,
                )

                save_path = os.path.join(epoch_dir, f"{img_id}.png")
                fig.savefig(save_path, dpi=150, bbox_inches="tight",
                            facecolor=fig.get_facecolor())
                plt.close(fig)

            except Exception as e:
                # Don't let viz errors crash training
                print(f"[Visualizer] Warning: skipped image {idx}: {e}")

    model.train()  # restore training mode
    print(f"[Visualizer] Saved {len(indices)} images → {epoch_dir}")
