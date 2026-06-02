import argparse
import os
import sys
import time
from typing import Dict

import torch
import yaml
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from dataset.flir_aligned import FlirAlignedDataset
from dataset.voc import MultimodalVOCDataset
from utils.coco_detection_metrics import compute_coco_detection_metrics
from utils.logger import setup_logger
from utils.metrics import MetricsTracker
from utils.seed_utils import set_seed
from utils.visualizer import save_detection_visualizations


def parse_args():
    parser = argparse.ArgumentParser(description="Multimodal Faster R-CNN trainer (FLIR / LLVIP)")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--grad_accum_steps", type=int, default=None)
    parser.add_argument("--init_checkpoint", default=None, help="Optional checkpoint for model-weight initialization only")
    parser.add_argument("--data_root", default=None, help="Override dataset root")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--checkpoint_interval", type=int, default=10)
    parser.add_argument("--vis_images", type=int, default=12)
    parser.add_argument("--vis_interval", type=int, default=5)
    parser.add_argument("--score_thresh", type=float, default=None)
    return parser.parse_args()


def build_model(model_config: dict, num_classes: int):
    from model.faster_rcnn import FasterRCNN
    return FasterRCNN(model_config=model_config, num_classes=num_classes)


def collate_fn(batch):
    rgbs, irs, targets, ids = [], [], [], []
    for rgb, ir, target, img_id in batch:
        rgbs.append(rgb)
        irs.append(ir)
        targets.append(target)
        ids.append(img_id)
    return rgbs, irs, targets, ids


def _split_file(ds_cfg: Dict, split: str) -> str:
    key = f"{split}_split_file"
    split_file = ds_cfg[key]
    if os.path.isabs(split_file):
        return split_file
    return os.path.join(ds_cfg["root"], split_file)


def build_dataset(split: str, ds_cfg: Dict, training: bool):
    # Auto-detect LLVIP vs FLIR based on config keys
    if "rgb_train_path" in ds_cfg:
        # LLVIP-style config using MultimodalVOCDataset
        root = ds_cfg["root"]
        if split == "train":
            rgb_dir = os.path.join(root, ds_cfg["rgb_train_path"])
            ir_dir = os.path.join(root, ds_cfg["ir_train_path"])
            ann_dir = os.path.join(root, ds_cfg["ann_train_path"])
        else:
            rgb_dir = os.path.join(root, ds_cfg["rgb_test_path"])
            ir_dir = os.path.join(root, ds_cfg["ir_test_path"])
            ann_dir = os.path.join(root, ds_cfg["ann_test_path"])
        return MultimodalVOCDataset(
            split="train" if training else "test",
            rgb_dir=rgb_dir,
            ir_dir=ir_dir,
            ann_dir=ann_dir,
            image_size=tuple(ds_cfg.get("image_size", [512, 640])),
            scale_range=tuple(ds_cfg.get("scale_range", [0.7, 1.3])),
            brightness=ds_cfg.get("rgb_brightness_jitter", 0.30),
            contrast=ds_cfg.get("rgb_contrast_jitter", 0.30),
            hflip_prob=ds_cfg.get("hflip_prob", 0.5),
            cutout_prob=ds_cfg.get("cutout_prob", 0.0),
            cutout_scale=tuple(ds_cfg.get("cutout_scale", [0.05, 0.12])),
        )
    # FLIR-style config
    return FlirAlignedDataset(
        root=ds_cfg["root"],
        split_file=_split_file(ds_cfg, split),
        rgb_template=ds_cfg.get("rgb_template", "JPEGImages/FLIR_{id}_RGB.jpg"),
        ir_template=ds_cfg.get("ir_template", "JPEGImages/FLIR_{id}_PreviewData.jpeg"),
        ann_template=ds_cfg.get("ann_template", "Annotations/FLIR_{id}_PreviewData.xml"),
        image_size=tuple(ds_cfg.get("image_size", [512, 640])),
        training=training,
        scale_range=tuple(ds_cfg.get("scale_range", [0.85, 1.15])),
        rgb_brightness=ds_cfg.get("rgb_brightness_jitter", ds_cfg.get("brightness_jitter", 0.20)),
        rgb_contrast=ds_cfg.get("rgb_contrast_jitter", ds_cfg.get("contrast_jitter", 0.20)),
        rgb_saturation=ds_cfg.get("rgb_saturation_jitter", 0.0),
        rgb_hue=ds_cfg.get("rgb_hue_jitter", 0.0),
        rgb_random_grayscale_prob=ds_cfg.get("rgb_random_grayscale_prob", 0.0),
        thermal_brightness=ds_cfg.get("thermal_brightness_jitter", 0.0),
        thermal_contrast=ds_cfg.get("thermal_contrast_jitter", 0.0),
        hflip_prob=ds_cfg.get("hflip_prob", 0.5),
        cutout_prob=ds_cfg.get("cutout_prob", 0.15),
        cutout_scale=tuple(ds_cfg.get("cutout_scale", [0.05, 0.12])),
        keep_empty=ds_cfg.get("keep_empty", True),
    )


def make_scheduler(optimizer, num_epochs: int, warmup_epochs: int, min_lr: float, warmup_start_factor: float):
    if warmup_epochs <= 0 or warmup_epochs >= num_epochs:
        return CosineAnnealingLR(optimizer, T_max=max(num_epochs, 1), eta_min=min_lr)

    warmup = LinearLR(
        optimizer,
        start_factor=warmup_start_factor,
        end_factor=1.0,
        total_iters=warmup_epochs,
    )
    cosine = CosineAnnealingLR(
        optimizer,
        T_max=max(num_epochs - warmup_epochs, 1),
        eta_min=min_lr,
    )
    return SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs])


def format_group_lrs(optimizer):
    return ", ".join(f"{group['lr']:.2e}" for group in optimizer.param_groups)


def format_metric(metrics, name: str) -> str:
    if metrics is None:
        return "NA"
    return f"{metrics[name]:.2f}"


def set_detector_score_thresh(model, score_thresh: float):
    detector = getattr(model, "detector", None)
    roi_heads = getattr(detector, "roi_heads", None)
    if roi_heads is None or not hasattr(roi_heads, "score_thresh"):
        return None
    previous = roi_heads.score_thresh
    roi_heads.score_thresh = float(score_thresh)
    return previous


def set_finetune_stage(model, epoch: int, tr_cfg: Dict, optimizer=None, logger=None):
    freeze_after = tr_cfg.get("freeze_backbone_for_finetune_after")
    if freeze_after is None:
        return

    freeze_after = int(freeze_after)
    stage_b = epoch >= freeze_after
    for name, param in model.named_parameters():
        if stage_b:
            trainable = ("rpn" in name) or ("roi_heads" in name)
        else:
            trainable = True
        param.requires_grad = trainable

    if stage_b and epoch == freeze_after:
        if optimizer is not None:
            lr_factor = float(tr_cfg.get("stage_b_lr_factor", 0.5))
            for group in optimizer.param_groups:
                group["lr"] *= lr_factor
        if logger is not None:
            logger.info("Fine-tune stage: backbone/fusion frozen; training RPN and RoI heads only")


@torch.no_grad()
def evaluate(
    model,
    loader,
    device,
    num_classes: int,
    pr_score_thresh: float,
    eval_score_thresh: float,
    desc: str,
):
    model.eval()
    previous_score_thresh = set_detector_score_thresh(model, eval_score_thresh)
    all_preds, all_gts = [], []

    try:
        for rgbs, irs, targets, _ in tqdm(loader, desc=desc, leave=False, ncols=90):
            rgb = torch.stack(rgbs).to(device, non_blocking=True)
            ir = torch.stack(irs).to(device, non_blocking=True)
            outputs = model(rgb, ir)

            for output, target in zip(outputs, targets):
                preds = []
                for box, score, label in zip(output["boxes"], output["scores"], output["labels"]):
                    if score.item() >= eval_score_thresh:
                        preds.append([*box.detach().cpu().tolist(), score.item(), int(label.item())])

                gts = []
                for box, label in zip(target["boxes"], target["labels"]):
                    gts.append([*box.tolist(), int(label.item())])

                all_preds.append(preds)
                all_gts.append(gts)
    finally:
        if previous_score_thresh is not None:
            set_detector_score_thresh(model, previous_score_thresh)

    metrics = compute_coco_detection_metrics(
        all_preds=all_preds,
        all_gts=all_gts,
        num_classes=num_classes,
        score_thresh=pr_score_thresh,
    )
    model.train()
    return metrics


def main():
    args = parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(args.config, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)

    ds_cfg = dict(cfg["dataset_params"])
    if args.data_root:
        ds_cfg["root"] = os.path.abspath(args.data_root)

    # Path check: verify dataset root directory exists
    if not os.path.isdir(ds_cfg["root"]):
        print(f"\n[ERROR] Dataset root directory not found: '{ds_cfg['root']}'")
        print("Please check the 'root' path in your config YAML file or use the --data_root argument to specify the correct path.")
        sys.exit(1)

    # Path check: verify split files / subdirectories exist
    if "rgb_train_path" in ds_cfg:
        # LLVIP checks
        for folder_key in ["rgb_train_path", "ir_train_path", "ann_train_path"]:
            folder_path = os.path.join(ds_cfg["root"], ds_cfg[folder_key])
            if not os.path.isdir(folder_path):
                print(f"\n[ERROR] LLVIP directory not found: '{folder_path}'")
                print(f"Please verify that the '{folder_key}' path in your config exists under the dataset root.")
                sys.exit(1)
    else:
        # FLIR checks
        for split_key in ["train_split_file", "test_split_file"]:
            split_file = ds_cfg[split_key]
            split_path = os.path.join(ds_cfg["root"], split_file) if not os.path.isabs(split_file) else split_file
            if not os.path.isfile(split_path):
                print(f"\n[ERROR] FLIR split file not found: '{split_path}'")
                print("Please make sure your dataset root and split files are placed correctly.")
                sys.exit(1)

    tr_cfg = cfg["train_params"]
    model_cfg = cfg.get("model_params", {})

    num_epochs = args.epochs if args.epochs is not None else tr_cfg["num_epochs"]
    batch_size = args.batch_size if args.batch_size is not None else tr_cfg["batch_size"]
    lr = args.lr if args.lr is not None else tr_cfg["lr"]
    seed = args.seed if args.seed is not None else tr_cfg.get("seed", 42)
    num_workers = args.num_workers if args.num_workers is not None else tr_cfg.get("num_workers", min(8, os.cpu_count() or 4))
    pr_score_thresh = args.score_thresh if args.score_thresh is not None else tr_cfg.get(
        "precision_recall_score_thresh", tr_cfg.get("score_thresh", 0.50)
    )
    eval_score_thresh = float(tr_cfg.get("eval_score_thresh", model_cfg.get("box_score_thresh", 0.001)))
    vis_score_thresh = float(tr_cfg.get("vis_score_thresh", tr_cfg.get("score_thresh", 0.50)))

    warmup_epochs = int(tr_cfg.get("warmup_epochs", 5))
    min_lr = float(tr_cfg.get("min_lr", 1e-7))
    warmup_start_lr = float(tr_cfg.get("warmup_start_lr", 1e-7))
    warmup_start_factor = max(min(warmup_start_lr / max(lr, 1e-12), 1.0), 1e-4)
    freeze_low_level_epochs = int(tr_cfg.get("freeze_low_level_epochs", 0))
    freeze_backbone_epochs = int(tr_cfg.get("freeze_backbone_epochs", 0))
    val_ratio = float(tr_cfg.get("val_ratio", 0.15))
    weight_decay = float(tr_cfg.get("weight_decay", 1e-4))
    checkpoint_metric = str(tr_cfg.get("checkpoint_metric", "val_map")).lower()
    if checkpoint_metric not in {"val_map", "test_map", "val_ap75", "test_ap75", "test_ap50"}:
        raise ValueError(f"Unsupported checkpoint_metric: {checkpoint_metric}")
    grad_accum_steps = args.grad_accum_steps if args.grad_accum_steps is not None else tr_cfg.get("grad_accum_steps", 1)
    grad_accum_steps = max(1, int(grad_accum_steps))
    effective_batch = batch_size * grad_accum_steps

    out_root = args.output_dir or os.path.join("outputs", tr_cfg.get("task_name", "experiment"))
    ckpt_dir = os.path.join(out_root, "checkpoints")
    metric_dir = os.path.join(out_root, "metrics")
    vis_dir = os.path.join(out_root, "visualizations")
    log_path = os.path.join(out_root, "training.log")

    for directory in (ckpt_dir, metric_dir, vis_dir):
        os.makedirs(directory, exist_ok=True)

    logger = setup_logger("train_flir", log_path)
    logger.info("=" * 80)
    logger.info(f"Experiment   : {out_root}")
    logger.info(f"Dataset      : Multimodal (auto-detected from config)")
    logger.info(f"Model        : Multimodal Faster R-CNN")
    logger.info(f"Device       : {device}  GPU={args.gpu}")
    logger.info(f"AMP          : {args.amp and device.type == 'cuda'}")
    logger.info(f"Epochs       : {num_epochs}")
    logger.info(f"Batch size   : {batch_size}")
    logger.info(f"Grad accum   : {grad_accum_steps} step(s), effective batch={effective_batch}")
    logger.info(f"Base LR      : {lr:.2e}")
    logger.info(f"Weight decay : {weight_decay:.2e}")
    logger.info(f"Eval score   : {eval_score_thresh:.3f} candidate threshold for AP")
    logger.info(f"PR score     : {pr_score_thresh:.2f} for precision/recall")
    logger.info(f"Vis score    : {vis_score_thresh:.2f} for saved images")
    logger.info(f"Checkpoint   : best_model.pth selected by {checkpoint_metric}")
    logger.info(f"Seed         : {seed}")
    logger.info("=" * 80)

    set_seed(seed)

    train_aug = build_dataset("train", ds_cfg, training=True)
    train_plain = build_dataset("train", ds_cfg, training=False)
    test_set = build_dataset("test", ds_cfg, training=False)

    total_train = len(train_aug)
    if val_ratio > 0:
        n_val = max(1, int(round(val_ratio * total_train)))
        n_train = total_train - n_val
        generator = torch.Generator().manual_seed(seed)
        indices = torch.randperm(total_train, generator=generator).tolist()
        train_indices = indices[:n_train]
        val_indices = indices[n_train:]
        train_set = Subset(train_aug, train_indices)
        val_set = Subset(train_plain, val_indices)
    else:
        train_set = train_aug
        val_set = None

    logger.info(
        f"Dataset size : train={len(train_set):,}  val_from_train={len(val_set) if val_set is not None else 0:,}  "
        f"paper_test={len(test_set):,}"
    )

    pin_memory = device.type == "cuda"
    persistent_workers = num_workers > 0

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
    val_loader = None
    if val_set is not None:
        val_loader = DataLoader(
            val_set,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
        )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )

    model = build_model(model_cfg, num_classes=ds_cfg["num_classes"]).to(device)
    if args.init_checkpoint:
        checkpoint = torch.load(args.init_checkpoint, map_location=device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        logger.info(f"Init checkpoint: {args.init_checkpoint}")
        logger.info(f"Init missing keys={len(missing)} unexpected keys={len(unexpected)}")

    if hasattr(model, "get_param_groups"):
        optimizer = AdamW(model.get_param_groups(lr), weight_decay=weight_decay)
    else:
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    scheduler = make_scheduler(
        optimizer=optimizer,
        num_epochs=num_epochs,
        warmup_epochs=warmup_epochs,
        min_lr=min_lr,
        warmup_start_factor=warmup_start_factor,
    )
    scaler = GradScaler(enabled=args.amp and device.type == "cuda")
    tracker = MetricsTracker(os.path.join(metric_dir, "metrics.csv"))

    best_score = float("-inf")
    best_snapshot = {}
    total_train_start = time.time()
    last_freeze_state = None

    logger.info(
        f"{'Ep':>4} | {'Loss':>8} | {'LRs':>26} | {'Val mAP':>8} | "
        f"{'Val AP50':>8} | {'Test mAP':>8} | {'Test AP50':>9} | {'P':>6} | {'R':>6}"
    )

    for epoch in range(num_epochs):
        model.train()

        freeze_active = epoch < freeze_low_level_epochs
        if hasattr(model, "set_low_level_backbone_trainable"):
            model.set_low_level_backbone_trainable(not freeze_active)
        if freeze_active != last_freeze_state:
            logger.info("Backbone      : %s low-level stem/layer1", "freezing" if freeze_active else "unfreezing")
            last_freeze_state = freeze_active

        # Full backbone freeze/unfreeze for D2 pretrained models
        if hasattr(model, "set_backbone_trainable") and freeze_backbone_epochs > 0:
            if epoch < freeze_backbone_epochs:
                model.set_backbone_trainable(False)
                if epoch == 0:
                    logger.info("Backbone      : FROZEN for first %d epochs (D2 weights)", freeze_backbone_epochs)
            elif epoch == freeze_backbone_epochs:
                model.set_backbone_trainable(True)
                logger.info("Backbone      : UNFROZEN at epoch %d", epoch)

        set_finetune_stage(model, epoch, tr_cfg, optimizer=optimizer, logger=logger)

        total_loss = 0.0
        num_batches = 0
        epoch_start = time.time()

        optimizer.zero_grad(set_to_none=True)

        for batch_idx, (rgbs, irs, targets, _) in enumerate(tqdm(train_loader, desc=f"Ep{epoch:03d} train", leave=False, ncols=90)):
            rgb = torch.stack(rgbs).to(device, non_blocking=True)
            ir = torch.stack(irs).to(device, non_blocking=True)
            targets_dev = [{key: value.to(device) for key, value in target.items()} for target in targets]

            with autocast(enabled=args.amp and device.type == "cuda"):
                loss_dict = model(rgb, ir, targets_dev)
                loss = sum(loss_dict.values())
                scaled_loss = loss / grad_accum_steps

            if not torch.isfinite(loss):
                logger.warning("Non-finite loss at epoch %d, batch skipped", epoch)
                optimizer.zero_grad(set_to_none=True)
                continue

            scaler.scale(scaled_loss).backward()

            should_step = ((batch_idx + 1) % grad_accum_steps == 0) or ((batch_idx + 1) == len(train_loader))
            if should_step:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        scheduler.step()

        val_metrics = None
        if val_loader is not None:
            val_metrics = evaluate(
                model,
                val_loader,
                device,
                ds_cfg["num_classes"],
                pr_score_thresh=pr_score_thresh,
                eval_score_thresh=eval_score_thresh,
                desc=f"Ep{epoch:03d} val ",
            )
        test_metrics = evaluate(
            model,
            test_loader,
            device,
            ds_cfg["num_classes"],
            pr_score_thresh=pr_score_thresh,
            eval_score_thresh=eval_score_thresh,
            desc=f"Ep{epoch:03d} test",
        )
        group_lrs = format_group_lrs(optimizer)

        logger.info(
            f"{epoch:>4} | {avg_loss:>8.5f} | {group_lrs:>26} | "
            f"{format_metric(val_metrics, 'map_coco'):>8} | {format_metric(val_metrics, 'ap50'):>8} | "
            f"{test_metrics['map_coco']:>8.2f} | {test_metrics['ap50']:>9.2f} | "
            f"{test_metrics['precision']:>6.2f} | {test_metrics['recall']:>6.2f}"
        )
        logger.info(f"Epoch time   : {(time.time() - epoch_start) / 60.0:.2f} min")

        if val_metrics is not None:
            tracker.log(epoch=epoch, split="val", loss=round(avg_loss, 6), lr=group_lrs, **val_metrics)
        tracker.log(epoch=epoch, split="test", loss=round(avg_loss, 6), lr=group_lrs, **test_metrics)

        metric_source, metric_name = checkpoint_metric.split("_", 1)
        selected_metrics = val_metrics if metric_source == "val" else test_metrics
        if selected_metrics is None:
            raise ValueError(f"checkpoint_metric={checkpoint_metric} requires a validation split")
        selected_score = selected_metrics["map_coco" if metric_name == "map" else metric_name]

        if selected_score > best_score:
            best_score = selected_score
            best_snapshot = {"epoch": epoch, "val_metrics": val_metrics, "test_metrics": test_metrics}
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "val_metrics": val_metrics,
                    "test_metrics": test_metrics,
                    "class_names": train_aug.classes,
                },
                os.path.join(ckpt_dir, "best_model.pth"),
            )
            logger.info(f"Best checkpoint saved on {checkpoint_metric}={best_score:.2f}%")

        if args.checkpoint_interval > 0 and (epoch + 1) % args.checkpoint_interval == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "val_metrics": val_metrics,
                    "test_metrics": test_metrics,
                    "class_names": train_aug.classes,
                },
                os.path.join(ckpt_dir, f"epoch_{epoch + 1:03d}.pth"),
            )

        if (epoch + 1) % args.vis_interval == 0 or epoch == 0:
            save_detection_visualizations(
                model=model,
                dataset=val_set if val_set is not None else test_set,
                output_dir=vis_dir,
                epoch=epoch,
                device=device,
                max_images=args.vis_images,
                score_thresh=vis_score_thresh,
            )

    logger.info("=" * 80)
    logger.info(f"Training time : {(time.time() - total_train_start) / 3600.0:.2f} h")
    if best_snapshot:
        logger.info(f"Best epoch    : {best_snapshot['epoch']}")
        if best_snapshot["val_metrics"] is not None:
            logger.info(f"Best val mAP  : {best_snapshot['val_metrics']['map_coco']:.2f}%")
        logger.info(f"Test AP50     : {best_snapshot['test_metrics']['ap50']:.2f}%")
        logger.info(f"Test AP75     : {best_snapshot['test_metrics']['ap75']:.2f}%")
        logger.info(f"Test mAP      : {best_snapshot['test_metrics']['map_coco']:.2f}%")
        logger.info(f"Test precision: {best_snapshot['test_metrics']['precision']:.2f}%")
        logger.info(f"Test recall   : {best_snapshot['test_metrics']['recall']:.2f}%")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
