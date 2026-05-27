"""faster_rcnn_v9.py - V4 + Detectron2 COCO backbone + freeze/unfreeze."""
import math
import os
import pickle
from collections import OrderedDict
from typing import Dict, Iterable, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50
from torchvision.models.detection import FasterRCNN as TorchFasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import FeaturePyramidNetwork, MultiScaleRoIAlign, distance_box_iou_loss
from torchvision.ops.feature_pyramid_network import LastLevelMaxPool

D2_URL = ("https://dl.fbaipublicfiles.com/detectron2/"
          "COCO-Detection/faster_rcnn_R_50_FPN_3x/137849458/model_final_280758.pkl")
_LMAP = {"backbone.bottom_up.res2":"layer1","backbone.bottom_up.res3":"layer2",
         "backbone.bottom_up.res4":"layer3","backbone.bottom_up.res5":"layer4"}

def _map_d2(k):
    if k=="backbone.bottom_up.stem.conv1.weight": return "conv1.weight"
    if k.startswith("backbone.bottom_up.stem.conv1.norm."):
        return f"bn1.{k.split('backbone.bottom_up.stem.conv1.norm.')[-1]}"
    for d2p,tvl in _LMAP.items():
        if not k.startswith(d2p+"."): continue
        r=k[len(d2p)+1:]
        r=r.replace(".shortcut.norm.",".downsample.1.").replace(".shortcut.",".downsample.0.")
        for ci in ("1","2","3"): r=r.replace(f".conv{ci}.norm.",f".bn{ci}.")
        return f"{tvl}.{r}"
    return None

def _load_d2():
    d=os.path.join(os.path.expanduser("~"),".cache","d2_weights")
    os.makedirs(d,exist_ok=True); p=os.path.join(d,"model_final_280758.pkl")
    if not os.path.isfile(p):
        print(f"[D2] Downloading to {p}"); torch.hub.download_url_to_file(D2_URL,p)
    with open(p,"rb") as f: m=pickle.load(f,encoding="latin1")["model"]
    s={}
    for k,v in m.items():
        tk=_map_d2(k)
        if tk: s[tk]=torch.tensor(np.array(v)).float()
    return s


def sigmoid_focal_loss_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
    reduction: str = "sum",
) -> torch.Tensor:
    prob = torch.sigmoid(logits)
    ce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = prob * targets + (1.0 - prob) * (1.0 - targets)
    alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
    loss = alpha_t * ((1.0 - p_t) ** gamma) * ce_loss

    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    return loss


def generalized_box_iou_loss_fn(
    pred_boxes: torch.Tensor,
    target_boxes: torch.Tensor,
    reduction: str = "sum",
    eps: float = 1e-7,
) -> torch.Tensor:
    if pred_boxes.numel() == 0:
        return pred_boxes.sum() * 0.0

    pred_x1, pred_y1, pred_x2, pred_y2 = pred_boxes.unbind(dim=1)
    tgt_x1, tgt_y1, tgt_x2, tgt_y2 = target_boxes.unbind(dim=1)

    inter_x1 = torch.maximum(pred_x1, tgt_x1)
    inter_y1 = torch.maximum(pred_y1, tgt_y1)
    inter_x2 = torch.minimum(pred_x2, tgt_x2)
    inter_y2 = torch.minimum(pred_y2, tgt_y2)

    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    inter = inter_w * inter_h

    area_pred = (pred_x2 - pred_x1).clamp(min=0) * (pred_y2 - pred_y1).clamp(min=0)
    area_tgt = (tgt_x2 - tgt_x1).clamp(min=0) * (tgt_y2 - tgt_y1).clamp(min=0)
    union = area_pred + area_tgt - inter + eps
    iou = inter / union

    enc_x1 = torch.minimum(pred_x1, tgt_x1)
    enc_y1 = torch.minimum(pred_y1, tgt_y1)
    enc_x2 = torch.maximum(pred_x2, tgt_x2)
    enc_y2 = torch.maximum(pred_y2, tgt_y2)

    enc_area = (enc_x2 - enc_x1).clamp(min=0) * (enc_y2 - enc_y1).clamp(min=0) + eps
    giou = iou - ((enc_area - union) / enc_area)
    loss = 1.0 - giou

    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    return loss


def decode_boxes(rel_codes: torch.Tensor, boxes: torch.Tensor, weights: Optional[Iterable[float]] = None) -> torch.Tensor:
    if rel_codes.numel() == 0:
        return rel_codes.reshape(0, 4)

    wx, wy, ww, wh = weights if weights is not None else (10.0, 10.0, 5.0, 5.0)

    widths = boxes[:, 2] - boxes[:, 0]
    heights = boxes[:, 3] - boxes[:, 1]
    ctr_x = boxes[:, 0] + 0.5 * widths
    ctr_y = boxes[:, 1] + 0.5 * heights

    dx = rel_codes[:, 0] / wx
    dy = rel_codes[:, 1] / wy
    dw = rel_codes[:, 2] / ww
    dh = rel_codes[:, 3] / wh

    dw = torch.clamp(dw, max=math.log(1000.0 / 16.0))
    dh = torch.clamp(dh, max=math.log(1000.0 / 16.0))

    pred_ctr_x = dx * widths + ctr_x
    pred_ctr_y = dy * heights + ctr_y
    pred_w = torch.exp(dw) * widths
    pred_h = torch.exp(dh) * heights

    pred_boxes = torch.zeros_like(rel_codes)
    pred_boxes[:, 0] = pred_ctr_x - 0.5 * pred_w
    pred_boxes[:, 1] = pred_ctr_y - 0.5 * pred_h
    pred_boxes[:, 2] = pred_ctr_x + 0.5 * pred_w
    pred_boxes[:, 3] = pred_ctr_y + 0.5 * pred_h
    return pred_boxes


def sanitize_xyxy_boxes(boxes: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    x1 = torch.minimum(boxes[:, 0], boxes[:, 2])
    y1 = torch.minimum(boxes[:, 1], boxes[:, 3])
    x2 = torch.maximum(boxes[:, 0], boxes[:, 2])
    y2 = torch.maximum(boxes[:, 1], boxes[:, 3])
    x2 = torch.maximum(x2, x1 + eps)
    y2 = torch.maximum(y2, y1 + eps)
    return torch.stack((x1, y1, x2, y2), dim=1)


class ChannelAttention(nn.Module):
    def __init__(self, in_channels: int, reduction: int = 16):
        super().__init__()
        mid = max(in_channels // reduction, 4)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, in_channels, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.mlp(self.avg_pool(x)) + self.mlp(self.max_pool(x))
        gate = torch.sigmoid(gate).unsqueeze(-1).unsqueeze(-1)
        return x * gate


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = x.mean(dim=1, keepdim=True)
        mx = x.max(dim=1, keepdim=True).values
        gate = torch.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))
        return x * gate


class CBAM(nn.Module):
    def __init__(self, in_channels: int, reduction: int = 16):
        super().__init__()
        self.channel_attn = ChannelAttention(in_channels, reduction=reduction)
        self.spatial_attn = SpatialAttention(kernel_size=7)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.spatial_attn(self.channel_attn(x))


class LearnableWeightedFusion(nn.Module):
    def __init__(self, init_alpha: float = 0.5):
        super().__init__()
        init_logit = math.log(init_alpha / max(1.0 - init_alpha, 1e-6))
        self.logit = nn.Parameter(torch.tensor(float(init_logit)))

    def forward(self, rgb_feat: torch.Tensor, ir_feat: torch.Tensor) -> torch.Tensor:
        alpha = torch.sigmoid(self.logit)
        return alpha * rgb_feat + (1.0 - alpha) * ir_feat


class CBAMFusionBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.weighted_sum = LearnableWeightedFusion(init_alpha=0.5)
        self.cbam = CBAM(channels)
        self.out_proj = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, rgb_feat: torch.Tensor, ir_feat: torch.Tensor) -> torch.Tensor:
        fused = self.weighted_sum(rgb_feat, ir_feat)
        fused = self.cbam(fused)
        return self.out_proj(fused)


class CrossAttentionFusion(nn.Module):
    def __init__(self, in_ch: int = 2048, attn_dim: int = 256, num_heads: int = 8):
        super().__init__()
        if attn_dim % num_heads != 0:
            raise ValueError(f"attn_dim ({attn_dim}) must be divisible by num_heads ({num_heads})")

        self.base_fusion = LearnableWeightedFusion(init_alpha=0.5)
        self.attn_mix = LearnableWeightedFusion(init_alpha=0.5)
        self.proj_q = nn.Linear(in_ch, attn_dim, bias=False)
        self.proj_k = nn.Linear(in_ch, attn_dim, bias=False)
        self.proj_v = nn.Linear(in_ch, attn_dim, bias=False)
        self.mha = nn.MultiheadAttention(attn_dim, num_heads, batch_first=True, dropout=0.0)
        self.proj_out = nn.Linear(attn_dim, in_ch, bias=False)
        self.norm = nn.LayerNorm(in_ch)
        self.cbam = CBAM(in_ch)
        self.out_proj = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, rgb_feat: torch.Tensor, ir_feat: torch.Tensor) -> torch.Tensor:
        batch_size, channels, height, width = rgb_feat.shape

        rgb_seq = rgb_feat.flatten(2).permute(0, 2, 1)
        ir_seq = ir_feat.flatten(2).permute(0, 2, 1)

        q = self.proj_q(rgb_seq)
        k = self.proj_k(ir_seq)
        v = self.proj_v(ir_seq)

        attn_out, _ = self.mha(q, k, v)
        attn_out = self.proj_out(attn_out)
        attn_out = self.norm(attn_out + rgb_seq)
        attn_out = attn_out.permute(0, 2, 1).reshape(batch_size, channels, height, width)

        base = self.base_fusion(rgb_feat, ir_feat)
        fused = self.attn_mix(attn_out, base)
        fused = self.cbam(fused)
        return self.out_proj(fused)


class DualResNetBackbone(nn.Module):
    def __init__(self, pretrained: bool = True, freeze_backbone: bool = False):
        super().__init__()
        rgb = resnet50(weights=None)
        ir = resnet50(weights=None)
        if pretrained:
            d2 = _load_d2()
            m1,_ = rgb.load_state_dict(d2, strict=False)
            print(f"[D2->RGB] loaded {len(d2)} keys, missing={len(m1)}")
            m2,_ = ir.load_state_dict(d2, strict=False)
            print(f"[D2->IR]  loaded {len(d2)} keys, missing={len(m2)}")
        self.rgb_stem = nn.Sequential(rgb.conv1, rgb.bn1, rgb.relu, rgb.maxpool)
        self.rgb_layer1 = rgb.layer1
        self.rgb_layer2 = rgb.layer2
        self.rgb_layer3 = rgb.layer3
        self.rgb_layer4 = rgb.layer4
        ir_conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            ir_conv1.weight.copy_(ir.conv1.weight.mean(dim=1, keepdim=True))
        self.ir_stem = nn.Sequential(ir_conv1, ir.bn1, ir.relu, ir.maxpool)
        self.ir_layer1 = ir.layer1
        self.ir_layer2 = ir.layer2
        self.ir_layer3 = ir.layer3
        self.ir_layer4 = ir.layer4
        if freeze_backbone:
            self._freeze_all()

    def _all_mods(self):
        return [self.rgb_stem,self.rgb_layer1,self.rgb_layer2,self.rgb_layer3,self.rgb_layer4,
                self.ir_stem,self.ir_layer1,self.ir_layer2,self.ir_layer3,self.ir_layer4]
    def _freeze_all(self):
        for m in self._all_mods():
            for p in m.parameters(): p.requires_grad=False
            m.eval()
    def _unfreeze_all(self):
        for m in self._all_mods():
            for p in m.parameters(): p.requires_grad=True
            m.train()
    def train(self, mode=True):
        super().train(mode)
        for m in self._all_mods():
            fp=next(m.parameters(),None)
            if fp is not None and not fp.requires_grad: m.eval()
        return self

    def forward(self, rgb: torch.Tensor, ir: torch.Tensor) -> Dict[str, torch.Tensor]:
        rgb = self.rgb_stem(rgb)
        rgb_c1 = self.rgb_layer1(rgb)
        rgb_c2 = self.rgb_layer2(rgb_c1)
        rgb_c3 = self.rgb_layer3(rgb_c2)
        rgb_c4 = self.rgb_layer4(rgb_c3)
        ir = self.ir_stem(ir)
        ir_c1 = self.ir_layer1(ir)
        ir_c2 = self.ir_layer2(ir_c1)
        ir_c3 = self.ir_layer3(ir_c2)
        ir_c4 = self.ir_layer4(ir_c3)
        return {"rgb_c1":rgb_c1,"rgb_c2":rgb_c2,"rgb_c3":rgb_c3,"rgb_c4":rgb_c4,
                "ir_c1":ir_c1,"ir_c2":ir_c2,"ir_c3":ir_c3,"ir_c4":ir_c4}


class MultimodalBackboneWithFPN(nn.Module):
    def __init__(self, fpn_out: int = 256, pretrained_backbone: bool = True, freeze_backbone: bool = False):
        super().__init__()
        self.body = DualResNetBackbone(pretrained=pretrained_backbone, freeze_backbone=freeze_backbone)
        self.fusion_c1 = CBAMFusionBlock(256)
        self.fusion_c2 = CBAMFusionBlock(512)
        self.fusion_c3 = CBAMFusionBlock(1024)
        self.fusion_c4 = CrossAttentionFusion(in_ch=2048, attn_dim=256, num_heads=8)
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=[256, 512, 1024, 2048],
            out_channels=fpn_out,
            extra_blocks=LastLevelMaxPool(),
        )
        self.out_channels = fpn_out

    def forward(self, rgb: torch.Tensor, ir: torch.Tensor) -> Dict[str, torch.Tensor]:
        feats = self.body(rgb, ir)

        fused = OrderedDict(
            [
                ("0", self.fusion_c1(feats["rgb_c1"], feats["ir_c1"])),
                ("1", self.fusion_c2(feats["rgb_c2"], feats["ir_c2"])),
                ("2", self.fusion_c3(feats["rgb_c3"], feats["ir_c3"])),
                ("3", self.fusion_c4(feats["rgb_c4"], feats["ir_c4"])),
            ]
        )
        return self.fpn(fused)


class FasterRCNN(nn.Module):
    def __init__(self, model_config: Optional[Dict] = None, num_classes: int = 2):
        super().__init__()
        model_config = model_config or {}
        rpn_cfg = model_config.get("rpn", {})
        roi_cfg = model_config.get("roi_head", {})

        def cfg_value(name: str, default):
            if name.startswith("rpn_"):
                short = name.replace("rpn_", "")
                if short in rpn_cfg:
                    return rpn_cfg[short]
            if name.startswith("box_"):
                short = name.replace("box_", "")
                if short in roi_cfg:
                    return roi_cfg[short]
            return model_config.get(name, default)

        backbone = MultimodalBackboneWithFPN(
            fpn_out=model_config.get("fpn_out_channels", 256),
            pretrained_backbone=model_config.get("pretrained_backbone", True),
            freeze_backbone=model_config.get("freeze_backbone", False),
        )

        anchor_sizes = model_config.get("anchor_sizes", ((16,), (32,), (64,), (128,), (256,)))
        anchor_ratios = model_config.get("anchor_aspect_ratios", (0.25, 0.5, 1.0, 2.0, 3.0))
        anchor_gen = AnchorGenerator(
            sizes=tuple(tuple(level) for level in anchor_sizes),
            aspect_ratios=(tuple(anchor_ratios),) * len(anchor_sizes),
        )

        roi_pooler = MultiScaleRoIAlign(
            featmap_names=["0", "1", "2", "3"],
            output_size=7,
            sampling_ratio=2,
        )

        self.detector = TorchFasterRCNN(
            backbone=backbone,
            num_classes=num_classes,
            rpn_anchor_generator=anchor_gen,
            box_roi_pool=roi_pooler,
            box_score_thresh=cfg_value("box_score_thresh", 0.05),
            box_nms_thresh=cfg_value("box_nms_thresh", 0.5),
            box_detections_per_img=cfg_value("box_detections_per_img", 100),
            box_fg_iou_thresh=cfg_value("box_fg_iou_thresh", roi_cfg.get("fg_iou_thresh", 0.5)),
            box_bg_iou_thresh=cfg_value("box_bg_iou_thresh", roi_cfg.get("bg_iou_thresh_hi", 0.5)),
            rpn_pre_nms_top_n_train=cfg_value("rpn_pre_nms_top_n_train", 2000),
            rpn_pre_nms_top_n_test=cfg_value("rpn_pre_nms_top_n_test", 1000),
            rpn_post_nms_top_n_train=cfg_value("rpn_post_nms_top_n_train", 2000),
            rpn_post_nms_top_n_test=cfg_value("rpn_post_nms_top_n_test", 1000),
            rpn_nms_thresh=cfg_value("rpn_nms_thresh", rpn_cfg.get("nms_thresh", 0.7)),
            min_size=model_config.get("min_size", 512),
            max_size=model_config.get("max_size", 640),
        )

        self.focal_alpha = float(model_config.get("focal_alpha", 0.25))
        self.focal_gamma = float(model_config.get("focal_gamma", 2.0))
        self.box_loss_type = str(model_config.get("box_loss_type", "giou")).lower()
        self._patch_rpn_loss()

    def _patch_rpn_loss(self) -> None:
        def focal_rpn_loss(rpn_module, objectness, pred_bbox_deltas, labels, regression_targets):
            sampled_pos_inds, sampled_neg_inds = rpn_module.fg_bg_sampler(labels)
            sampled_pos_inds = torch.where(torch.cat(sampled_pos_inds, dim=0))[0]
            sampled_neg_inds = torch.where(torch.cat(sampled_neg_inds, dim=0))[0]
            sampled_inds = torch.cat([sampled_pos_inds, sampled_neg_inds], dim=0)

            objectness = objectness.flatten()
            labels_flat = torch.cat(labels, dim=0)
            regression_targets_flat = torch.cat(regression_targets, dim=0)

            if sampled_pos_inds.numel() > 0:
                box_loss = F.smooth_l1_loss(
                    pred_bbox_deltas[sampled_pos_inds],
                    regression_targets_flat[sampled_pos_inds],
                    beta=1.0 / 9.0,
                    reduction="sum",
                ) / max(sampled_inds.numel(), 1)
            else:
                box_loss = pred_bbox_deltas.sum() * 0.0

            objectness_loss = sigmoid_focal_loss_with_logits(
                objectness[sampled_inds],
                labels_flat[sampled_inds].float(),
                alpha=self.focal_alpha,
                gamma=self.focal_gamma,
                reduction="sum",
            ) / max(sampled_inds.numel(), 1)

            return objectness_loss, box_loss

        self.detector.rpn.compute_loss = focal_rpn_loss.__get__(self.detector.rpn, type(self.detector.rpn))

    def _compute_roi_losses(self, features, proposals, image_sizes, targets):
        roi_heads = self.detector.roi_heads
        proposals, matched_idxs, labels, _ = roi_heads.select_training_samples(proposals, targets)

        box_features = roi_heads.box_roi_pool(features, proposals, image_sizes)
        box_features = roi_heads.box_head(box_features)
        class_logits, box_regression = roi_heads.box_predictor(box_features)

        labels_flat = torch.cat(labels, dim=0)
        classification_loss = F.cross_entropy(class_logits, labels_flat)

        positive_indices = torch.where(labels_flat > 0)[0]
        if positive_indices.numel() == 0:
            box_loss = box_regression.sum() * 0.0
        else:
            num_classes = class_logits.shape[1]
            box_regression = box_regression.reshape(box_regression.shape[0], num_classes, 4)
            labels_pos = labels_flat[positive_indices]
            box_regression_pos = box_regression[positive_indices, labels_pos]

            proposals_flat = torch.cat(proposals, dim=0)
            proposals_pos = proposals_flat[positive_indices]

            matched_gt_boxes = []
            for matched_idxs_per_image, targets_per_image in zip(matched_idxs, targets):
                gt_boxes = targets_per_image["boxes"]
                if gt_boxes.numel() == 0:
                    matched_gt_boxes.append(
                        torch.zeros((matched_idxs_per_image.numel(), 4), device=proposals_pos.device, dtype=proposals_pos.dtype)
                    )
                else:
                    matched_gt_boxes.append(gt_boxes[matched_idxs_per_image.clamp(min=0)])

            matched_gt_boxes = torch.cat(matched_gt_boxes, dim=0)
            target_boxes_pos = matched_gt_boxes[positive_indices]

            weights = getattr(roi_heads.box_coder, "weights", (10.0, 10.0, 5.0, 5.0))
            pred_boxes = sanitize_xyxy_boxes(decode_boxes(box_regression_pos, proposals_pos, weights=weights))
            target_boxes_pos = sanitize_xyxy_boxes(target_boxes_pos)
            if self.box_loss_type in {"diou", "distance_iou", "distance_box_iou"}:
                box_loss = distance_box_iou_loss(pred_boxes, target_boxes_pos, reduction="sum") / max(labels_flat.numel(), 1)
            else:
                box_loss = generalized_box_iou_loss_fn(pred_boxes, target_boxes_pos, reduction="sum") / max(labels_flat.numel(), 1)

        return {"loss_classifier": classification_loss, "loss_box_reg": box_loss}

    def get_param_groups(self, base_lr: float):
        body = self.detector.backbone.body

        def params_of(*modules):
            params: List[nn.Parameter] = []
            for module in modules:
                params.extend(list(module.parameters()))
            return params

        low_backbone = params_of(
            body.rgb_stem,
            body.rgb_layer1,
            body.rgb_layer2,
            body.ir_stem,
            body.ir_layer1,
            body.ir_layer2,
        )
        high_backbone = params_of(
            body.rgb_layer3,
            body.rgb_layer4,
            body.ir_layer3,
            body.ir_layer4,
        )
        fusion_and_heads = params_of(
            self.detector.backbone.fusion_c1,
            self.detector.backbone.fusion_c2,
            self.detector.backbone.fusion_c3,
            self.detector.backbone.fusion_c4,
            self.detector.backbone.fpn,
            self.detector.rpn,
            self.detector.roi_heads,
        )

        return [
            {"params": low_backbone, "lr": base_lr * 0.1},
            {"params": high_backbone, "lr": base_lr * 0.2},
            {"params": fusion_and_heads, "lr": base_lr},
        ]

    def set_backbone_trainable(self, trainable: bool):
        body = self.detector.backbone.body
        if trainable: body._unfreeze_all()
        else: body._freeze_all()

    def set_low_level_backbone_trainable(self, trainable: bool = True) -> None:
        frozen_modules = [
            self.detector.backbone.body.rgb_stem,
            self.detector.backbone.body.rgb_layer1,
            self.detector.backbone.body.ir_stem,
            self.detector.backbone.body.ir_layer1,
        ]

        for module in frozen_modules:
            for param in module.parameters():
                param.requires_grad = trainable

            if trainable:
                module.train()
            else:
                module.eval()
                for submodule in module.modules():
                    if isinstance(submodule, nn.BatchNorm2d):
                        submodule.eval()

    def forward(self, rgb, ir, targets=None):
        if isinstance(rgb, torch.Tensor):
            rgb_list = [rgb[index] for index in range(rgb.shape[0])]
            ir_list = [ir[index] for index in range(ir.shape[0])]
        else:
            rgb_list = list(rgb)
            ir_list = list(ir)

        rgb_batch = torch.stack(rgb_list)
        ir_batch = torch.stack(ir_list)

        images_t, targets = self.detector.transform(rgb_list, targets)
        _, _, height, width = images_t.tensors.shape
        ir_resized = F.interpolate(ir_batch, size=(height, width), mode="bilinear", align_corners=False)

        features = self.detector.backbone(images_t.tensors, ir_resized)
        proposals, proposal_losses = self.detector.rpn(images_t, features, targets)

        if self.training:
            detector_losses = self._compute_roi_losses(features, proposals, images_t.image_sizes, targets)
            losses = {}
            losses.update(proposal_losses)
            losses.update(detector_losses)
            return losses

        detections, _ = self.detector.roi_heads(features, proposals, images_t.image_sizes, targets)
        detections = self.detector.transform.postprocess(
            detections,
            images_t.image_sizes,
            [(img.shape[-2], img.shape[-1]) for img in rgb_list],
        )
        return detections
