import os
import random
import re
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image, ImageDraw, ImageEnhance, ImageOps
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF
from tqdm import tqdm


FLIR_CLASSES = ["background", "person", "car", "bicycle"]

LABEL_ALIASES = {
    "people": "person",
    "person": "person",
    "pedestrian": "person",
    "car": "car",
    "cars": "car",
    "vehicle": "car",
    "bicycle": "bicycle",
    "bike": "bicycle",
    "cyclist": "bicycle",
}


def _read_split_ids(split_file: str) -> List[str]:
    ids: List[str] = []
    with open(split_file, "r", encoding="utf-8") as handle:
        for line in handle:
            token = line.strip()
            if not token:
                continue

            name = os.path.basename(token)
            match = re.search(r"FLIR_(\d+)", name)
            if match:
                ids.append(match.group(1))
                continue

            parts = name.split("_")
            if len(parts) >= 2 and parts[1].isdigit():
                ids.append(parts[1])
            else:
                ids.append(os.path.splitext(name)[0])
    return ids


def _resolve_template(root: str, template: str, image_id: str) -> str:
    path = template.format(id=image_id)
    if not os.path.isabs(path):
        path = os.path.join(root, path)
    return path


def _parse_annotation(xml_path: str, label2idx: Dict[str, int]) -> List[Dict]:
    detections: List[Dict] = []
    if not os.path.exists(xml_path):
        return detections

    tree = ET.parse(xml_path)
    root = tree.getroot()

    for obj in root.findall("object"):
        raw_name = (obj.findtext("name") or "").strip().lower()
        name = LABEL_ALIASES.get(raw_name)
        if name not in label2idx:
            continue

        bbox = obj.find("bndbox")
        if bbox is None:
            continue

        x1 = int(float(bbox.findtext("xmin", "0")))
        y1 = int(float(bbox.findtext("ymin", "0")))
        x2 = int(float(bbox.findtext("xmax", "0")))
        y2 = int(float(bbox.findtext("ymax", "0")))

        if x2 <= x1 or y2 <= y1:
            continue

        detections.append({"bbox": [x1, y1, x2, y2], "label": label2idx[name]})

    return detections


def load_flir_infos(
    root: str,
    split_file: str,
    rgb_template: str,
    ir_template: str,
    ann_template: str,
    label2idx: Dict[str, int],
    keep_empty: bool = True,
) -> List[Dict]:
    image_ids = _read_split_ids(split_file)
    infos: List[Dict] = []
    missing = 0
    dropped_empty = 0

    for image_id in tqdm(image_ids, desc=f"Loading {os.path.basename(split_file)}"):
        rgb_path = _resolve_template(root, rgb_template, image_id)
        ir_path = _resolve_template(root, ir_template, image_id)
        ann_path = _resolve_template(root, ann_template, image_id)

        if not os.path.exists(rgb_path) or not os.path.exists(ir_path):
            missing += 1
            continue

        detections = _parse_annotation(ann_path, label2idx)
        if not detections and not keep_empty:
            dropped_empty += 1
            continue

        infos.append(
            {
                "img_id": f"FLIR_{image_id}",
                "rgb_path": rgb_path,
                "ir_path": ir_path,
                "ann_path": ann_path,
                "detections": detections,
            }
        )

    print(
        f"Loaded {len(infos)} FLIR samples from {split_file} "
        f"(missing_pairs={missing}, empty_removed={dropped_empty})"
    )
    return infos


class FlirAlignedDataset(Dataset):
    def __init__(
        self,
        root: str,
        split_file: str,
        rgb_template: str = "JPEGImages/FLIR_{id}_RGB.jpg",
        ir_template: str = "JPEGImages/FLIR_{id}_PreviewData.jpeg",
        ann_template: str = "Annotations/FLIR_{id}_PreviewData.xml",
        image_size: Tuple[int, int] = (512, 640),
        training: bool = False,
        scale_range: Tuple[float, float] = (0.85, 1.15),
        rgb_brightness: float = 0.20,
        rgb_contrast: float = 0.20,
        rgb_saturation: float = 0.0,
        rgb_hue: float = 0.0,
        rgb_random_grayscale_prob: float = 0.0,
        thermal_brightness: float = 0.0,
        thermal_contrast: float = 0.0,
        hflip_prob: float = 0.5,
        cutout_prob: float = 0.15,
        cutout_scale: Tuple[float, float] = (0.05, 0.12),
        keep_empty: bool = True,
    ):
        self.root = root
        self.split_file = split_file
        self.training = training

        self.classes = FLIR_CLASSES
        self.label2idx = {name: idx for idx, name in enumerate(self.classes)}
        self.idx2label = {idx: name for name, idx in self.label2idx.items()}

        self.infos = load_flir_infos(
            root=root,
            split_file=split_file,
            rgb_template=rgb_template,
            ir_template=ir_template,
            ann_template=ann_template,
            label2idx=self.label2idx,
            keep_empty=keep_empty,
        )

        self.new_h, self.new_w = image_size
        self.scale_range = scale_range
        self.rgb_brightness = rgb_brightness
        self.rgb_contrast = rgb_contrast
        self.rgb_saturation = rgb_saturation
        self.rgb_hue = rgb_hue
        self.rgb_random_grayscale_prob = rgb_random_grayscale_prob
        self.thermal_brightness = thermal_brightness
        self.thermal_contrast = thermal_contrast
        self.hflip_prob = hflip_prob
        self.cutout_prob = cutout_prob
        self.cutout_scale = cutout_scale

        self.rgb_mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
        self.rgb_std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)
        self.ir_mean = torch.tensor([0.5], dtype=torch.float32).view(1, 1, 1)
        self.ir_std = torch.tensor([0.25], dtype=torch.float32).view(1, 1, 1)

    def __len__(self) -> int:
        return len(self.infos)

    def _resize_pair(self, rgb: Image.Image, ir: Image.Image, boxes: torch.Tensor, size: Tuple[int, int]):
        target_h, target_w = size
        src_w, src_h = rgb.size

        rgb = rgb.resize((target_w, target_h), Image.BILINEAR)
        ir = ir.resize((target_w, target_h), Image.BILINEAR)

        if boxes.numel() > 0:
            scale = boxes.new_tensor(
                [
                    target_w / max(src_w, 1),
                    target_h / max(src_h, 1),
                    target_w / max(src_w, 1),
                    target_h / max(src_h, 1),
                ]
            )
            boxes = boxes * scale

        return rgb, ir, boxes

    def _clip_boxes_and_labels(self, boxes: torch.Tensor, labels: torch.Tensor):
        if boxes.numel() == 0:
            return boxes.reshape(0, 4), labels.reshape(0)

        boxes[:, 0::2] = boxes[:, 0::2].clamp(0, self.new_w)
        boxes[:, 1::2] = boxes[:, 1::2].clamp(0, self.new_h)
        keep = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
        return boxes[keep], labels[keep]

    def _random_scale_and_fit(self, rgb: Image.Image, ir: Image.Image, boxes: torch.Tensor, labels: torch.Tensor):
        scale = random.uniform(*self.scale_range)
        scaled_h = max(1, int(round(self.new_h * scale)))
        scaled_w = max(1, int(round(self.new_w * scale)))

        rgb, ir, boxes = self._resize_pair(rgb, ir, boxes, (scaled_h, scaled_w))

        if scaled_w >= self.new_w and scaled_h >= self.new_h:
            left = random.randint(0, scaled_w - self.new_w)
            top = random.randint(0, scaled_h - self.new_h)
            rgb = rgb.crop((left, top, left + self.new_w, top + self.new_h))
            ir = ir.crop((left, top, left + self.new_w, top + self.new_h))
            if boxes.numel() > 0:
                boxes[:, 0::2] -= left
                boxes[:, 1::2] -= top
        else:
            left = random.randint(0, self.new_w - scaled_w)
            top = random.randint(0, self.new_h - scaled_h)
            rgb_canvas = Image.new("RGB", (self.new_w, self.new_h), (0, 0, 0))
            ir_canvas = Image.new("L", (self.new_w, self.new_h), 0)
            rgb_canvas.paste(rgb, (left, top))
            ir_canvas.paste(ir, (left, top))
            rgb, ir = rgb_canvas, ir_canvas
            if boxes.numel() > 0:
                boxes[:, 0::2] += left
                boxes[:, 1::2] += top

        return (*self._clip_boxes_and_labels(boxes, labels), rgb, ir)

    def _random_flip(self, rgb: Image.Image, ir: Image.Image, boxes: torch.Tensor, labels: torch.Tensor):
        if random.random() >= self.hflip_prob:
            return rgb, ir, boxes, labels

        rgb = rgb.transpose(Image.FLIP_LEFT_RIGHT)
        ir = ir.transpose(Image.FLIP_LEFT_RIGHT)

        if boxes.numel() > 0:
            x1 = self.new_w - boxes[:, 2]
            x2 = self.new_w - boxes[:, 0]
            boxes[:, 0] = x1
            boxes[:, 2] = x2

        boxes, labels = self._clip_boxes_and_labels(boxes, labels)
        return rgb, ir, boxes, labels

    def _random_photometric(self, rgb: Image.Image, ir: Image.Image):
        if self.rgb_brightness > 0:
            brightness = 1.0 + random.uniform(-self.rgb_brightness, self.rgb_brightness)
            rgb = ImageEnhance.Brightness(rgb).enhance(brightness)
        if self.rgb_contrast > 0:
            contrast = 1.0 + random.uniform(-self.rgb_contrast, self.rgb_contrast)
            rgb = ImageEnhance.Contrast(rgb).enhance(contrast)
        if self.rgb_saturation > 0:
            saturation = 1.0 + random.uniform(-self.rgb_saturation, self.rgb_saturation)
            rgb = ImageEnhance.Color(rgb).enhance(saturation)
        if self.rgb_hue > 0:
            hue_shift = int(round(random.uniform(-self.rgb_hue, self.rgb_hue) * 255))
            h, s, v = rgb.convert("HSV").split()
            h = h.point(lambda px: (px + hue_shift) % 256)
            rgb = Image.merge("HSV", (h, s, v)).convert("RGB")
        if self.rgb_random_grayscale_prob > 0 and random.random() < self.rgb_random_grayscale_prob:
            rgb = ImageOps.grayscale(rgb).convert("RGB")
        if self.thermal_brightness > 0:
            brightness = 1.0 + random.uniform(-self.thermal_brightness, self.thermal_brightness)
            ir = ImageEnhance.Brightness(ir).enhance(brightness)
        if self.thermal_contrast > 0:
            contrast = 1.0 + random.uniform(-self.thermal_contrast, self.thermal_contrast)
            ir = ImageEnhance.Contrast(ir).enhance(contrast)
        return rgb, ir

    def _random_cutout(self, rgb: Image.Image, ir: Image.Image):
        if random.random() >= self.cutout_prob:
            return rgb, ir

        area_frac = random.uniform(*self.cutout_scale)
        cutout_h = max(1, int(self.new_h * area_frac))
        cutout_w = max(1, int(self.new_w * area_frac))
        left = random.randint(0, max(0, self.new_w - cutout_w))
        top = random.randint(0, max(0, self.new_h - cutout_h))
        right = left + cutout_w
        bottom = top + cutout_h

        ImageDraw.Draw(rgb).rectangle([left, top, right, bottom], fill=(0, 0, 0))
        ImageDraw.Draw(ir).rectangle([left, top, right, bottom], fill=0)
        return rgb, ir

    def _apply_train_transforms(self, rgb: Image.Image, ir: Image.Image, boxes: torch.Tensor, labels: torch.Tensor):
        boxes, labels, rgb, ir = self._random_scale_and_fit(rgb, ir, boxes, labels)
        rgb, ir, boxes, labels = self._random_flip(rgb, ir, boxes, labels)
        rgb, ir = self._random_photometric(rgb, ir)
        rgb, ir = self._random_cutout(rgb, ir)
        return rgb, ir, boxes, labels

    def __getitem__(self, idx: int):
        info = self.infos[idx]

        rgb = Image.open(info["rgb_path"]).convert("RGB")
        ir = Image.open(info["ir_path"]).convert("L")

        boxes = torch.tensor([d["bbox"] for d in info["detections"]], dtype=torch.float32).reshape(-1, 4)
        labels = torch.tensor([d["label"] for d in info["detections"]], dtype=torch.int64)

        rgb, ir, boxes = self._resize_pair(rgb, ir, boxes, (self.new_h, self.new_w))
        boxes, labels = self._clip_boxes_and_labels(boxes, labels)

        if self.training:
            rgb, ir, boxes, labels = self._apply_train_transforms(rgb, ir, boxes, labels)

        area = (
            (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            if boxes.numel() > 0
            else torch.zeros((0,), dtype=torch.float32)
        )

        rgb_tensor = (TF.to_tensor(rgb) - self.rgb_mean) / self.rgb_std
        ir_tensor = (TF.to_tensor(ir) - self.ir_mean) / self.ir_std

        target = {
            "boxes": boxes,
            "labels": labels,
            "area": area,
            "iscrowd": torch.zeros((boxes.shape[0],), dtype=torch.int64),
            "image_id": torch.tensor([idx], dtype=torch.int64),
        }

        return rgb_tensor, ir_tensor, target, info["img_id"]
