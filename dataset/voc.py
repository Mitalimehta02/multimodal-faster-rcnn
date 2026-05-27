import os
import random
import xml.etree.ElementTree as ET
from typing import Dict, List, Tuple

import torch
from PIL import Image, ImageDraw, ImageEnhance
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF
from tqdm import tqdm


def load_images_and_anns(ann_dir: str, rgb_dir: str, ir_dir: str, label2idx: Dict[str, int]) -> List[Dict]:
    infos: List[Dict] = []

    valid_ids = sorted(
        f.replace(".jpg", "")
        for f in os.listdir(rgb_dir)
        if f.endswith(".jpg") and os.path.exists(os.path.join(ir_dir, f))
    )

    print(f"Found {len(valid_ids)} paired RGB/IR images in {rgb_dir}")

    for img_id in tqdm(valid_ids):
        ann_file = os.path.join(ann_dir, img_id + ".xml")
        detections = []

        if os.path.exists(ann_file):
            tree = ET.parse(ann_file)
            root = tree.getroot()

            for obj in root.findall("object"):
                name = obj.find("name").text
                if name != "person":
                    continue

                bbox = obj.find("bndbox")
                x1 = int(float(bbox.find("xmin").text))
                y1 = int(float(bbox.find("ymin").text))
                x2 = int(float(bbox.find("xmax").text))
                y2 = int(float(bbox.find("ymax").text))

                if x2 <= x1 or y2 <= y1:
                    continue

                detections.append(
                    {
                        "bbox": [x1, y1, x2, y2],
                        "label": label2idx["person"],
                    }
                )

        infos.append({"img_id": img_id, "detections": detections})

    print(f"Loaded {len(infos)} samples")
    return infos


class MultimodalVOCDataset(Dataset):
    def __init__(
        self,
        split: str,
        rgb_dir: str,
        ir_dir: str,
        ann_dir: str,
        image_size: Tuple[int, int] = (512, 640),
        scale_range: Tuple[float, float] = (0.7, 1.3),
        brightness: float = 0.30,
        contrast: float = 0.30,
        hflip_prob: float = 0.5,
        cutout_prob: float = 0.5,
        cutout_scale: Tuple[float, float] = (0.10, 0.20),
    ):
        self.split = split
        self.rgb_dir = rgb_dir
        self.ir_dir = ir_dir
        self.ann_dir = ann_dir

        self.classes = ["background", "person"]
        self.label2idx = {c: i for i, c in enumerate(self.classes)}
        self.idx2label = {i: c for i, c in enumerate(self.classes)}

        self.infos = load_images_and_anns(ann_dir, rgb_dir, ir_dir, self.label2idx)

        self.new_h, self.new_w = image_size
        self.scale_range = scale_range
        self.brightness = brightness
        self.contrast = contrast
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

    def _clip_boxes(self, boxes: torch.Tensor) -> torch.Tensor:
        if boxes.numel() == 0:
            return boxes.reshape(0, 4)

        boxes[:, 0::2] = boxes[:, 0::2].clamp(0, self.new_w)
        boxes[:, 1::2] = boxes[:, 1::2].clamp(0, self.new_h)
        keep = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
        return boxes[keep]

    def _random_scale_and_fit(self, rgb: Image.Image, ir: Image.Image, boxes: torch.Tensor):
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

        return rgb, ir, self._clip_boxes(boxes)

    def _random_flip(self, rgb: Image.Image, ir: Image.Image, boxes: torch.Tensor):
        if random.random() >= self.hflip_prob:
            return rgb, ir, boxes

        rgb = rgb.transpose(Image.FLIP_LEFT_RIGHT)
        ir = ir.transpose(Image.FLIP_LEFT_RIGHT)

        if boxes.numel() > 0:
            x1 = self.new_w - boxes[:, 2]
            x2 = self.new_w - boxes[:, 0]
            boxes[:, 0] = x1
            boxes[:, 2] = x2

        return rgb, ir, self._clip_boxes(boxes)

    def _random_photometric(self, rgb: Image.Image, ir: Image.Image):
        brightness = 1.0 + random.uniform(-self.brightness, self.brightness)
        contrast = 1.0 + random.uniform(-self.contrast, self.contrast)

        rgb = ImageEnhance.Brightness(rgb).enhance(brightness)
        ir = ImageEnhance.Brightness(ir).enhance(brightness)

        rgb = ImageEnhance.Contrast(rgb).enhance(contrast)
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

        rgb_draw = ImageDraw.Draw(rgb)
        ir_draw = ImageDraw.Draw(ir)
        rgb_draw.rectangle([left, top, right, bottom], fill=(0, 0, 0))
        ir_draw.rectangle([left, top, right, bottom], fill=0)
        return rgb, ir

    def _apply_train_transforms(self, rgb: Image.Image, ir: Image.Image, boxes: torch.Tensor):
        rgb, ir, boxes = self._random_scale_and_fit(rgb, ir, boxes)
        rgb, ir, boxes = self._random_flip(rgb, ir, boxes)
        rgb, ir = self._random_photometric(rgb, ir)
        rgb, ir = self._random_cutout(rgb, ir)
        return rgb, ir, boxes

    def __getitem__(self, idx: int):
        info = self.infos[idx]
        img_id = info["img_id"]

        rgb_path = os.path.join(self.rgb_dir, img_id + ".jpg")
        ir_path = os.path.join(self.ir_dir, img_id + ".jpg")

        rgb = Image.open(rgb_path).convert("RGB")
        ir = Image.open(ir_path).convert("L")

        boxes = torch.tensor(
            [d["bbox"] for d in info["detections"]],
            dtype=torch.float32,
        ).reshape(-1, 4)

        rgb, ir, boxes = self._resize_pair(rgb, ir, boxes, (self.new_h, self.new_w))

        if self.split == "train":
            rgb, ir, boxes = self._apply_train_transforms(rgb, ir, boxes)

        boxes = self._clip_boxes(boxes)
        labels = torch.ones((boxes.shape[0],), dtype=torch.int64)
        area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) if boxes.numel() > 0 else torch.zeros((0,), dtype=torch.float32)

        rgb_tensor = TF.to_tensor(rgb)
        ir_tensor = TF.to_tensor(ir)

        rgb_tensor = (rgb_tensor - self.rgb_mean) / self.rgb_std
        ir_tensor = (ir_tensor - self.ir_mean) / self.ir_std

        target = {
            "boxes": boxes,
            "labels": labels,
            "area": area,
            "iscrowd": torch.zeros((boxes.shape[0],), dtype=torch.int64),
            "image_id": torch.tensor([idx], dtype=torch.int64),
        }

        return rgb_tensor, ir_tensor, target, img_id
