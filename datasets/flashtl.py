"""FlashTL-1K dataset loader used by the public FlashNet training/evaluation code."""
from __future__ import annotations
import os
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, WeightedRandomSampler
import torchvision.transforms as T

CLASS_NAMES = ["Blinking_Red", "Blinking_Yellow"]
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv"}


def infer_label(path: str) -> Tuple[int, str]:
    p = path.lower().replace("\\", "/")
    if "red" in p:
        return 0, "Blinking_Red"
    if "yellow" in p:
        return 1, "Blinking_Yellow"
    raise ValueError(f"Cannot infer FlashTL-1K label from path: {path}")


def scan_dataset(data_root: str) -> pd.DataFrame:
    rows = []
    for root, _, files in os.walk(data_root):
        for name in sorted(files):
            if Path(name).suffix.lower() not in VIDEO_EXTS:
                continue
            path = str(Path(root) / name)
            try:
                label, label_name = infer_label(path)
            except ValueError:
                continue
            rows.append({
                "path": path,
                "video_name": Path(name).stem,
                "label": label,
                "label_name": label_name,
            })
    if not rows:
        raise RuntimeError(f"No labeled videos found under: {data_root}")
    return pd.DataFrame(rows)


def attach_context(df: pd.DataFrame, context_csv: Optional[str]) -> pd.DataFrame:
    if not context_csv:
        return df
    ctx = pd.read_csv(context_csv)
    if "video_name" not in ctx.columns:
        raise ValueError("context CSV must contain a 'video_name' column")
    keep = ["video_name"] + [c for c in [
        "gdino_bbox_x1", "gdino_bbox_y1", "gdino_bbox_x2", "gdino_bbox_y2",
        "fft_flicker_ratio", "fft_dominant_freq_hz",
        "clip_is_day", "clip_is_night", "clip_is_rain", "clip_is_inside",
    ] if c in ctx.columns]
    return df.merge(ctx[keep].drop_duplicates("video_name"), on="video_name", how="left")


def build_transform(img_size: int, train: bool):
    if train:
        return T.Compose([
            T.ToPILImage(),
            T.Resize((img_size + 32, img_size + 32)),
            T.RandomCrop((img_size, img_size)),
            T.RandomHorizontalFlip(0.5),
            T.ColorJitter(0.4, 0.4, 0.3, 0.1),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    return T.Compose([
        T.ToPILImage(),
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


class FlashTLDataset(Dataset):
    """Returns (video, label, raw_brightness_trace, flicker_ratio)."""

    def __init__(self, df: pd.DataFrame, num_frames: int = 16,
                 img_size: int = 224, train: bool = False):
        self.df = df.reset_index(drop=True)
        self.num_frames = num_frames
        self.transform = build_transform(img_size, train)

    def __len__(self):
        return len(self.df)

    @staticmethod
    def _bbox_from_row(row, width: int, height: int):
        cols = ["gdino_bbox_x1", "gdino_bbox_y1", "gdino_bbox_x2", "gdino_bbox_y2"]
        if not all(c in row.index and pd.notna(row[c]) for c in cols):
            return None
        x1, y1, x2, y2 = [int(float(row[c])) for c in cols]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(width, x2), min(height, y2)
        return (x1, y1, x2, y2) if x2 > x1 and y2 > y1 else None

    def _read_video(self, row):
        cap = cv2.VideoCapture(row["path"])
        total = max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 1)
        indices = np.linspace(0, total - 1, self.num_frames, dtype=int)

        tensors, intensities, last = [], [], None
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, bgr = cap.read()
            if ok and bgr is not None:
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                last = rgb
            else:
                rgb = last if last is not None else np.zeros((360, 640, 3), np.uint8)

            h, w = rgb.shape[:2]
            bbox = self._bbox_from_row(row, w, h)
            roi = rgb
            if bbox is not None:
                x1, y1, x2, y2 = bbox
                roi = rgb[y1:y2, x1:x2]

            gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
            intensities.append(float(gray.mean()))
            tensors.append(self.transform(rgb))

        cap.release()
        raw_trace = np.asarray(intensities, dtype=np.float32)
        mean = float(raw_trace.mean())
        rho = float(raw_trace.std() / mean) if mean > 0 else 0.0

        return (
            torch.stack(tensors),
            torch.from_numpy(raw_trace),
            torch.tensor(rho, dtype=torch.float32),
        )

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        video, trace, rho = self._read_video(row)
        return video, torch.tensor(int(row["label"]), dtype=torch.long), trace, rho


def make_weighted_sampler(labels):
    labels = np.asarray(labels, dtype=int)
    counts = np.bincount(labels)
    sample_weights = 1.0 / counts[labels]
    return WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
