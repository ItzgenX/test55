"""
src/data/local_depth.py
------------------------
Dataset/DataModule for the depth branch's precomputed-map workflow. Shape
follows the original repo's own local.py (ImageFolderDataset / ImageDataModule,
see RULE 0 in DEPTH-BRANCH-architecture.md) plus reading the jsonl manifest
that compute_depth.py writes -- nothing here is copied from any segmentation
branch's dataset class.

Depth stays float in [0,1], 3-channel after expansion (expansion happens
here at load time, never on disk -- compute_depth.py stores 1-channel).
The RGB image was never re-saved by compute_depth.py, so this loads the
original source image and re-applies the exact same crop (recorded per-entry
in the manifest) to keep image and depth pixel-aligned.
"""

import json
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

PROJECT_ROOT = Path(os.path.abspath(__file__)).parent.parent.parent


class DepthJsonDataset(Dataset):
    def __init__(self, jsonl_path: str, caption_prefix: str = ""):
        self.jsonl_path = Path(jsonl_path)
        if not self.jsonl_path.exists():
            raise FileNotFoundError(
                f"{self.jsonl_path} does not exist -- run compute_depth.py first "
                f"to produce it (it writes <output_dir>/train.jsonl and val.jsonl)."
            )
        with open(self.jsonl_path, "r", encoding="utf-8") as f:
            self.entries = [json.loads(l) for l in f if l.strip()]
        if len(self.entries) == 0:
            raise ValueError(f"{self.jsonl_path} is empty")

        self.caption_prefix = caption_prefix
        self.rgb_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx: int):
        e = self.entries[idx]

        img = Image.open(PROJECT_ROOT / e["source"]).convert("RGB")
        w, h = img.size
        orig_w, orig_h = e["orig_size"]
        if (w, h) != (orig_w, orig_h):
            raise ValueError(
                f"{e['source']} is {w}x{h} but the manifest recorded {orig_w}x{orig_h} "
                f"at precompute time -- the source dataset changed since compute_depth.py ran."
            )
        crop_top = e["crop_top"]
        crop_bottom = e["crop_bottom"]
        img = img.crop((0, crop_top, w, h - crop_bottom))
        cropped_w, cropped_h = e["cropped_size"]
        assert img.size == (cropped_w, cropped_h), (img.size, e["cropped_size"])

        img_t = self.rgb_transform(img)  # [3,H,W] in [-1,1]

        depth_path = PROJECT_ROOT / e["depth_path"]
        if depth_path.suffix == ".npy":
            depth_arr = np.load(depth_path).astype(np.float32)
        elif depth_path.suffix == ".png":
            depth_arr = np.array(Image.open(depth_path)).astype(np.float32) / 65535.0
        else:
            raise ValueError(f"unknown depth file extension: {depth_path.suffix}")

        depth_t = torch.from_numpy(depth_arr).unsqueeze(0)  # [1,H,W] in [0,1]
        depth_t = depth_t.repeat(3, 1, 1)  # expand to 3 channels at load time, not on disk

        assert img_t.shape[-2:] == depth_t.shape[-2:] == (cropped_h, cropped_w), (
            img_t.shape,
            depth_t.shape,
        )

        caption = self.caption_prefix + e.get("prompt", "")

        return {"jpg": img_t, "caption": caption, "depth": depth_t}


class DepthJsonDataModule:
    def __init__(
        self,
        train_jsonl: str,
        val_jsonl: str,
        batch_size: int = 4,
        val_batch_size: int = 1,
        workers: int = 4,
        val_workers: int = 1,
        caption_prefix: str = "",
    ):
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size
        self.workers = workers
        self.val_workers = val_workers

        self.train_dataset = DepthJsonDataset(train_jsonl, caption_prefix)
        self.val_dataset = DepthJsonDataset(val_jsonl, caption_prefix)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.val_batch_size, shuffle=False, num_workers=self.val_workers)
