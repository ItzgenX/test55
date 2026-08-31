import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from src.data.transforms import build_seg_preprocess, normalize_size
from src.encoders.seg_encoder import seg_colorize_ids, seg_palette_tensor


class SegJsonDataset(Dataset):
    """Reads a {"raw_image_path", "seg_path", "prompt"} jsonl manifest (see
    seg_map_calculations.py, which builds these). "seg_path" points at a raw
    class-ID PNG (mode "L", values 0..num_classes-1, saved by
    precompute_segmentation_maps) -- this class colourises it at LOAD time
    via seg_colorize_ids, using the same palette as everywhere else in the
    pipeline (calc time / here / segformer_inference.py's display panel).

    Returned batch: "jpg" (RGB, [-1,1], the VAE's target), "seg" (conditioning
    map -- meant for model.forward(..., skip_encode=True) so it goes straight
    to the mapper instead of through a live encoder), "caption" (the prompt).

    conditioning: "classid" (default) -- raw class-id Long map, shifted 1..19
      (0 reserved NULL/padding), for SegIDStructureMapperXL's embedding stem.
      "rgb" -- legacy palette-colourised [0,1] float map, for
      FixedStructureMapperXL's conv stem. Must match whichever mapper_network
      the experiment config selects (segidXL vs fsmXL) -- picking the wrong
      one doesn't crash, it silently trains a mismatched conditioning shape/
      dtype (Long ids fed to a 3-channel RGB conv, or vice versa).
    """

    def __init__(self, jsonl_path: str, size, resize_mode: str = "aspect", conditioning: str = "classid"):
        self.jsonl_path = Path(jsonl_path)
        self.size = normalize_size(size)  # (width, height)
        self.resize_mode = resize_mode
        assert conditioning in ("classid", "rgb"), f"conditioning must be 'classid' or 'rgb', got {conditioning!r}"
        self.conditioning = conditioning

        with open(self.jsonl_path, "r", encoding="utf-8") as f:
            self.rows = [json.loads(line) for line in f if line.strip()]

        # Same function seg_map_calculations.py uses for the RGB it feeds
        # SegFormer -- the single source of truth that keeps a seg map's
        # geometry matched to the RGB image it's paired with.
        self.rgb_transform = build_seg_preprocess(size=self.size, resize_mode=resize_mode)
        self.palette = seg_palette_tensor()

    def __len__(self) -> int:
        return len(self.rows)

    def _load_seg_colormap(self, seg_path: str) -> torch.Tensor:
        w, h = self.size
        ids_pil = Image.open(seg_path).convert("L")
        if ids_pil.size != (w, h):
            # NEAREST only: averaging class ids during resize would fabricate
            # classes that were never in the source map.
            ids_pil = ids_pil.resize((w, h), Image.NEAREST)
        ids = torch.from_numpy(np.asarray(ids_pil, dtype=np.int64))  # [H, W]
        # IGNORE_ID (255) has no palette entry -- SegFormer's dense classifier
        # never predicts it, but the grounded_sam branch's encoder legitimately
        # does for pixels no detection touched, and seg_colorize_ids does a raw
        # palette[ids] lookup with no bounds check -- palette[255] on a 19-entry
        # palette is a real IndexError there (caught by actually running
        # grounded_sam training). Dormant here since this branch's ids never hit
        # it, but fixed for consistency -- same code, same latent bug.
        ignore_mask = ids >= self.palette.shape[0]
        ids_safe = torch.where(ignore_mask, torch.zeros_like(ids), ids)
        colour = seg_colorize_ids(ids_safe, self.palette)[0]  # [3, H, W] in [0, 1]
        colour = torch.where(ignore_mask.unsqueeze(0), torch.zeros_like(colour), colour)
        return colour

    def _load_seg_ids(self, seg_path: str) -> torch.Tensor:
        """Raw class-id PNG (values 0..18) -> Long [H, W] shifted to 1..19.
        0 is reserved as the NULL/padding_idx row in SegIDStructureMapperXL's
        embedding table -- see that class's docstring for why the shift
        matters beyond labeling: model.py's CFG dropout zeroes `c` directly,
        so id 0 must decode to "no conditioning", not "class 0" (road)."""
        w, h = self.size
        ids_pil = Image.open(seg_path).convert("L")
        if ids_pil.size != (w, h):
            # NEAREST only: interpolating class ids would fabricate classes
            # that were never in the source map.
            ids_pil = ids_pil.resize((w, h), Image.NEAREST)
        ids = torch.from_numpy(np.asarray(ids_pil, dtype=np.int64))  # [H, W], 0..18
        return ids + 1  # -> 1..19, 0 reserved NULL

    def __getitem__(self, idx: int) -> dict:
        row = self.rows[idx]

        rgb = Image.open(row["raw_image_path"]).convert("RGB")

        seg = self._load_seg_ids(row["seg_path"]) if self.conditioning == "classid" \
            else self._load_seg_colormap(row["seg_path"])

        return {
            "jpg": self.rgb_transform(rgb),
            "seg": seg,
            "caption": row.get("prompt") or "",
        }


class SegJsonDataModule:
    def __init__(
        self,
        train_jsonl: str,
        val_jsonl: str,
        size,
        resize_mode: str = "aspect",
        batch_size: int = 4,
        val_batch_size: int = 1,
        workers: int = 4,
        val_workers: int = 1,
        conditioning: str = "classid",
    ):
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size
        self.workers = workers
        self.val_workers = val_workers

        self.train_dataset = SegJsonDataset(train_jsonl, size, resize_mode, conditioning)
        self.val_dataset = SegJsonDataset(val_jsonl, size, resize_mode, conditioning)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.workers)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset, batch_size=self.val_batch_size, shuffle=False, num_workers=self.val_workers)
