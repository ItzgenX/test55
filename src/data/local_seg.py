import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from src.data.transforms import build_seg_preprocess, normalize_size
from src.encoders.seg_encoder import seg_colorize_ids
from src.encoders.grounded_sam_encoder import carla_palette_tensor


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

    conditioning: "classid" (default) -- raw class-id Long map, shifted 1..28
      (0 reserved NULL/padding, 29 reserved IGNORE/void), for
      SegIDStructureMapperXL's embedding stem. "rgb" -- legacy palette-
      colourised [0,1] float map, for FixedStructureMapperXL's conv stem.
      Must match whichever mapper_network the experiment config selects
      (segidXL vs fsmXL).
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
        # grounded_sam's own 28-class CARLA-vocabulary palette (see
        # src/encoders/grounded_sam_encoder.py) -- NOT segformer's 19-class
        # SEG_CITYSCAPES_PALETTE. This branch's copy of this file is
        # intentionally specialized to its own encoder's class scheme, same
        # as the rest of the branch (segformer's copy of this file keeps its
        # own palette import unchanged).
        self.palette = carla_palette_tensor()

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
        # never predicts it, but GroundedSamEncoder legitimately does for
        # pixels no detection touched (src/encoders/grounded_sam_encoder.py),
        # and seg_colorize_ids does a raw palette[ids] lookup with no bounds
        # check -- palette[255] on a 19-entry palette is a real IndexError,
        # not hypothetical (caught by actually running grounded_sam training
        # for the first time). Same fix as GroundedSamEncoder.forward(): mask
        # out-of-range ids before the lookup, paint those pixels black after.
        ignore_mask = ids >= self.palette.shape[0]
        ids_safe = torch.where(ignore_mask, torch.zeros_like(ids), ids)
        colour = seg_colorize_ids(ids_safe, self.palette)[0]  # [3, H, W] in [0, 1]
        colour = torch.where(ignore_mask.unsqueeze(0), torch.zeros_like(colour), colour)
        return colour

    def _load_seg_ids(self, seg_path: str) -> torch.Tensor:
        """Raw class-id PNG (values 0..27, or 255=IGNORE_ID) -> Long [H, W]
        remapped for SegIDStructureMapperXL's embedding table: 0..27 -> 1..28
        (0 reserved NULL/padding_idx, used by model.py's CFG dropout), 255 ->
        29 (a DEDICATED ignore/void row, distinct from NULL -- "no detection
        touched this pixel" is real per-pixel information, not the same
        thing as "the whole map was CFG-dropped"). See that mapper class's
        docstring for the full index scheme."""
        w, h = self.size
        ids_pil = Image.open(seg_path).convert("L")
        if ids_pil.size != (w, h):
            # NEAREST only: interpolating class ids would fabricate classes
            # that were never in the source map.
            ids_pil = ids_pil.resize((w, h), Image.NEAREST)
        ids = torch.from_numpy(np.asarray(ids_pil, dtype=np.int64))  # [H, W], 0..27 or 255
        ignore_mask = ids >= self.palette.shape[0]  # catches 255 (and any other out-of-range id)
        return torch.where(ignore_mask, torch.full_like(ids, 29), ids + 1)

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
