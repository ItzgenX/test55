"""One-off sanity check: how accurate are seg_map_calculations.py's predicted
class-ID maps against Cityscapes' own human-annotated ground truth
(data/dataset/extracted/gtFine)?

Not part of the training/inference pipeline -- this is a diagnostic to
answer "is SegFormer-b5 actually getting these right" before trusting it
as a conditioning signal.

Important: gtFine's *_gtFine_labelIds.png uses Cityscapes' full 34-class
raw "id" scheme (road=7, sidewalk=8, ...), NOT the 19-class "trainId"
scheme (0-18) SegformerEncoder predicts in. This script converts id ->
trainId using the standard public Cityscapes mapping before comparing;
without that conversion the ids wouldn't even be on the same scale.

Usage:
    python check_seg_accuracy.py --pred_jsonl data/seg_training_aspect/train.jsonl --gtfine_root data/dataset/extracted/gtFine --split train
"""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from src.utils import compute_miou

# Standard Cityscapes id -> trainId mapping (cityscapesscripts labels.py).
# 255 = ignore (not one of the 19 evaluated classes).
CITYSCAPES_ID_TO_TRAINID = {
    0: 255, 1: 255, 2: 255, 3: 255, 4: 255, 5: 255, 6: 255,
    7: 0, 8: 1, 9: 255, 10: 255,
    11: 2, 12: 3, 13: 4, 14: 255, 15: 255, 16: 255,
    17: 5, 18: 255, 19: 6, 20: 7, 21: 8, 22: 9, 23: 10,
    24: 11, 25: 12, 26: 13, 27: 14, 28: 15, 29: 255, 30: 255,
    31: 16, 32: 17, 33: 18, -1: 255,
}
_LUT = np.full(256, 255, dtype=np.uint8)
for _id, _train_id in CITYSCAPES_ID_TO_TRAINID.items():
    if 0 <= _id < 256:
        _LUT[_id] = _train_id


def id_to_trainid(label_ids: np.ndarray) -> np.ndarray:
    return _LUT[label_ids]


def find_gt_path(raw_image_path: str, gtfine_root: Path, split: str) -> Path | None:
    # our adapted sample folders are named after the original Cityscapes stem
    # (city_seq_frame), which is also the parent folder name here.
    stem = Path(raw_image_path).parent.name  # e.g. aachen_000000_000019
    city = stem.split("_")[0]
    return gtfine_root / split / city / f"{stem}_gtFine_labelIds.png"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pred_jsonl", required=True, help="a train/val.jsonl written by seg_map_calculations.py")
    p.add_argument("--gtfine_root", default="data/dataset/extracted/gtFine")
    p.add_argument("--split", required=True, choices=["train", "val"])
    args = p.parse_args()

    gtfine_root = Path(args.gtfine_root)
    with open(args.pred_jsonl, "r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]

    mious = []
    per_image = []
    for row in rows:
        gt_path = find_gt_path(row["raw_image_path"], gtfine_root, args.split)
        if not gt_path.exists():
            print(f"[WARN] no ground truth for {row['raw_image_path']} (looked for {gt_path}) -- skipping")
            continue

        pred = np.asarray(Image.open(row["seg_path"]).convert("L"))  # already at target size, 0-18

        gt_raw = Image.open(gt_path).convert("L")
        # NEAREST + same (width, height) target as the prediction: a fair,
        # like-for-like comparison against what the actual training pipeline
        # would see, not an idealized full-resolution comparison.
        gt_raw = gt_raw.resize(pred.shape[::-1], Image.NEAREST)  # PIL wants (W, H)
        gt_ids = id_to_trainid(np.asarray(gt_raw))

        valid = gt_ids != 255
        if not valid.any():
            continue

        miou = compute_miou(
            __import__("torch").from_numpy(pred[valid].astype(np.int64)),
            __import__("torch").from_numpy(gt_ids[valid].astype(np.int64)),
            num_classes=19,
        )
        mious.append(miou)
        per_image.append((Path(row["raw_image_path"]).parent.name, miou))
        print(f"  {Path(row['raw_image_path']).parent.name}: mIoU = {miou:.4f}")

    if mious:
        print(f"\nmean mIoU over {len(mious)} images ({args.split}): {sum(mious) / len(mious):.4f}")
        print("(SegFormer-b5's published Cityscapes val mIoU is ~0.84; this sample should land in that neighborhood)")
    else:
        print("\nNo scoreable images found.")


if __name__ == "__main__":
    main()
