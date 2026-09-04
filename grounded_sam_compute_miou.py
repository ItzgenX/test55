"""
grounded_sam_compute_miou.py
-----------------------------
Compute mean IoU between two sets of precomputed segmentation maps -- e.g.
ground-truth segmentation (from real/original images) vs. segmentation of
this model's own generated images -- to score how well generation follows
the segmentation conditioning it was given.

QUICK COMMANDS (run from repo root with conda loradapter env active):
  # --- Compare two segmentation-map PNGs per row ---
  python grounded_sam_compute_miou.py --jsonl my_pairs.jsonl

  # --- Custom key names / class count / per-image CSV output ---
  python grounded_sam_compute_miou.py --jsonl my_pairs.jsonl \\
      --gt_key gt_seg_path --gen_key gen_seg_path --num_classes 28 \\
      --output results.csv

  # --- Also save side-by-side comparison images (GT | GENERATED | ERROR) ---
  python grounded_sam_compute_miou.py --jsonl my_pairs.jsonl \\
      --viz_dir viz_out --viz_limit 20

VISUALIZATION (--viz_dir): for the first --viz_limit scored pairs (default
20 -- this can run on a full dataset, so it doesn't render every single one
unless you raise the limit), writes a 3-panel comparison image per pair
into --viz_dir: GT (colorized) | GENERATED (colorized) | ERROR MAP (white
where they agree, red where they disagree, black for pixels excluded by
IGNORE on either side). Uses the SAME carla_palette_tensor()/
seg_colorize_ids this whole project's training and inference scripts use
for coloring -- not a separate palette, so these visualizations look
exactly like every other seg-map panel in this repo. Each filename embeds
its own mIoU score, so sorting the folder by name also sorts by score.

INPUT: a jsonl with one object per line, each holding two keys pointing at
precomputed class-id segmentation PNGs (mode "L", 0..num_classes-1 real
classes, 255=IGNORE for unmatched pixels -- exactly what
grounded_sam_map_calculations.py writes):
  {"gt_seg_path": "...", "gen_seg_path": "..."}
Default key names are gt_seg_path/gen_seg_path; override with --gt_key/
--gen_key if your manifest uses different names.

BOTH SIDES MUST BE THE SAME CLASS VOCABULARY, produced by the SAME
detection pipeline -- if your ground-truth maps came from a different
source (a different encoder, a different class scheme), the class ids
won't line up and mIoU is meaningless. This repo's own convention:
grounded_sam_map_calculations.py writes 0-indexed CARLA class ids
(road=0..guard_rail=27) with 255=IGNORE for unmatched pixels -- reuse
that same script for BOTH the ground-truth side and the generated-image
side, so both maps come from the identical process. --num_classes
defaults to 28 (this branch's CARLA_CLASSES count) to match.

IGNORE HANDLING: 255 is excluded automatically -- it never equals any
class id in range(num_classes), so unmatched pixels on either side simply
never contribute to any class's intersection/union (compute_miou's own
behavior, reused unchanged from src/utils.py -- not reimplemented here).

Reuses src.utils.compute_miou (already used for live mIoU scoring during
training) -- this script is just a jsonl-driven batch wrapper around it,
not a new metric implementation.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from src.utils import compute_miou

PROJECT_ROOT = Path(__file__).resolve().parent


def load_ids(path: str) -> torch.Tensor:
    p = Path(path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    if not p.is_file():
        raise FileNotFoundError(f"segmentation map not found: {p}")
    arr = np.asarray(Image.open(p).convert("L"), dtype=np.int64)
    return torch.from_numpy(arr)


def _label_bar(width: int, text: str, bar_h: int = 28) -> np.ndarray:
    """Small self-contained version of grounded_sam_inference.py's own
    _seg_label_bar (dark banner, centered text) -- NOT imported from there
    on purpose: that file pulls in diffusers/hydra/torch's full pipeline
    stack, which this lightweight analysis script has no other reason to
    depend on. Same visual style, zero extra dependencies."""
    bar = Image.new("RGB", (width, bar_h), color=(25, 25, 25))
    draw = ImageDraw.Draw(bar)
    bbox = draw.textbbox((0, 0), text)
    tx = max(0, (width - (bbox[2] - bbox[0])) // 2)
    draw.text((tx, 5), text, fill=(255, 230, 80))
    return np.asarray(bar)


def save_comparison_image(gt_ids: torch.Tensor, gen_ids: torch.Tensor, miou: float,
                           num_classes: int, out_path: Path) -> None:
    """GT | GENERATED | ERROR MAP, colorized with this repo's own CARLA
    palette (the single source of truth every other script uses -- see
    grounded_sam_encoder.py's carla_palette_tensor()), so this looks
    exactly like every other seg-map panel in the project, not a
    one-off style."""
    from src.encoders.grounded_sam_encoder import carla_palette_tensor
    from src.encoders.map_colorize import seg_colorize_ids

    palette = carla_palette_tensor()

    def colorize(ids: torch.Tensor) -> np.ndarray:
        ignore_mask = ids >= palette.shape[0]
        safe_ids = torch.where(ignore_mask, torch.zeros_like(ids), ids)
        colour = seg_colorize_ids(safe_ids, palette)[0]  # [3,H,W] in [0,1]
        colour = torch.where(ignore_mask.unsqueeze(0), torch.zeros_like(colour), colour)
        return (colour.permute(1, 2, 0).numpy() * 255).astype(np.uint8)

    gt_rgb = colorize(gt_ids)
    gen_rgb = colorize(gen_ids)

    # Error map: white where classes agree, red where they disagree, black
    # where either side is IGNORE (255) -- excluded pixels shouldn't read
    # as "wrong", they were never scored by compute_miou either.
    ignored = (gt_ids >= num_classes) | (gen_ids >= num_classes)
    mismatch = (gt_ids != gen_ids) & ~ignored
    h, w = gt_ids.shape
    error_rgb = np.zeros((h, w, 3), dtype=np.uint8)
    error_rgb[~ignored.numpy() & ~mismatch.numpy()] = (255, 255, 255)
    error_rgb[mismatch.numpy()] = (220, 20, 20)

    label_row = np.concatenate([
        _label_bar(w, "GROUND TRUTH"),
        _label_bar(w, "GENERATED"),
        _label_bar(w, f"ERROR (mIoU={miou:.3f})"),
    ], axis=1)
    imgs_row = np.concatenate([gt_rgb, gen_rgb, error_rgb], axis=1)
    grid = np.concatenate([label_row, imgs_row], axis=0)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(grid).save(out_path, quality=95)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jsonl", required=True, help="Manifest with one row per image pair.")
    ap.add_argument("--gt_key", default="gt_seg_path", help="Key for the ground-truth segmentation map path.")
    ap.add_argument("--gen_key", default="gen_seg_path", help="Key for the generated-image segmentation map path.")
    ap.add_argument("--num_classes", type=int, default=28,
                     help="Real class count (this branch's CARLA_CLASSES=28). "
                          "255=IGNORE is handled automatically, not counted here.")
    ap.add_argument("--output", default=None, help="Optional path to write a per-image CSV (gt_path,gen_path,miou).")
    ap.add_argument("--viz_dir", default=None,
                     help="If set, save a GT|GENERATED|ERROR comparison image per scored pair "
                          "(up to --viz_limit) into this directory.")
    ap.add_argument("--viz_limit", type=int, default=20,
                     help="Max number of comparison images to render (default 20) -- avoids "
                          "silently rendering thousands of images on a full-dataset run.")
    args = ap.parse_args()

    rows = []
    with open(args.jsonl, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if args.gt_key not in row or args.gen_key not in row:
                print(f"[skip] line {lineno}: missing {args.gt_key!r} or {args.gen_key!r}")
                continue
            rows.append(row)

    if not rows:
        print(f"No valid rows found in {args.jsonl} (looked for keys {args.gt_key!r}/{args.gen_key!r})")
        sys.exit(1)

    viz_dir = Path(args.viz_dir) if args.viz_dir else None
    n_viz_written = 0

    per_image = []
    for i, row in enumerate(rows):
        gt_ids = load_ids(row[args.gt_key])
        gen_ids = load_ids(row[args.gen_key])
        if gt_ids.shape != gen_ids.shape:
            print(f"[skip] row {i}: shape mismatch gt={tuple(gt_ids.shape)} vs gen={tuple(gen_ids.shape)} "
                  f"({row[args.gt_key]} vs {row[args.gen_key]}) -- both must come from the same "
                  f"--width/--height run of grounded_sam_map_calculations.py")
            continue
        miou = compute_miou(gen_ids, gt_ids, args.num_classes)
        per_image.append((row[args.gt_key], row[args.gen_key], miou))
        print(f"[{i + 1}/{len(rows)}] miou={miou:.4f}  gt={row[args.gt_key]}  gen={row[args.gen_key]}")

        if viz_dir is not None and n_viz_written < args.viz_limit:
            stem = Path(row[args.gen_key]).stem
            out_path = viz_dir / f"{n_viz_written:04d}_miou{miou:.3f}_{stem}.jpg"
            save_comparison_image(gt_ids, gen_ids, miou, args.num_classes, out_path)
            n_viz_written += 1

    if not per_image:
        print("No image pairs scored (all rows skipped) -- nothing to report.")
        sys.exit(1)

    mean_miou = sum(m for _, _, m in per_image) / len(per_image)
    print()
    print(f"Scored {len(per_image)}/{len(rows)} pairs.")
    print(f"Mean mIoU: {mean_miou:.4f}")
    if viz_dir is not None:
        print(f"Wrote {n_viz_written} comparison image(s) -> {viz_dir}")

    if args.output:
        out_path = Path(args.output)
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["gt_seg_path", "gen_seg_path", "miou"])
            for gt_path, gen_path, miou in per_image:
                writer.writerow([gt_path, gen_path, f"{miou:.6f}"])
            writer.writerow(["MEAN", "", f"{mean_miou:.6f}"])
        print(f"Wrote per-image results -> {out_path}")


if __name__ == "__main__":
    main()
