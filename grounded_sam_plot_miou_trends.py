"""
grounded_sam_plot_miou_trends.py
-----------------------------------
Plot mIoU TRENDS ACROSS multiple checkpoints/epochs -- a different question
than grounded_sam_compute_miou.py's per-image scoring of ONE checkpoint.
Reads one GT/generated jsonl per checkpoint (same format and same
gt_seg_path/gen_seg_path convention grounded_sam_compute_miou.py uses),
scores each checkpoint's mean mIoU AND per-class IoU, then saves two plots:

  1. mean_miou_trend.png  -- line chart, mean mIoU vs checkpoint/epoch.
     Shows whether controllability is actually improving over training.
  2. per_class_iou_heatmap.png -- heatmap, class (x-axis, labeled with
     real CARLA class names) vs checkpoint/epoch (y-axis), IoU as color.
     Shows WHICH classes are learned well vs poorly, and how that changes
     over training -- a single mean number can hide a model that's great
     at "road"/"sky" but has learned nothing about "bicycle"/"rider".

QUICK COMMANDS (run from repo root with conda loradapter env active):
  # --- One jsonl per checkpoint, explicit labels ---
  python grounded_sam_plot_miou_trends.py \\
      --jsonl outputs/eval/epoch00.jsonl --label epoch00 \\
      --jsonl outputs/eval/epoch05.jsonl --label epoch05 \\
      --jsonl outputs/eval/epoch10.jsonl --label epoch10 \\
      --output_dir miou_trends

  # --- Labels auto-derived from each jsonl's own filename (no --label needed) ---
  python grounded_sam_plot_miou_trends.py \\
      --jsonl outputs/eval/epoch=00-step=0002000.jsonl \\
      --jsonl outputs/eval/epoch=10-step=0022000.jsonl \\
      --output_dir miou_trends

Pass --jsonl once per checkpoint you want on the trend line, in whatever
order you want them plotted (NOT auto-sorted -- pass them in the order
you want, e.g. earliest checkpoint first). Each --jsonl uses the SAME
format as grounded_sam_compute_miou.py: rows of
{"gt_seg_path": "...", "gen_seg_path": "..."} (--gt_key/--gen_key
override the key names, same as that script).

Pass --label right after each --jsonl to name that checkpoint's row/point
explicitly; if you omit --label for a given --jsonl, its own filename stem
is used instead (e.g. "epoch=00-step=0002000.jsonl" -> "epoch=00-step=0002000").
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.encoders.grounded_sam_encoder import CARLA_CLASSES

PROJECT_ROOT = Path(__file__).resolve().parent


def load_ids(path: str) -> torch.Tensor:
    p = Path(path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    if not p.is_file():
        raise FileNotFoundError(f"segmentation map not found: {p}")
    arr = np.asarray(Image.open(p).convert("L"), dtype=np.int64)
    return torch.from_numpy(arr)


def per_class_iou(pred_ids: torch.Tensor, target_ids: torch.Tensor, num_classes: int) -> np.ndarray:
    """[num_classes] array, NaN for a class absent from both pred and target
    in this one image (same "don't dilute the score" rule as
    src.utils.compute_miou, kept per-class here instead of averaged away
    so the heatmap can show it) -- NaN cells render as blank/masked in the
    heatmap, not as a fake 0.0 that would look like a real failure."""
    pred_ids = pred_ids.flatten()
    target_ids = target_ids.flatten()
    ious = np.full(num_classes, np.nan, dtype=np.float64)
    for c in range(num_classes):
        pred_c = pred_ids == c
        target_c = target_ids == c
        if not pred_c.any() and not target_c.any():
            continue
        intersection = (pred_c & target_c).sum().item()
        union = (pred_c | target_c).sum().item()
        ious[c] = intersection / union if union > 0 else 0.0
    return ious


def score_checkpoint(jsonl_path: Path, gt_key: str, gen_key: str, num_classes: int) -> np.ndarray:
    """Returns [num_classes] array: this checkpoint's per-class IoU,
    averaged (nan-aware) across every scored image pair in the jsonl."""
    rows = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    per_image = []
    for row in rows:
        if gt_key not in row or gen_key not in row:
            continue
        gt_ids = load_ids(row[gt_key])
        gen_ids = load_ids(row[gen_key])
        if gt_ids.shape != gen_ids.shape:
            print(f"  [skip] shape mismatch: {row[gt_key]} vs {row[gen_key]}")
            continue
        per_image.append(per_class_iou(gen_ids, gt_ids, num_classes))

    if not per_image:
        raise ValueError(f"no valid pairs scored in {jsonl_path}")

    stacked = np.stack(per_image, axis=0)  # [n_images, num_classes]
    # A class absent from every image in this jsonl is an all-NaN column --
    # expected and handled (renders as blank in the heatmap), but numpy's
    # nanmean still warns "Mean of empty slice" on those columns even
    # though np.errstate(invalid=...) doesn't cover that specific
    # RuntimeWarning category. Silenced deliberately here, not globally.
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Mean of empty slice")
        return np.nanmean(stacked, axis=0)  # [num_classes], NaN where no image ever had that class


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jsonl", action="append", required=True, dest="jsonls",
                     help="One GT/generated jsonl per checkpoint; repeat once per checkpoint, in plot order.")
    ap.add_argument("--label", action="append", default=[], dest="labels",
                     help="Label for the checkpoint named by the --jsonl immediately before it. "
                          "Optional -- defaults to that jsonl's own filename stem.")
    ap.add_argument("--gt_key", default="gt_seg_path")
    ap.add_argument("--gen_key", default="gen_seg_path")
    ap.add_argument("--num_classes", type=int, default=28)
    ap.add_argument("--output_dir", default="miou_trends")
    args = ap.parse_args()

    if len(args.labels) not in (0, len(args.jsonls)):
        raise SystemExit(
            f"Got {len(args.jsonls)} --jsonl but {len(args.labels)} --label -- "
            f"pass either zero --label (auto-derive all) or exactly one per --jsonl."
        )
    labels = args.labels if args.labels else [Path(j).stem for j in args.jsonls]

    per_class_by_checkpoint = []
    mean_by_checkpoint = []
    for jsonl_path, label in zip(args.jsonls, labels):
        print(f"Scoring {label} ({jsonl_path}) ...")
        classes = score_checkpoint(Path(jsonl_path), args.gt_key, args.gen_key, args.num_classes)
        per_class_by_checkpoint.append(classes)
        with np.errstate(invalid="ignore"):
            mean_by_checkpoint.append(float(np.nanmean(classes)))
        print(f"  mean mIoU: {mean_by_checkpoint[-1]:.4f}")

    matrix = np.stack(per_class_by_checkpoint, axis=0)  # [n_checkpoints, num_classes]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # 1. Line chart: mean mIoU vs checkpoint.
    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 0.9), 5))
    ax.plot(range(len(labels)), mean_by_checkpoint, marker="o")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_xlabel("checkpoint")
    ax.set_ylabel("mean mIoU")
    ax.set_title("Mean mIoU across checkpoints")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    line_path = output_dir / "mean_miou_trend.png"
    fig.savefig(line_path, dpi=150)
    plt.close(fig)
    print(f"\nWrote {line_path}")

    # 2. Heatmap: class (x, labeled with real names) vs checkpoint (y).
    class_names = list(CARLA_CLASSES[: args.num_classes])
    fig_w = max(8, len(class_names) * 0.4)
    fig_h = max(3, len(labels) * 0.5)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(matrix, aspect="auto", vmin=0.0, vmax=1.0, cmap="viridis")
    ax.set_xticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=90)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("class")
    ax.set_ylabel("checkpoint")
    ax.set_title("Per-class IoU across checkpoints (blank = class never present)")
    fig.colorbar(im, ax=ax, label="IoU")
    fig.tight_layout()
    heatmap_path = output_dir / "per_class_iou_heatmap.png"
    fig.savefig(heatmap_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {heatmap_path}")


if __name__ == "__main__":
    main()
