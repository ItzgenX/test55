"""
segformer_evaluate.py
---------------------
One-shot evaluation of a finished segformer_inference.py run: mIoU
(controllability) + FID (realism), from the same manifest the inference
run used. Port of the grounded_sam branch's grounded_sam_evaluate.py --
same workflow, same outputs, same matching rules; only the segmenter
differs (SegFormer-B5, 19 Cityscapes classes, instead of Grounding DINO +
SAM, 28 CARLA classes).

WHAT IT DOES
  For every row of --jsonl (the manifest you passed to inference as
  inference.json_file):
    1. Finds that row's generated image(s) inside --run_dir.
    2. Segments each generated image with the SAME SegFormer-B5 model and the
       SAME preprocessing (src.data.transforms.build_seg_preprocess, direct
       resize to 1280x800) that seg_map_calculations.py used for the
       dataset's own maps.
    3. mIoU: generated-image segmentation (NEW) vs. the row's existing
       seg_path (OLD -- the map the image was generated FROM).
  Then, across the whole set:
    4. FID: the real input images (raw_image_path) vs. the generated images.

WHY FID USES IMAGES, NOT SEGMENTATIONS
  FID compares Inception features of two sets of natural RGB images -- it
  answers "do the generated images look like the real ones?". Running it on
  class-ID maps would be meaningless. The segmentation comparison is mIoU.

WHAT THE mIoU NUMBER MEANS
  Both sides come from SegFormer, not human labels, so mIoU here measures
  "does SegFormer see the same layout in the generated image as it saw in
  the real one" -- how well generation followed the conditioning. It is NOT
  accuracy against Cityscapes ground truth. Two numbers are reported:
    - mean per-image mIoU : average of each image's own mIoU (same definition
                            as segformer_inference.py's inline metrics.txt)
    - dataset mIoU        : per-class IoU from intersections/unions summed over
                            ALL pixels of ALL images, then averaged over classes
  NOT directly comparable with grounded_sam's mIoU: different segmenter,
  different class list (19 vs 28), different difficulty. Compare checkpoints
  WITHIN a branch with it; FID on the same test images is the number that is
  comparable ACROSS the two branches.

HOW GENERATED IMAGES ARE MATCHED TO MANIFEST ROWS
  By the seg map's file stem S = Path(seg_path).stem, searched recursively
  under --run_dir. Both inference output modes are recognised:
    default mode            : S_predicted.jpg   / S_<k>_predicted.jpg
    save_generated_only=true: S.png             / S_<k>.png
  (<k> = sample index when inference.n_samples > 1; every sample is scored.)
  RAW SEG GEN (empty-prompt) images are deliberately NOT scored.
  Run with --dry_run first: matching only, no model loaded.

OUTPUTS (default --out_dir = <run_dir>/eval/)
  gen_seg/<gen stem>_gen_seg.png  segmentation of each generated image (raw class
                                   IDs, same format as the dataset's own maps;
                                   reused on re-runs unless --no_skip_existing)
  eval_pairs.jsonl                 one row per generated image: original_image_path,
                                   generated_image_path, gt_seg_path (OLD = dataset's
                                   seg_path), gen_seg_path (NEW), prompt, miou
                                   (same keys as grounded_sam's eval_pairs.jsonl)
  miou_per_image.csv               one row per generated image
  miou_viz/                        OLD SEG | NEW SEG | ERROR panels (first --viz_limit)
  metrics.txt                      both mIoUs, per-class IoU, FID, counts, settings

QUICK COMMANDS (repo root, project env active):
  # 1) check matching only, no GPU work:
  python segformer_evaluate.py \\
      --run_dir outputs/inference_seg/runs/YYYY-MM-DD/HH-MM-SS \\
      --jsonl data/dataset/seg_training_aspect/test.jsonl --dry_run

  # 2) full evaluation:
  python segformer_evaluate.py \\
      --run_dir outputs/inference_seg/runs/YYYY-MM-DD/HH-MM-SS \\
      --jsonl data/dataset/seg_training_aspect/test.jsonl \\
      --fid_weights_path checkpoints/local_models/inception-2015-12-05.pth

  # mIoU only / FID only:
  ... --skip_fid        ... --skip_miou

FID OFFLINE NOTE: torch_fidelity's InceptionV3 weights are not a Hugging
Face model. Download once and pass --fid_weights_path:
  https://github.com/toshas/torch-fidelity/releases/download/v0.2.0/weights-inception-2015-12-05-6726825d.pth
Without it, the first run tries to download via torch.hub.

FID SAMPLE SIZE: FID is a distribution metric -- with fewer than a few
hundred images per side it is noisy and biased upward.
"""
import argparse
import csv
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(os.path.abspath(__file__)).parent
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


# ===================================================================== #
#  Matching generated images to manifest rows                             #
# ===================================================================== #

def _resolve(path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else PROJECT_ROOT / p


def load_manifest(jsonl_path: Path) -> list[dict]:
    rows = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not row.get("seg_path"):
                print(f"[skip] manifest line {lineno}: no 'seg_path'")
                continue
            row["_lineno"] = lineno
            rows.append(row)
    return rows


def index_generated_images(run_dir: Path, exclude_dir: Path) -> list[Path]:
    """Every image file under run_dir, except anything inside our own out_dir
    (so a re-run never picks up its own gen_seg/ or miou_viz/ outputs)."""
    files = []
    for p in run_dir.rglob("*"):
        if p.suffix.lower() not in IMAGE_EXTS or not p.is_file():
            continue
        try:
            p.relative_to(exclude_dir)
            continue
        except ValueError:
            pass
        files.append(p)
    return files


def match_rows(rows: list[dict], gen_files: list[Path]) -> list[dict]:
    """Attach row['_gen_paths'] = sorted list of generated images for that row."""
    by_name = {}
    for p in gen_files:
        by_name.setdefault(p.name, []).append(p)
    names = list(by_name)

    for row in rows:
        stem = Path(row["seg_path"].replace("\\", "/")).stem
        pat = re.compile(rf"^{re.escape(stem)}(_\d+)?(_predicted\.(jpg|jpeg|png)|\.(png|jpg|jpeg))$",
                         re.IGNORECASE)
        hits = []
        for n in names:
            if pat.match(n):
                hits.extend(by_name[n])
        row["_stem"] = stem
        row["_gen_paths"] = sorted(hits)
    return rows


# ===================================================================== #
#  mIoU                                                                    #
# ===================================================================== #

def load_ids(path: Path, size_wh: tuple[int, int]) -> torch.Tensor:
    """Raw class-ID PNG -> [H, W] long. NEAREST resize only if the stored map
    isn't already size_wh (averaging class IDs would invent classes)."""
    img = Image.open(path).convert("L")
    if img.size != size_wh:
        img = img.resize(size_wh, Image.NEAREST)
    return torch.from_numpy(np.asarray(img, dtype=np.int64))


def segment_image(encoder, preprocess, img_path: Path, device) -> torch.Tensor:
    """Same path as seg_map_calculations.py's precompute: build_seg_preprocess
    (direct resize + [-1, 1] normalize) -> encoder.label_ids, so generated-image
    maps come from the identical process as the dataset's maps."""
    img = Image.open(img_path).convert("RGB")
    t = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        return encoder.label_ids(t)[0].cpu().long()


def per_class_counts(pred: torch.Tensor, target: torch.Tensor, num_classes: int):
    """Per-class (intersection, union) pixel counts -- same semantics as
    compute_miou (an id >= num_classes never matches a real class)."""
    pred, target = pred.flatten(), target.flatten()
    inter = np.zeros(num_classes, dtype=np.int64)
    union = np.zeros(num_classes, dtype=np.int64)
    for c in range(num_classes):
        p, t = pred == c, target == c
        inter[c] = (p & t).sum().item()
        union[c] = (p | t).sum().item()
    return inter, union


def _label_bar(width: int, text: str, bar_h: int = 28) -> np.ndarray:
    bar = Image.new("RGB", (width, bar_h), color=(25, 25, 25))
    draw = ImageDraw.Draw(bar)
    bbox = draw.textbbox((0, 0), text)
    draw.text((max(0, (width - (bbox[2] - bbox[0])) // 2), 5), text, fill=(255, 230, 80))
    return np.asarray(bar)


def save_comparison_image(gt_ids, gen_ids, miou, num_classes, out_path: Path) -> None:
    """OLD SEG | NEW SEG | ERROR, coloured with this branch's own Cityscapes
    palette (src.encoders.seg_encoder, the same one every other script here
    uses). Error map: white = same class, red = different, black = id outside
    the class range on either side (not scored)."""
    from src.encoders.seg_encoder import seg_palette_tensor, seg_colorize_ids
    palette = seg_palette_tensor()

    def colorize(ids):
        bad = ids >= palette.shape[0]
        colour = seg_colorize_ids(torch.where(bad, torch.zeros_like(ids), ids), palette)
        colour = colour[0] if colour.dim() == 4 else colour            # [3, H, W]
        colour = torch.where(bad.unsqueeze(0), torch.zeros_like(colour), colour)
        return (colour.permute(1, 2, 0).numpy() * 255).astype(np.uint8)

    ignored = (gt_ids >= num_classes) | (gen_ids >= num_classes)
    mismatch = (gt_ids != gen_ids) & ~ignored
    h, w = gt_ids.shape
    err = np.zeros((h, w, 3), dtype=np.uint8)
    err[(~ignored & ~mismatch).numpy()] = (255, 255, 255)
    err[mismatch.numpy()] = (220, 20, 20)

    labels = np.concatenate([_label_bar(w, "OLD SEG (dataset)"), _label_bar(w, "NEW SEG (generated)"),
                             _label_bar(w, f"ERROR (mIoU={miou:.3f})")], axis=1)
    imgs = np.concatenate([colorize(gt_ids), colorize(gen_ids), err], axis=1)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.concatenate([labels, imgs], axis=0)).save(out_path, quality=95)


# ===================================================================== #
#  FID                                                                     #
# ===================================================================== #

class _ResizedImageDataset(torch.utils.data.Dataset):
    """uint8 [3, H, W] tensors at one fixed size -- what torch_fidelity expects
    from a Dataset input. Real images get the same direct (no crop/pad) resize
    training used, so the real set matches what the model was trained to produce."""

    def __init__(self, paths: list[Path], size_wh: tuple[int, int]):
        self.paths = paths
        self.size_wh = size_wh

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        if img.size != self.size_wh:
            img = img.resize(self.size_wh, Image.BICUBIC)
        return torch.from_numpy(np.array(img, dtype=np.uint8)).permute(2, 0, 1).contiguous()


def compute_fid(real_paths, gen_paths, size_wh, device, batch_size, weights_path):
    import torch_fidelity
    metrics = torch_fidelity.calculate_metrics(
        input1=_ResizedImageDataset(real_paths, size_wh),
        input2=_ResizedImageDataset(gen_paths, size_wh),
        cuda=str(device).startswith("cuda"),
        batch_size=batch_size,
        fid=True,
        verbose=False,
        feature_extractor_weights_path=weights_path,
    )
    return float(metrics[torch_fidelity.KEY_METRIC_FID])


# ===================================================================== #
#  Main                                                                    #
# ===================================================================== #

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run_dir", required=True, help="segformer_inference.py output folder (its Hydra run dir).")
    ap.add_argument("--jsonl", required=True, help="The manifest that inference run used (inference.json_file).")
    ap.add_argument("--out_dir", default=None, help="Where results go (default: <run_dir>/eval).")
    ap.add_argument("--width", type=int, default=1280,
                    help="MUST match the size the dataset maps were computed at (default 1280).")
    ap.add_argument("--height", type=int, default=800,
                    help="MUST match the size the dataset maps were computed at (default 800).")
    ap.add_argument("--local_files_only", type=lambda s: s.lower() != "false", default=True)
    ap.add_argument("--seg_model_path", default="checkpoints/local_models/segformer-b5-cityscapes")
    ap.add_argument("--seg_model_name", default="nvidia/segformer-b5-finetuned-cityscapes-1024-1024")
    ap.add_argument("--device", default=None, help="Auto-detects CUDA if omitted; 'cpu' to force CPU.")
    ap.add_argument("--no_skip_existing", action="store_true",
                    help="Recompute generated-image segmentations even if gen_seg/ already has them.")
    ap.add_argument("--viz_limit", type=int, default=20, help="Max OLD|NEW|ERROR panels to save (0 = none).")
    ap.add_argument("--fid_weights_path", default=None, help="Local InceptionV3 weights for torch_fidelity.")
    ap.add_argument("--fid_batch_size", type=int, default=16)
    ap.add_argument("--skip_miou", action="store_true")
    ap.add_argument("--skip_fid", action="store_true")
    ap.add_argument("--dry_run", action="store_true",
                    help="Only match manifest rows to generated images and report; load no models.")
    args = ap.parse_args()

    run_dir = _resolve(args.run_dir).resolve()
    jsonl = _resolve(args.jsonl).resolve()
    out_dir = (_resolve(args.out_dir) if args.out_dir else run_dir / "eval").resolve()
    size_wh = (args.width, args.height)
    if args.width % 32 or args.height % 32:
        sys.exit(f"--width/--height ({args.width}x{args.height}) must be divisible by 32 (SDXL).")
    if not run_dir.is_dir():
        sys.exit(f"--run_dir not found: {run_dir}")
    if not jsonl.is_file():
        sys.exit(f"--jsonl not found: {jsonl}")

    # ── 1. Match ────────────────────────────────────────────────────────────
    rows = match_rows(load_manifest(jsonl), index_generated_images(run_dir, out_dir))
    matched = [r for r in rows if r["_gen_paths"]]
    missing = [r for r in rows if not r["_gen_paths"]]
    n_gen = sum(len(r["_gen_paths"]) for r in matched)

    print("=" * 64)
    print("  segformer_evaluate.py")
    print(f"  run_dir : {run_dir}")
    print(f"  jsonl   : {jsonl}  ({len(rows)} rows)")
    print(f"  out_dir : {out_dir}")
    print(f"  matched : {len(matched)}/{len(rows)} rows -> {n_gen} generated image(s)")
    print("=" * 64)
    for r in missing[:20]:
        print(f"  [no generated image] line {r['_lineno']}: {r['_stem']}")
    if len(missing) > 20:
        print(f"  ... and {len(missing) - 20} more rows without a generated image")

    if args.dry_run:
        for r in matched[:10]:
            print(f"  {r['_stem']}  ->  {[str(p.relative_to(run_dir)) for p in r['_gen_paths']]}")
        print("\n[dry_run] matching only -- no models loaded, nothing written.")
        return
    if not matched:
        sys.exit("No generated images matched any manifest row -- check --run_dir/--jsonl (try --dry_run).")

    out_dir.mkdir(parents=True, exist_ok=True)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cpu" and not args.device:
        print("[WARN] No CUDA device found -- running on CPU (slow). Pass --device cpu to silence this.")

    summary = [
        "segformer_evaluate.py results",
        f"timestamp      : {time.strftime('%Y-%m-%dT%H:%M:%S')}",
        f"run_dir        : {run_dir}",
        f"jsonl          : {jsonl}",
        f"size           : {args.width}x{args.height}",
        f"rows matched   : {len(matched)}/{len(rows)}  ({n_gen} generated images)",
        "",
    ]

    # ── 2. mIoU ─────────────────────────────────────────────────────────────
    if not args.skip_miou:
        from src.encoders.seg_encoder import SegmentationEncoder
        from src.data.transforms import build_seg_preprocess
        from src.utils import compute_miou

        if args.local_files_only:
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
            seg_model = str(PROJECT_ROOT / args.seg_model_path)
        else:
            seg_model = args.seg_model_name
        print(f"[model] segformer = {seg_model}")
        encoder = SegmentationEncoder(size=size_wh, model=seg_model,
                                      local_files_only=args.local_files_only).to(device)
        encoder.eval()
        preprocess = build_seg_preprocess(size=size_wh, resize_mode="aspect")
        nc = int(encoder.num_classes)
        id2label = getattr(encoder.segformer.config, "id2label", {}) or {}
        class_names = [str(id2label.get(c, id2label.get(str(c), f"class_{c}"))) for c in range(nc)]

        gen_seg_dir = out_dir / "gen_seg"
        viz_dir = out_dir / "miou_viz"
        gen_seg_dir.mkdir(exist_ok=True)

        tot_inter = np.zeros(nc, dtype=np.int64)
        tot_union = np.zeros(nc, dtype=np.int64)
        per_image = []
        pairs = []      # -> eval_pairs.jsonl (old seg path + new seg path per generated image)
        n_viz = 0
        n_failed = 0
        jobs = [(r, g) for r in matched for g in r["_gen_paths"]]
        for i, (row, gen_path) in enumerate(jobs, start=1):
            gt_path = _resolve(row["seg_path"])
            if not gt_path.is_file():
                print(f"[skip] dataset seg map missing: {gt_path}")
                n_failed += 1
                continue
            gen_seg_path = gen_seg_dir / f"{gen_path.stem}_gen_seg.png"
            try:
                if gen_seg_path.is_file() and not args.no_skip_existing:
                    gen_ids = load_ids(gen_seg_path, size_wh)
                else:
                    gen_ids = segment_image(encoder, preprocess, gen_path, device)
                    Image.fromarray(gen_ids.numpy().astype("uint8"), mode="L").save(gen_seg_path)
                gt_ids = load_ids(gt_path, size_wh)
            except Exception as e:  # one bad file must not kill a long run
                print(f"[skip] {gen_path.name}: {type(e).__name__}: {e}")
                n_failed += 1
                continue

            miou = compute_miou(gen_ids, gt_ids, nc)
            inter, union = per_class_counts(gen_ids, gt_ids, nc)
            tot_inter += inter
            tot_union += union
            per_image.append([row["_stem"], str(gen_path), str(gt_path), str(gen_seg_path), miou])
            rp = row.get("raw_image_path")
            pairs.append({
                "original_image_path": str(_resolve(rp)) if rp else "",
                "generated_image_path": str(gen_path),
                "gt_seg_path": str(gt_path),
                "gen_seg_path": str(gen_seg_path),
                "prompt": row.get("prompt", ""),
                "miou": round(miou, 6),
            })
            print(f"[{i}/{len(jobs)}] mIoU={miou:.4f}  {gen_path.name}")

            if n_viz < args.viz_limit:
                save_comparison_image(gt_ids, gen_ids, miou, nc,
                                      viz_dir / f"{n_viz:04d}_miou{miou:.3f}_{gen_path.stem}.jpg")
                n_viz += 1

        del encoder
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()

        with open(out_dir / "eval_pairs.jsonl", "w", encoding="utf-8") as f:
            for p in pairs:
                f.write(json.dumps(p) + "\n")
        print(f"[pairs] wrote {len(pairs)} rows -> {out_dir / 'eval_pairs.jsonl'}")

        with open(out_dir / "miou_per_image.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["stem", "generated_image", "dataset_seg", "generated_seg", "miou"])
            for r in per_image:
                w.writerow(r[:4] + [f"{r[4]:.6f}"])

        if per_image:
            mean_img = float(np.mean([r[4] for r in per_image]))
            present = tot_union > 0
            class_iou = np.where(present, tot_inter / np.maximum(tot_union, 1), np.nan)
            dataset_miou = float(np.nanmean(class_iou)) if present.any() else 0.0
            summary += [
                f"mIoU scored    : {len(per_image)} generated images ({n_failed} skipped)",
                f"mean per-image mIoU : {mean_img:.4f}",
                f"dataset mIoU        : {dataset_miou:.4f}",
                "",
                "per-class IoU (dataset-level; '-' = class absent from both sides everywhere):",
            ]
            for c in range(nc):
                v = "-" if not present[c] else f"{class_iou[c]:.4f}"
                summary.append(f"  {c:>2} {class_names[c]:<20} {v}")
            summary.append("")
            print(f"\nmean per-image mIoU: {mean_img:.4f}   dataset mIoU: {dataset_miou:.4f}")
        else:
            summary += ["mIoU: no image could be scored", ""]

    # ── 3. FID ──────────────────────────────────────────────────────────────
    if not args.skip_fid:
        real_paths, n_real_missing = [], 0
        for r in matched:
            rp = r.get("raw_image_path")
            if rp and _resolve(rp).is_file():
                real_paths.append(_resolve(rp))
            else:
                n_real_missing += 1
        gen_paths = [g for r in matched for g in r["_gen_paths"]]
        print(f"\n[FID] real set: {len(real_paths)} images ({n_real_missing} missing raw_image_path)"
              f"  |  generated set: {len(gen_paths)} images")
        if len(real_paths) < 2 or len(gen_paths) < 2:
            print("[FID] need at least 2 images per side -- skipped.")
            summary += ["FID: skipped (fewer than 2 images on a side)", ""]
        else:
            small = min(len(real_paths), len(gen_paths)) < 100
            if small:
                print("[FID] WARNING: <100 images on a side -- the value will be noisy and biased high.")
            fid = compute_fid(real_paths, gen_paths, size_wh, device,
                              args.fid_batch_size, args.fid_weights_path)
            print(f"[FID] {fid:.4f}")
            summary += [
                f"FID            : {fid:.4f}"
                + ("   (WARNING: <100 images per side, noisy -- compare runs on the same set only)" if small else ""),
                f"FID sets       : real={len(real_paths)}  generated={len(gen_paths)}",
                "",
            ]

    (out_dir / "metrics.txt").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print(f"\nDone. Results -> {out_dir}")


if __name__ == "__main__":
    main()
