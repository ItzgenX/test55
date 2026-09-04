"""
seg_map_calculations.py
-----------------------------
STAGE C — pre-compute and CACHE semantic-segmentation maps for training images
using SegFormer-b5-Cityscapes, then build seg_training/{train,val,test}.json.

This mirrors the structure of a depth-map calculation script (relative-path-
preserving output tree, one-pass atomic JSON building, loud self-verification)
but is its own file, kept separate from depth. The seg-specific differences,
each deliberate:

  • OUTPUT = raw class-ID 8-bit PNG (values 0..18), NOT a grayscale depth ramp.
    We store the raw IDs (canonical, hand-editable, re-palette-able) and let the
    dataset colourise them at LOAD time with SEG_CITYSCAPES_PALETTE (the SSOT in
    src/encoders/seg_encoder.py). This script never writes colour — only labels.

  • The SegmentationEncoder (in src/encoders/seg_encoder.py) is IMPORTED here and
    used via label_ids(), so the maps saved for training are produced by exactly the
    same _predict_ids() code the live inference encoder runs — the train/inference
    parity rule.

  • local_files_only is threaded from CLI to the encoder, matching depth's pattern.
    Default = True (offline; the b5 checkpoint must be in checkpoints/local_models/).

LOCKED MODEL: nvidia/segformer-b5-finetuned-cityscapes-1024-1024
  (b5, not b0 — measured mIoU 0.76/0.66 train/val on real Cityscapes ground
  truth in this repo, and b5 is the documented highest-accuracy SegFormer
  variant; see check_seg_accuracy.py)

RESIZE_MODE:
  --resize_mode aspect (only mode supported) -- non-square direct resize to
  an explicit --width/--height target, no pad, no crop (see below). UNLIKE
  the grounded_sam branch, this gets BAKED INTO THE SAVED MAP -- output
  folders/filenames are mode-named, and segformer_training.py/
  segformer_inference.py must be configured with the SAME resize_mode used here.

  # --- Non-square target (no pad band), matching a 1280x800 source ---
  python seg_map_calculations.py --data_dir data/ --resize_mode aspect --width 512 --height 320

TYPICAL WORKFLOW (data_dir mode — recommended, mirrors depth):
  # builds data/seg_training_aspect/{train,val,test}.jsonl from data/{train,val,test}.jsonl
  python seg_map_calculations.py --data_dir data/ --resize_mode aspect --width 512 --height 320

  # then train (same resize_mode):
  python segformer_training.py experiment=train_seg resize_mode=aspect

QUICK COMMANDS (run from repo root with conda loradapter env active):
  # --- SINGLE IMAGE: one new CARLA/real-world photo -> map saved BESIDE it ---
  #     (<stem>_seg_map_<resize_mode>.png in the image's own folder; prints
  #      the ready-to-run segformer_inference.py command for the pair)
  python seg_map_calculations.py --image path/to/frame.jpg --resize_mode aspect --width 512 --height 320

  # --- SINGLE JSONL: entries with raw_image_path (+ prompt) -> maps in a
  #     SIBLING <images_root>_seg_map_<resize_mode>/ folder (mirrored structure)
  #     + a new <stem>_seg.jsonl beside the input with the STANDARD keys
  #     raw_image_path / seg_path / prompt (self-verified) ---
  python seg_map_calculations.py --json_file data/my_frames.jsonl --resize_mode aspect --width 512 --height 320

  # --- Dry run: 15 images, verify pipeline before committing to full dataset ---
  python seg_map_calculations.py --data_dir data/ --dry_run_n 15

  # --- Full run: all images (639 train + 137 val + 137 test) ---
  python seg_map_calculations.py --data_dir data/

  # --- Re-compute everything (force overwrite of existing PNGs) ---
  python seg_map_calculations.py --data_dir data/ --no_skip

  # --- Dataset-SCAN mode: dataset lives outside the repo; scans for raw_image.jpg,
  #     saves seg maps to a SIBLING folder (mirrored structure), rebuilds
  #     seg_training_<mode>/*.jsonl from data/{train,val,test}.jsonl ---
  python seg_map_calculations.py --dataset_dir /path/to/custome_dataset --data_dir data/ --image_path target

JSON ENTRY FORMAT produced:
  {"raw_image_path": "data/raw/000417/raw_image.jpg",
   "seg_path":       "data/raw_seg/000417/raw_image.png",
   "prompt":         "..."}

GPU / DEVICE:
  --device defaults to auto-detect: uses the first visible CUDA GPU, and
  REFUSES to run (raises with a fix checklist) if none is visible -- this
  script never silently falls back to CPU. Pass --device cpu to explicitly
  opt into CPU, or --device cuda:N to pin a specific GPU on a multi-GPU box.
  --batch_size defaults to auto-scaling from the detected GPU's VRAM (see
  src/utils.py auto_batch_size()) -- pass --batch_size explicitly to disable
  auto-scaling and use an exact value.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from src.data.transforms import build_seg_preprocess
from src.encoders.seg_encoder import SEG_CITYSCAPES_PALETTE, SegmentationEncoder
from src.utils import resolve_device, auto_batch_size

# LOCKED model (b5, not b0 — see check_seg_accuracy.py for measured mIoU). Use --local_files_only False for first download.
DEFAULT_SEG_MODEL = "nvidia/segformer-b5-finetuned-cityscapes-1024-1024"

PROJECT_ROOT = Path(os.path.abspath(__file__)).parent
SCRIPT_NAME = "seg_map_calc"


class _Tee:
    """Duplicates every write to stdout/stderr into a real log file too.

    This script's progress/status is entirely print()-based (hundreds of
    call sites) -- converting all of them to a logging.Logger would be a
    large, risky rewrite for a diagnostic-only script. A tee gets the actual
    goal (every run leaves a persistent, re-readable log under outputs/logs/,
    not just console output that vanishes when the terminal closes) without
    touching any of those call sites. Same mandatory-logging guarantee
    grounded_sam_map_calculations.py's logging.FileHandler setup provides,
    just via a different, lower-risk mechanism for a print()-based script.
    """

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()


def _setup_logging(log_dir: Path) -> Path:
    """Mandatory console+file logging -- every run writes a timestamped log
    under outputs/logs/, mirroring grounded_sam_map_calculations.py's own
    mandatory-logging guarantee (see _Tee docstring for why the mechanism
    differs here)."""
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"{SCRIPT_NAME}_{ts}.log"
    log_file = open(log_path, "w", encoding="utf-8")
    # stdout only, NOT stderr: tqdm's progress bar writes carriage-return
    # (\r) updates to stderr by default -- teeing that into a plain text
    # file would spam it with hundreds of overlapping redraws instead of a
    # readable log. print()'s actual status/summary lines (the useful
    # content) all go to stdout, which this does capture.
    sys.stdout = _Tee(sys.__stdout__, log_file)
    print(f"Log file: {log_path}")
    return log_path


def parse_bool(value):
    """Parse True/False (any case, plus 1/0, yes/no) from the CLI into a bool."""
    if isinstance(value, bool):
        return value
    s = str(value).strip().lower()
    if s in ("true", "1", "yes", "y", "t"):
        return True
    if s in ("false", "0", "no", "n", "f"):
        return False
    raise argparse.ArgumentTypeError(
        f"--local_files_only expects True or False, got: {value!r}"
    )


# ============================================================================ #
#  HELPERS (mirrors depth_map_calculations.py — adapted, not copy-pasted)      #
# ============================================================================ #

def _get_image_path(entry: dict, image_path: str = "source") -> str:
    """
    Return the source image path from a JSONL entry.

    Checks `image_path` first (default "source", override with --image_path).
    Falls back to "raw_image_path" so seg_training/ JSONLs also work.
    Change the key your manifests use without touching any other code:
        python seg_map_calculations.py --data_dir data/ --image_path target
    """
    if image_path in entry:
        return entry[image_path]
    if "raw_image_path" in entry:
        return entry["raw_image_path"]
    raise KeyError(
        f"Entry has neither '{image_path}' nor 'raw_image_path'. Keys: {list(entry.keys())}"
    )


def _seg_out_path(src_path: str, output_dir: Path, input_dir: Path | None) -> Path:
    """
    Compute where to save the seg-ID PNG for src_path, mirroring the input tree.

    When input_dir is given the relative folder structure is preserved:
        input_dir/000417/raw_image.jpg  ->  output_dir/000417/raw_image.png
    This avoids stem collisions when all images share the same filename
    (every scene here is named 'raw_image.jpg', so the sub-folder IS the identity).
    Same logic as depth's _depth_out_path — kept identical because the folder
    layout is identical; only the variable names differ.
    """
    src = Path(src_path)
    if input_dir is not None:
        try:
            rel = src.relative_to(input_dir)
            return output_dir / rel.parent / (rel.stem + ".png")
        except ValueError:
            pass
    # Fallback: keep the image's OWN parent folder in the output path.
    # Returning a bare "<stem>.png" here was the 2026-07 bug — every image is
    # named raw_image.jpg, so all maps collapsed onto one file and overwrote
    # each other. Keeping the parent folder makes each path unique.
    return output_dir / src.parent.name / (src.stem + ".png")


def precompute_segmentation_maps(
    image_paths: list,
    output_dir: Path,
    size: int = 512,
    batch_size: int = 4,
    model_name: str = DEFAULT_SEG_MODEL,
    device: str = "cuda",
    skip_existing: bool = True,
    input_dir: Path = None,
    local_files_only: bool = True,
    out_path_fn=None,
    resize_mode: str = "aspect",
) -> dict:
    """
    Core routine: segment a list of images, save each as a raw class-ID PNG.

    The "segmentation" in the name satisfies the visual-identity rule: you can
    tell at a glance this function belongs to the segmentation pipeline.

    Returns: dict {image path (str) -> saved seg PNG path (str)} for every image
    that was processed or found already cached.

    Notes:
      • RGB preprocessing uses build_seg_preprocess() — the SHARED function
        from src/data/transforms.py — so it is byte-identical to what
        local_seg.py's dataset loader builds for training's RGB image. This is
        what guarantees the map's geometry matches the RGB image it's paired
        with (both squared the SAME way, whichever resize_mode is chosen).
      • encoder.label_ids() returns raw class IDs; we save them as PIL mode "L"
        (8-bit grayscale, values 0..num_classes-1). Colour palette is applied
        later, at load time, by SegJsonDataset._load_seg_colormap().
      • local_files_only: True = load strictly from local disk (offline); the b5
        checkpoint must be in checkpoints/local_models/segformer-b5-cityscapes.
        False = allow downloading from HF hub on first use.
      • out_path_fn: optional callable(image_path: Path) -> Path. When given, it
        FULLY decides where each seg PNG is saved (overrides output_dir/input_dir).
        Used by dataset-scan mode to save the map into a SIBLING folder next to
        the dataset root (mirrored structure), named <folder>_seg_map.png. When
        None, the default _seg_out_path placement is used.
      • resize_mode: "aspect" (only mode supported) — direct resize to an
        explicit --width/--height target before SegFormer sees it, no pad,
        no crop. THIS GETS BAKED INTO THE SAVED MAP: unlike the grounded_sam
        branch (which resizes a native-resolution map live at load time),
        here the map is already at its final size by the time it's written
        to disk, so training/inference must use a map computed with the
        SAME width/height they're configured for. Callers are
        responsible for mode-naming the output folder/filename so two runs
        with different modes never collide or get mixed up.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # One place decides the output path for an image, honouring out_path_fn.
    def _resolve_out(p):
        return out_path_fn(Path(p)) if out_path_fn is not None else _seg_out_path(p, output_dir, input_dir)

    # ---- Collision guard: no two images may save to the SAME PNG --------- #
    # Before doing any GPU work, check where every image WILL be saved. If two
    # images resolve to the same output file, the second would overwrite the
    # first and training would pair images with the wrong map (the 2026-07 bug).
    # Stop loudly and show one colliding pair instead of overwriting silently.
    seen = {}
    for p in image_paths:
        key = os.path.normcase(str(_resolve_out(p)))
        if key in seen:
            raise SystemExit(
                f"[FATAL] Two images would write the SAME seg map file:\n"
                f"    {seen[key]}\n    {p}\n  both -> {key}\n"
                f"  Each image needs its own map. This means the output naming "
                f"lost the per-image folder — check the dataset layout."
            )
        seen[key] = p

    # ---- Load the SAME encoder used at live inference ----------------------- #
    # Using SegmentationEncoder (not SegformerForSemanticSegmentation directly)
    # guarantees the offline maps and the live encoder share _predict_ids() byte-
    # for-byte — the train/inference parity guarantee.
    print(f"\nLoading segmentation encoder: {model_name}")
    print(f"  (local_files_only={local_files_only})")
    encoder = SegmentationEncoder(
        size=size, model=model_name, local_files_only=local_files_only
    )
    encoder = encoder.to(device).eval()
    num_classes = encoder.num_classes
    print(f"Segmentation encoder ready ({num_classes} classes).\n")

    # ---- SHARED RGB preprocessing ----------------------------------------- #
    # build_seg_preprocess is the SINGLE SOURCE OF TRUTH for squaring. Using it
    # here (offline calc) AND in local_seg.py's dataset loader (training) is
    # what guarantees the seg map's geometry matches the RGB image it's paired
    # with. Do not inline this or duplicate it.
    preprocess = build_seg_preprocess(size=size, resize_mode=resize_mode)

    path_to_seg = {}
    processed = skipped = errors = 0

    # tqdm's own bar updates via \\r, which never reaches the log file (see
    # _Tee -- stderr is deliberately not teed to avoid spamming the file with
    # redraws). Print a real progress line into stdout (and therefore the log
    # file) at a fixed number of checkpoints regardless of dataset size, so a
    # `tail -f`/re-opened log always shows real, recent progress -- same fix
    # as grounded_sam_map_calculations.py's precompute_grounded_sam_maps.
    n_batches = max(1, (len(image_paths) + batch_size - 1) // batch_size)
    log_every = max(1, n_batches // 20)

    pbar = tqdm(range(0, len(image_paths), batch_size), desc="Computing seg maps")
    for _batch_i, batch_start in enumerate(pbar, start=1):
        batch_paths = image_paths[batch_start: batch_start + batch_size]

        # Cache-hit check: if the output PNG already exists, reuse it.
        to_process = []
        for p in batch_paths:
            out = _resolve_out(p)
            if skip_existing and out.exists():
                path_to_seg[str(p)] = str(out)
                skipped += 1
            else:
                to_process.append(p)
        if not to_process:
            continue

        tensors, valid_paths = [], []
        for p in to_process:
            # Split into two try/except blocks so a failure tells us WHICH step
            # broke: loading the file (Pillow) vs. preprocessing it
            # (build_seg_square_preprocess's SquarePad/Resize/ToTensor/Normalize
            # chain). The old single try block blamed every failure on "could
            # not load image" even when the real failure was in preprocess().
            try:
                img = Image.open(p).convert("RGB")
            except Exception as e:
                p_obj = Path(p)
                exists = p_obj.exists()
                print(f"\n[WARN] Image LOAD failed (Pillow could not open/decode the file) — tried:\n"
                      f"         {p}\n"
                      f"         reason: {e}\n"
                      f"         path.exists() = {exists}"
                      + ("" if exists else "  -> the path itself is wrong (case, mount point, or "
                         "missing --image_root — depending on which mode you're running)")
                      + (f"\n         file size = {p_obj.stat().st_size} bytes"
                         "  -> 0 bytes or a tiny file usually means a broken symlink or failed copy"
                         if exists else ""))
                errors += 1
                continue

            try:
                tensors.append(preprocess(img))      # [3, size, size] in [-1,1]
                valid_paths.append(p)
            except Exception as e:
                print(f"\n[WARN] Image PREPROCESS failed (SquarePad/Resize/ToTensor/Normalize) "
                      f"— the file loaded fine, the transform chain threw — path:\n"
                      f"         {p}\n"
                      f"         image size/mode: {img.size} {img.mode}\n"
                      f"         reason: {e}")
                errors += 1
        if not tensors:
            continue

        batch_tensor = torch.stack(tensors).to(device)   # [B, 3, size, size]
        with torch.no_grad():
            # label_ids -> [B, size, size] long ids in [0, num_classes-1]
            # This calls _predict_ids(), the SAME path the live encoder uses.
            id_maps = encoder.label_ids(batch_tensor)

        for i, src_path in enumerate(valid_paths):
            ids_np = id_maps[i].to(torch.uint8).cpu().numpy()   # [size, size] uint8

            # Self-check per image: no id may exceed the class table. A bad id
            # here would silently corrupt colourisation at load — fail loudly.
            if ids_np.max() >= num_classes:
                raise ValueError(
                    f"{Path(src_path).name}: predicted id {ids_np.max()} >= "
                    f"num_classes {num_classes}. Encoder/model mismatch."
                )

            out = _resolve_out(src_path)
            out.parent.mkdir(parents=True, exist_ok=True)
            # "L" mode = 8-bit single channel. Values ARE class ids, not brightness.
            Image.fromarray(ids_np, mode="L").save(out)

            path_to_seg[str(src_path)] = str(out)
            processed += 1

        pbar.set_postfix(processed=processed, skipped=skipped, errors=errors)
        if _batch_i % log_every == 0 or _batch_i == n_batches:
            print(f"  [progress] batch {_batch_i}/{n_batches} -- "
                  f"processed={processed} skipped={skipped} errors={errors}")

    print(f"\n{'='*52}")
    print(f"  Processed : {processed} images")
    print(f"  Skipped   : {skipped}   (already existed)")
    print(f"  Errors    : {errors}")
    print(f"  Output    : {output_dir}")
    print(f"{'='*52}\n")
    return path_to_seg


def _verify_segmentation_training_jsonl(
    jsonl_path: Path, num_classes: int
) -> tuple[int, int]:
    """
    Verify every entry in a seg-training output JSONL file.

    The "segmentation" in the name satisfies the visual-identity rule.

    For each entry checks:
      1. All three required keys present: raw_image_path, seg_path, prompt.
      2. seg_path exists on disk (the PNG was actually written).
      3. seg_path.stem == raw_image_path.stem — guards against any index
         mismatch that would silently pair the wrong seg map with an image.
      4. The PNG opens, is single-channel, and contains only valid class ids
         (< num_classes). This is seg-specific: continuous depth doesn't need
         this check; discrete class IDs can go out of range silently.

    Returns (n_passed, n_failed) and prints a one-line PASS/FAIL summary.
    """
    cwd = Path.cwd()
    with open(jsonl_path, "r", encoding="utf-8") as f:
        entries = [json.loads(line) for line in f if line.strip()]

    passed = failed = 0
    for i, entry in enumerate(entries):
        missing = [k for k in ("raw_image_path", "seg_path", "prompt")
                   if k not in entry]
        if missing:
            print(f"  [FAIL] entry {i}: missing keys {missing}")
            failed += 1
            continue

        raw_p = Path(entry["raw_image_path"])
        seg_p = Path(entry["seg_path"])

        # seg PNG must exist on disk.
        abs_seg = seg_p if seg_p.is_absolute() else cwd / seg_p
        if not abs_seg.exists():
            print(f"  [FAIL] entry {i}: seg PNG not on disk: {seg_p}")
            failed += 1
            continue

        # Stem must match (raw_image.jpg -> raw_image.png).
        if raw_p.stem != seg_p.stem:
            print(f"  [FAIL] entry {i}: stem mismatch — "
                  f"raw={raw_p.stem!r}  seg={seg_p.stem!r}")
            failed += 1
            continue

        # Seg-specific content check: PNG must be valid labels in [0, num_classes).
        try:
            arr = np.asarray(Image.open(abs_seg).convert("L"))
        except Exception as e:
            print(f"  [FAIL] entry {i}: cannot open seg PNG {seg_p}: {e}")
            failed += 1
            continue
        if arr.max() >= num_classes:
            print(f"  [FAIL] entry {i}: seg id {arr.max()} >= num_classes {num_classes}")
            failed += 1
            continue

        passed += 1

    status = "PASS" if failed == 0 else "FAIL"
    print(f"  [{status}] {jsonl_path.name}: {passed}/{len(entries)} entries valid"
          + (f", {failed} FAILED" if failed else ""))
    return passed, failed


def _find_split_jsonl(data_dir: Path, split: str) -> "Path | None":
    """
    Locate the JSONL file for a given split in data_dir.
    Exact match first (data_dir/{split}.jsonl), then any *.jsonl whose stem
    contains the split name; among those pick the shortest stem.
    """
    exact = data_dir / f"{split}.jsonl"
    if exact.exists():
        return exact
    candidates = sorted(
        [p for p in data_dir.glob("*.jsonl") if split.lower() in p.stem.lower()],
        key=lambda p: len(p.stem),
    )
    return candidates[0] if candidates else None


def build_segmentation_training_jsons(
    data_dir: Path,
    raw_dir: Path,
    seg_dir: Path,
    manifest_out: Path = None,
    size: int = 512,
    batch_size: int = 4,
    model_name: str = DEFAULT_SEG_MODEL,
    device: str = "cuda",
    skip_existing: bool = True,
    subset_n: int = None,
    local_files_only: bool = True,
    image_path: str = "source",
    image_root: Path = None,
    resize_mode: str = "aspect",
    output_root: Path = None,
) -> None:
    """
    Build data/seg_training/{train,val,test}.jsonl from data/{train,val,test}.jsonl.

    This is the --data_dir path. By default it does NOT hardcode an output
    folder for the maps: it looks at the image paths in your JSONLs, finds the
    folder common to all of them (the dataset root, e.g. .../custome_dataset),
    and saves each map into a SIBLING folder next to it
    (.../custome_dataset_seg_map_<resize_mode>/), mirroring the internal
    structure. So every image gets its OWN uniquely-named map and none can
    overwrite another. Pass output_root to redirect the PNGs somewhere else
    entirely instead (mirrors grounded_sam_map_calculations.py's own
    --output_root override) -- the manifest's seg_path values follow
    automatically, since they're built from wherever the maps actually landed.

    manifest_out controls where train/val/test.jsonl are written, independent
    of where the PNGs go -- same two-knob split as
    grounded_sam_map_calculations.py's --output_root/--manifest_out. If
    omitted, defaults to wherever the PNGs actually landed (output_root, or
    its own sibling-of-data_dir default), matching grounded_sam's default
    exactly -- resolved AFTER the PNG location is known (unlike
    grounded_sam's own main(), which can resolve output_root before calling
    its build function since it never needs to actually scan real images to
    do so; this branch's sibling default depends on the real images
    referenced in the manifest, so the same resolution has to happen here
    instead).

    Steps:
      0. Read every split's entries + resolve absolute image paths (in lockstep,
         so entry<->image never drift). Pool all images to find the dataset root.
      1. For each split, compute (or reuse cached) seg-ID PNGs, saved via
         _sibling_map_path — named after the image's FOLDER (unique), never the
         image filename (all 'raw_image.jpg' -> would collide).
      2. Build each output entry atomically from its OWN source entry
         (zip(entries, images)), with ABSOLUTE forward-slash paths:
             raw_image_path, seg_path, prompt (prompt copied verbatim).
      3. Write the manifest and verify it with the full 5-check verifier.
         (Per-pixel class-ID validity is enforced at write time inside
         precompute_segmentation_maps.)

    Args:
      data_dir         : folder with train.jsonl / val.jsonl / test.jsonl.
      manifest_out     : where the three-field manifests are written. None
                         (default) = same folder the PNGs landed in.
      image_path       : key in the source JSONL holding the image path (e.g. "target").
      image_root       : optional root prepended to RELATIVE image paths.
      subset_n         : if set, only the first N entries per split (dry run).
      raw_dir, seg_dir : legacy args, no longer used for saving (the sibling
                         folder is derived from the images). Kept so the CLI
                         stays backward-compatible.
    """
    cwd = Path.cwd()
    _root = image_root if image_root is not None else cwd

    # ---- Step 0: read all splits, resolve images, find ONE dataset root ---- #
    split_entries = {}   # split -> list[entry]
    split_images  = {}   # split -> list[abs Path], aligned 1:1 with entries
    all_images    = []
    for split in ["train", "val", "test"]:
        src_path = _find_split_jsonl(data_dir, split)
        if src_path is None:
            print(f"\n[WARN] No JSONL for split '{split}' found in {data_dir} — skipping.")
            continue
        with open(src_path, "r", encoding="utf-8") as f:
            all_entries = [json.loads(line) for line in f if line.strip()]
        entries = all_entries[:subset_n] if subset_n else all_entries
        abs_paths = []
        for entry in entries:
            p = Path(_get_image_path(entry, image_path))
            abs_p = (p if p.is_absolute() else _root / p).resolve()
            abs_paths.append(abs_p)
            if not abs_p.exists():
                print(f"  [WARN] Image not found on disk: {abs_p}")
        split_entries[split] = entries
        split_images[split]  = abs_paths
        all_images.extend(abs_paths)

    if not all_images:
        print("[ERROR] No images found in any split — nothing to do.")
        return

    # Dataset root: prefer an explicit --raw_dir when EVERY image actually
    # lives under it (a stable, caller-chosen anchor) -- otherwise fall back
    # to the deepest folder shared by all referenced images (no hardcoding).
    # The auto-derived root is fragile in a real way, not just cosmetic: it's
    # computed from whichever images happen to appear in THIS manifest, so
    # two runs against different subsets of the same real dataset (e.g. one
    # split file that only references a single city folder) can each derive
    # a DIFFERENT root and a different sibling location for what should be
    # one consistent dataset -- found by actually running this against
    # Cityscapes' own native layout (leftImg8bit/{split}/{city}/*.png),
    # where every split's images share only ONE common city folder. An
    # explicit --raw_dir (e.g. .../extracted) pins one stable answer instead.
    def _is_relative_to(p: Path, other: Path) -> bool:
        try:
            p.relative_to(other)
            return True
        except ValueError:
            return False

    if raw_dir is not None and raw_dir.exists() and all(_is_relative_to(p, raw_dir) for p in all_images):
        dataset_root = raw_dir
    else:
        dataset_root = Path(os.path.commonpath([str(p) for p in all_images]))
        if dataset_root.is_file():          # only one image -> commonpath is the file itself
            dataset_root = dataset_root.parent
    _suffix = f"_seg_map_{resize_mode}"
    # Explicit output_root wins outright (resolved, same reasoning as
    # grounded_sam_map_calculations.py's own --output_root: a relative path
    # left unresolved would inherit whatever the process cwd happens to be
    # when segformer_training.py's hydra.job.chdir=true later changes it,
    # breaking Image.open(seg_path) on the very first batch). Otherwise fall
    # back to the sibling-of-dataset-root default, unchanged from before.
    if output_root is not None:
        sibling_root = Path(output_root).resolve()
    else:
        sibling_root = dataset_root.parent / (dataset_root.name + _suffix)
    sibling_root.mkdir(parents=True, exist_ok=True)
    print(f"\nDataset root : {dataset_root}")
    print(f"Seg maps     : {sibling_root}  (resize_mode={resize_mode})"
          + ("" if output_root is not None else "  (sibling folder, mirrored structure)"))

    # manifest_out: explicit value wins outright (resolved, same reasoning as
    # output_root above); otherwise defaults to sibling_root itself -- the
    # SAME folder the PNGs just landed in -- matching
    # grounded_sam_map_calculations.py's own manifest_out-defaults-to-
    # output_root behavior exactly. Only resolvable here, not earlier in
    # main(), since sibling_root itself depends on the real images this
    # manifest references (see the docstring's note on this).
    manifest_root = Path(manifest_out).resolve() if manifest_out is not None else sibling_root
    manifest_root.mkdir(parents=True, exist_ok=True)
    print(f"Manifest     : {manifest_root}"
          + ("" if manifest_out is not None else "  (same folder as the seg maps)"))

    # STEM-based naming (mirrors run_json_file_mode's own _out_path), NOT
    # _sibling_map_path's folder-based naming (that one stays as-is -- it's
    # correct for --dataset_dir scan mode's real target layout, one image per
    # folder, always named "raw_image.jpg", where the FOLDER is the only
    # unique identity). Here the source manifest can legitimately have many
    # differently-named images sharing one folder (this exact bug, found by
    # actually running this mode against Cityscapes' native layout: 10 images
    # in one city folder all resolved to the SAME output path and got caught
    # by the collision guard below, not silently overwritten). Full rel_dir
    # is still preserved -- combined with a per-image stem filename, the pair
    # is collision-safe for BOTH shapes: a per-sample subfolder with a
    # repeated stem (rel_dir alone disambiguates), or a shared folder with
    # unique stems (stem alone disambiguates).
    def out_fn(p):
        p = Path(p)
        rel_dir = p.parent.relative_to(dataset_root)
        return sibling_root / rel_dir / (p.stem + _suffix + ".png")

    # ---- Per split: compute maps, then write + verify the manifest -------- #
    for split in ["train", "val", "test"]:
        if split not in split_entries:
            continue
        entries         = split_entries[split]
        abs_image_paths = split_images[split]

        print(f"\n{'='*56}")
        print(f"  {split}: processing {len(entries)} entries"
              + (" (dry-run subset)" if subset_n else ""))
        print(f"{'='*56}")

        path_to_seg = precompute_segmentation_maps(
            image_paths=[str(p) for p in abs_image_paths],
            output_dir=sibling_root,
            size=size,
            batch_size=batch_size,
            model_name=model_name,
            device=device,
            skip_existing=skip_existing,
            local_files_only=local_files_only,
            out_path_fn=out_fn,
            resize_mode=resize_mode,
        )

        # Build entries in ONE pass, each from its OWN source (zip keeps them
        # aligned). ABSOLUTE forward-slash paths: the dataset lives outside the
        # repo, so relative-to-repo paths would break when the repo moves.
        out_entries = []
        n_skipped   = 0
        for entry, abs_img_p in zip(entries, abs_image_paths):
            seg_abs_str = path_to_seg.get(str(abs_img_p))
            if seg_abs_str is None:
                print(f"  [WARN] No seg result for {abs_img_p} — entry skipped.")
                n_skipped += 1
                continue
            # ground_truth: carried through from the input row as-is (never
            # guessed from a naming convention -- a guessed path only holds
            # for one specific local layout). "" if the input row has none.
            # Same resolution/carry-forward discipline as
            # grounded_sam_map_calculations.py's own manifest writer -- this
            # script previously dropped the field entirely, losing the link
            # to real ground truth (e.g. data/custom_dataset's gtFine labels)
            # by the time training's own manifest was written.
            _gt = entry.get("ground_truth", "")
            if _gt and not Path(_gt).is_absolute():
                _gt = (Path.cwd() / _gt).resolve().as_posix()
            out_entries.append({
                "raw_image_path": abs_img_p.as_posix(),
                "seg_path":       Path(seg_abs_str).resolve().as_posix(),
                "prompt":         entry["prompt"],
                "ground_truth":   _gt,
            })

        out_path = manifest_root / f"{split}.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for entry in out_entries:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        skip_note = f"  ({n_skipped} skipped)" if n_skipped else ""
        print(f"\n  Written {len(out_entries)} entries -> {out_path}{skip_note}")
        # name_from="stem": this function names maps after the image's own
        # stem now (see out_fn above), not the folder -- matching
        # run_json_file_mode's convention, unlike --dataset_dir scan mode
        # (default name_from="folder") which this function used to share a
        # naming scheme with but no longer does.
        _verify_scan_seg_training_jsonl(out_path, "seg_path", _suffix, name_from="stem")


# ============================================================================ #
#  DATASET-SCAN MODE (segmentation)                                            #
#  Mirrors depth_map_calculations.py exactly: find raw_image.jpg by SCANNING,  #
#  save each seg map into a SIBLING folder (<dataset_dir>_seg_map/) that       #
#  MIRRORS the internal folder structure; then rebuild the split manifests    #
#  from the originals so every prompt + split assignment stays exactly mapped.#
# ============================================================================ #

def _sibling_map_path(
    image_path: Path, dataset_dir: Path, sibling_root: Path,
    suffix: str, ext: str = ".png",
) -> Path:
    """
    Given an image at    <dataset_dir>/000417/raw_image.jpg
    and sibling_root  =  <dataset_dir's parent>/<dataset_dir.name><suffix>
                         (e.g. .../custome_dataset_seg_map)
    return the map path  <sibling_root>/000417/000417<suffix><ext>

    The output tree MIRRORS dataset_dir's internal structure, living as a
    SIBLING of dataset_dir — the original dataset folder is never written into.
    """
    rel_dir     = image_path.parent.relative_to(dataset_dir)
    folder_name = image_path.parent.name
    return sibling_root / rel_dir / (folder_name + suffix + ext)


def _verify_scan_seg_training_jsonl(jsonl_path: Path, map_key: str, suffix: str,
                                    name_from: str = "folder") -> tuple[int, int]:
    """
    Verify a scan-mode seg output JSONL — the LAST line of defence before
    training trusts this file. Training reads each line as:
        "for THIS image, condition on THIS seg map, with THIS prompt"
    so a wrong line silently poisons training (no crash, no warning — the
    model just learns from mismatched pairs). Every check below therefore
    fails LOUDLY with the entry number and the exact reason.
    Mirrors depth's _verify_scan_training_jsonl exactly (seg twin).

    Per-entry checks (each line of the JSONL):
      1. KEYS      — raw_image_path, <map_key> (here "seg_path"), prompt all
                     present (a missing key would crash the dataloader later,
                     far from the real cause).
      2. ON DISK   — the seg PNG actually exists.
      3. SIBLING   — the map lives in the sibling tree, NOT inside the image's
                     own source folder (source dataset is read-only by rule).
      4. MIRRORED  — map's parent folder name equals the image's parent folder
                     name (e.g. both "000417") and the map file is named
                     <folder_name><suffix>.png — the guarantee that image
                     000417 is paired with 000417's map and no other.

    Whole-file check (across ALL lines together):
      5. UNIQUE    — no two entries may point at the SAME map file.
                     ADDED BECAUSE OF A REAL BUG (2026-07-02): a non-scan run
                     named maps after the image FILENAME; every image here is
                     'raw_image.jpg', so all 913 maps overwrote each other
                     into ONE file and every row referenced that survivor.
                     Checks 1-4 cannot catch that (each line looks fine alone)
                     — only comparing lines against each other exposes it.
                     Duplicate raw_image_path is flagged too (a source image
                     listed twice would be double-weighted in training).

    (Per-pixel class-ID validity (0..18) is already enforced at write time
    inside precompute_segmentation_maps, so it is not re-checked here.)
    Returns (n_passed, n_failed) and prints a one-line PASS/FAIL summary.
    """
    with open(jsonl_path, "r", encoding="utf-8") as f:
        entries = [json.loads(line) for line in f if line.strip()]

    passed = failed = 0
    for i, entry in enumerate(entries):
        # ---- check 1: all three required keys present ---------------------- #
        missing = [k for k in ("raw_image_path", map_key, "prompt") if k not in entry]
        if missing:
            print(f"  [FAIL] entry {i}: missing keys {missing}")
            failed += 1
            continue

        raw_p = Path(entry["raw_image_path"])
        map_p = Path(entry[map_key])

        # ---- check 2: the seg PNG really exists on disk -------------------- #
        if not map_p.exists():
            print(f"  [FAIL] entry {i}: map file not on disk: {map_p}")
            failed += 1
            continue

        # ---- check 3: map is in the SIBLING tree, source folder untouched -- #
        if map_p.parent == raw_p.parent:
            print(f"  [FAIL] entry {i}: map was written INTO the source dataset "
                  f"folder (expected a sibling tree): {map_p.parent}")
            failed += 1
            continue

        # ---- check 4a: mirrored structure — same leaf folder name ---------- #
        # image .../custome_dataset/000417/raw_image.jpg
        # map   .../custome_dataset_seg_map/000417/...   <- "000417" must match
        #
        # ONLY meaningful for name_from="folder" (scan mode): every image
        # there sits one folder below the mirrored sibling tree, by
        # construction (_sibling_map_path always emits <rel_dir>/<folder><suffix>.png).
        # name_from="stem" (json_file mode) has no such guarantee -- when every
        # image in the manifest shares ONE common parent folder (images_root
        # itself, e.g. a Cityscapes-native city folder with many differently
        # -named files), run_json_file_mode's own images_root/sibling_root
        # collapse to that shared folder, so the map's parent becomes
        # <folder>_seg_map_<mode> instead of <folder> -- a real naming
        # difference, not a pairing error (found by actually running this
        # mode against exactly that layout, not by re-reading). Check 3
        # (not written into the source folder) plus check 4b (filename must
        # be <stem><suffix>.png) plus check 5 (global uniqueness) already
        # fully establish correct image<->map pairing for stem mode without
        # this additional, mode-inappropriate requirement.
        if name_from == "folder" and map_p.parent.name != raw_p.parent.name:
            print(f"  [FAIL] entry {i}: mirrored leaf folder name mismatch — "
                  f"image={raw_p.parent.name!r}  map={map_p.parent.name!r}")
            failed += 1
            continue

        # ---- check 4b: map filename matches the mode's naming rule --------- #
        # name_from="folder" (scan mode): <folder_name><suffix>.png — named
        #   after the FOLDER because scan-mode images all share one filename
        #   ('raw_image.jpg'), so the folder is the only unique identity.
        # name_from="stem" (json_file mode): <image_stem><suffix>.png — json
        #   manifests may list many differently-named images per folder, so
        #   the stem is the identity there (check 5 still catches collisions
        #   if stems repeat within a folder).
        base = raw_p.parent.name if name_from == "folder" else raw_p.stem
        expected = base + suffix + ".png"
        if map_p.name != expected:
            print(f"  [FAIL] entry {i}: map name {map_p.name!r} != expected {expected!r}")
            failed += 1
            continue

        passed += 1

    # ---- check 5: GLOBAL uniqueness — the anti-collision check ------------- #
    # In a healthy manifest every image has its OWN map, so every map path
    # appears exactly once. Any count > 1 is the stem-collision bug.
    from collections import Counter
    map_counts = Counter(os.path.normcase(str(Path(e[map_key])))
                         for e in entries if map_key in e)
    img_counts = Counter(os.path.normcase(str(Path(e["raw_image_path"])))
                         for e in entries if "raw_image_path" in e)
    dup_maps = {p: n for p, n in map_counts.items() if n > 1}
    dup_imgs = {p: n for p, n in img_counts.items() if n > 1}
    if dup_maps:
        worst_path, worst_n = max(dup_maps.items(), key=lambda kv: kv[1])
        print(f"  [FAIL] {len(dup_maps)} map file(s) referenced by MULTIPLE entries "
              f"(worst: {worst_n} entries -> {worst_path}). Every image must have "
              f"its OWN map — this is the stem-collision bug; re-run in scan mode "
              f"(--dataset_dir) with --no_skip.")
        failed += sum(dup_maps.values())          # every colliding entry is invalid
        passed = max(0, passed - sum(dup_maps.values()))
    if dup_imgs:
        print(f"  [FAIL] {len(dup_imgs)} source image(s) listed more than once "
              f"(would be double-weighted in training).")
        failed += sum(n - 1 for n in dup_imgs.values())

    status = "PASS" if failed == 0 else "FAIL"
    print(f"  [{status}] {jsonl_path.name}: {passed}/{len(entries)} entries valid"
          + (f", {failed} FAILED" if failed else ""))
    return passed, failed


def build_seg_training_from_scan(
    dataset_dir: Path,
    data_dir: Path,
    output_dir: Path,
    image_name: str = "raw_image.jpg",
    image_path: str = "target",
    size: int = 512,
    batch_size: int = 4,
    model_name: str = DEFAULT_SEG_MODEL,
    device: str = "cuda",
    skip_existing: bool = True,
    local_files_only: bool = True,
    subset_n: int = None,
    resize_mode: str = "aspect",
) -> None:
    """
    DATASET-SCAN MODE for segmentation. Mirrors
    depth_map_calculations.build_depth_training_from_scan step for step:

    1. SCAN dataset_dir recursively for files named exactly `image_name`
       (default 'raw_image.jpg'). Every OTHER file is ignored, so the source
       dataset can freely contain other images per folder.
    2. Compute a seg map for each and save it into a SIBLING folder that
       MIRRORS dataset_dir's internal structure — the source dataset folder
       itself is NEVER written into. Mode-named so runs using different
       resize_mode values never collide:
           dataset_dir  = .../custome_dataset
           sibling_root = .../custome_dataset_seg_map_aspect
           .../custome_dataset/000417/raw_image.jpg
             -> .../custome_dataset_seg_map_aspect/000417/000417_seg_map_aspect.png
    3. Read the original split manifests (train/val/test .jsonl) in data_dir to
       recover each image's PROMPT and SPLIT, matching by ABSOLUTE image path.
    4. Write data/seg_training/{train,val,test}.jsonl with ABSOLUTE
       raw_image_path + seg_path + prompt. Verify each file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Sibling output tree: a NEW folder next to dataset_dir, same name + suffix.
    # Mode-named: the map's squaring is baked in at calc time (unlike the
    # grounded_sam branch), so two different-mode runs must never collide.
    _suffix = f"_seg_map_{resize_mode}"
    sibling_root = dataset_dir.parent / (dataset_dir.name + _suffix)
    sibling_root.mkdir(parents=True, exist_ok=True)
    print(f"Seg maps will be saved to (sibling, mirrored structure, resize_mode={resize_mode}): {sibling_root}")

    # ---- Step 1: SCAN for the target filename only ------------------------- #
    print(f"\nScanning for '{image_name}' under: {dataset_dir}")
    found = sorted(dataset_dir.rglob(image_name))
    if subset_n:
        found = found[:subset_n]
    if not found:
        print(f"[ERROR] No '{image_name}' found under {dataset_dir}. Nothing to do.")
        return
    print(f"Found {len(found)} '{image_name}' file(s)"
          + (f"  (dry-run: scan capped at first {subset_n})" if subset_n else "") + ".")

    # ---- Step 2: compute seg, saving to the mirrored sibling tree ---------- #
    out_fn = lambda p: _sibling_map_path(Path(p), dataset_dir, sibling_root, _suffix)
    path_to_seg = precompute_segmentation_maps(
        image_paths=[str(p) for p in found],
        output_dir=output_dir,            # not used for saving (out_fn overrides)
        size=size,
        batch_size=batch_size,
        model_name=model_name,
        device=device,
        skip_existing=skip_existing,
        local_files_only=local_files_only,
        out_path_fn=out_fn,
        resize_mode=resize_mode,
    )

    # ---- Step 3: index computed seg maps by NORMALISED absolute image path - #
    # THE HEART OF THE IMAGE<->MAP MAPPING (mirrors depth exactly).
    # path_to_seg came back from the compute step as {source image path ->
    # its saved seg PNG}, one pair per image — the pairing is guaranteed by
    # construction (each map was saved while processing exactly that image,
    # into <folder>_seg_map.png). Re-keyed by NORMALISED ABSOLUTE path
    # because one file can be spelled many ways ("D:/x.jpg", "d:\\x.jpg",
    # relative, different case on Windows): Path.resolve() collapses them to
    # one canonical absolute form, os.path.normcase() removes case/slash
    # differences. Without this, a JSONL spelling the path differently from
    # the scanner would silently match NOTHING.
    seg_by_img = {
        os.path.normcase(str(Path(img).resolve())): seg
        for img, seg in path_to_seg.items()
    }

    # ---- Step 4: rebuild each split manifest, matching by absolute path ---- #
    # The ORIGINAL data/{train,val,test}.jsonl are the single source of truth
    # for the two things a folder scan cannot know: each image's PROMPT and
    # its SPLIT. We walk each original split file, look the image up in
    # seg_by_img, and write a new line carrying image+map+prompt together.
    # Split boundaries are preserved EXACTLY — never re-shuffled (that would
    # leak train images into val and make val/loss a lie).
    for split in ("train", "val", "test"):
        src = _find_split_jsonl(data_dir, split)
        if src is None:
            print(f"\n[WARN] No JSONL for split '{split}' in {data_dir} — skipping.")
            continue

        with open(src, "r", encoding="utf-8") as f:
            entries = [json.loads(line) for line in f if line.strip()]

        out_entries = []
        n_skipped = 0
        for entry in entries:
            # Key holding the image path in the ORIGINAL jsonl is configurable
            # (--image_path, default "target" for this dataset).
            raw_str = _get_image_path(entry, image_path)
            # Same normalisation as Step 3 — both sides of the lookup MUST be
            # normalised identically or matching fails for spelling reasons.
            key = os.path.normcase(str(Path(raw_str).resolve()))
            seg = seg_by_img.get(key)
            if seg is None:
                # Not in the scanned/computed set (full run: file missing on
                # disk; dry run: outside the capped subset). Skip rather than
                # guess — a skipped line shows up in the count below; a wrong
                # pairing would be invisible and poison training.
                n_skipped += 1
                continue
            # ground_truth: same carry-forward discipline as the --data_dir
            # mode function above -- "" if the input row has none.
            _gt = entry.get("ground_truth", "")
            if _gt and not Path(_gt).is_absolute():
                _gt = (Path.cwd() / _gt).resolve().as_posix()
            # ABSOLUTE paths, as_posix (forward slashes work on Windows AND
            # Linux); dataset lives outside the repo so relative paths would
            # break the moment the repo moves.
            out_entries.append({
                "raw_image_path": Path(raw_str).resolve().as_posix(),
                "seg_path":       Path(seg).resolve().as_posix(),
                "prompt":         entry["prompt"],   # copied VERBATIM — never edited
                "ground_truth":   _gt,
            })

        out_path = output_dir / f"{split}.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for e in out_entries:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")

        if subset_n:
            note = f"  (dry-run: {n_skipped} entries not in the scanned subset — expected)"
        else:
            note = f"  ({n_skipped} entries had no seg on disk — skipped)" if n_skipped else ""
        print(f"\n  Written {len(out_entries)} entries -> {out_path}{note}")
        _verify_scan_seg_training_jsonl(out_path, "seg_path", _suffix)


def run_seg_directory_mode(args):
    """
    DIRECTORY MODE: scan --input_dir recursively for images, segment them, save
    class-ID PNGs preserving the relative folder structure. Use when you just want
    the label maps without (re)building the training JSONs.

    The "seg" prefix marks this as segmentation-pipeline code.
    """
    input_dir = Path(args.input_dir).resolve()
    if not input_dir.exists():
        print(f"[ERROR] Input directory not found: {input_dir}")
        return

    # Mode-named default: the map's squaring is baked in at calc time, so
    # runs using different resize_mode values must not collide. An explicit
    # --output_dir is trusted as-is.
    output_dir = (
        Path(args.output_dir).resolve() if args.output_dir
        else input_dir.parent / f"raw_seg_{args.resize_mode}"
    )

    valid_exts  = {".jpg", ".jpeg", ".png", ".webp"}
    image_paths = sorted(
        p for p in input_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in valid_exts
    )
    if not image_paths:
        print(f"[ERROR] No images found under: {input_dir}")
        return

    print(f"Found {len(image_paths)} images under: {input_dir}")
    precompute_segmentation_maps(
        image_paths=[str(p) for p in image_paths],
        output_dir=output_dir,
        size=args.size,
        batch_size=args.batch_size,
        model_name=args.model,
        device=args.device,
        skip_existing=not args.no_skip,
        input_dir=input_dir,
        local_files_only=args.local_files_only,
        resize_mode=args.resize_mode,
    )
    print("\nNext step — build training JSONLs with --data_dir, then segformer_training.py")


# ============================================================================ #
#  SINGLE-IMAGE MODE  (--image)                                                #
#  One image in -> one seg map saved BESIDE it: <dir>/<stem>_seg_map.png       #
#  This is the "I have one new CARLA/real-world photo, give me its map so I    #
#  can run segformer_inference.py on it" workflow.                             #
# ============================================================================ #

def run_single_image_mode(args) -> None:
    """
    Compute the seg map for exactly ONE image and save it NEXT TO the image
    (same folder), named `<image_stem>_seg_map_<resize_mode>.png`.

    Why beside the image (not an output tree): a single ad-hoc image has no
    dataset structure to mirror; keeping the map next to its source makes the
    (image, map) pair self-documenting and directly usable as
    segformer_inference.py's  "inference.seg_maps=[...]" + "inference.images=[...]".
    The `_seg_map` suffix matches the dataset-scan naming convention, so a
    file is recognisable as pipeline output wherever it lives. The resize_mode
    is baked into the FILENAME here (not a folder, since there is no separate
    output folder in this mode), so computing multiple modes for the same
    image never overwrites another mode's result.
    """
    image_path = Path(args.image).resolve()
    if not image_path.exists():
        raise FileNotFoundError(f"--image not found: {image_path}")

    out_path = image_path.parent / (image_path.stem + f"_seg_map_{args.resize_mode}.png")
    print(f"\n[single-image mode]")
    print(f"  image       : {image_path}")
    print(f"  resize_mode : {args.resize_mode}")
    print(f"  seg map     : {out_path}")

    # out_path_fn overrides ALL path derivation in the core routine — the
    # single file goes exactly beside its source, nowhere else.
    results = precompute_segmentation_maps(
        image_paths=[str(image_path)],
        output_dir=image_path.parent,          # unused (out_path_fn wins), required arg
        size=args.size,
        batch_size=1,
        model_name=args.model,
        device=args.device,
        skip_existing=not args.no_skip,
        local_files_only=args.local_files_only,
        out_path_fn=lambda src: out_path,
        resize_mode=args.resize_mode,
    )
    if str(image_path) in results:
        print(f"\nDone. Next: run inference with this map, e.g.:")
        print(f'  python segformer_inference.py ckpt_path=<best_model> resize_mode={args.resize_mode} '
              f'"inference.seg_maps=[{out_path.as_posix()}]" '
              f'"inference.images=[{image_path.as_posix()}]" '
              f'"inference.prompts=[\'your prompt\']"')


# ============================================================================ #
#  SINGLE-JSONL MODE  (--json_file)                                            #
#  One manifest in -> maps into a SIBLING <images_root>_seg_map/ folder        #
#  (mirrored structure, like dataset-scan mode) -> updated manifest written    #
#  beside the input with the STANDARD keys:                                    #
#      raw_image_path / seg_path / prompt                                      #
# ============================================================================ #

def run_json_file_mode(args) -> None:
    """
    Compute seg maps for every entry of ONE JSONL manifest.

    INPUT : a .jsonl where each line has at least the image-path key
            (--image_path, default 'raw_image_path'; 'prompt' is carried
            through if present, else saved as "").
    MAPS  : saved into a SIBLING folder of the images' common root —
            <root_parent>/<root_name>_seg_map_<resize_mode>/ — mirroring each
            image's relative folder structure, named <image_stem>_seg_map.png.
            The source image tree is never written into (same contract as
            dataset-scan mode; see _sibling_map_path's docstring). Mode-named
            so runs using different resize_mode values never collide — the
            map's squaring is baked in at calc time.
    OUTPUT: <input_stem>_seg.jsonl beside the input manifest, each line:
              {"raw_image_path": ..., "seg_path": ..., "prompt": ...}
            — the project-standard keys that segformer_training.py's dataset
            and segformer_inference.py's json mode both read directly. Paths
            are written ABSOLUTE with forward slashes (match scan mode).
    """
    json_path = Path(args.json_file).resolve()
    if not json_path.exists():
        raise FileNotFoundError(f"--json_file not found: {json_path}")

    image_root = Path(args.image_root).resolve() if args.image_root else None

    # ---- read entries + resolve every image path --------------------------- #
    # utf-8-sig: identical to utf-8 for normal files, but ALSO transparently
    # strips the BOM that Windows editors/PowerShell prepend — a BOM'd first
    # line otherwise crashes json.loads (found by execution 2026-07-20).
    entries, abs_paths = [], []
    with open(json_path, "r", encoding="utf-8-sig") as f:
        for ln, line in enumerate(f, 1):
            if not line.strip():
                continue
            entry = json.loads(line)
            p = Path(_get_image_path(entry, args.image_path))
            if not p.is_absolute():
                p = (image_root / p) if image_root else (Path.cwd() / p)
            p = p.resolve()
            if not p.exists():
                raise FileNotFoundError(f"line {ln}: image not found: {p}")
            entries.append(entry)
            abs_paths.append(p)

    if args.dry_run_n:
        entries, abs_paths = entries[: args.dry_run_n], abs_paths[: args.dry_run_n]
        print(f"\n[DRY RUN] First {len(entries)} entries only.")
    if not entries:
        print("[ERROR] No entries in the JSONL — nothing to do.")
        return

    # ---- sibling output root, mirroring from the images' common root ------- #
    # Common root = deepest folder shared by ALL images (no hardcoding),
    # exactly how --data_dir mode finds its dataset root.
    images_root  = Path(os.path.commonpath([str(p) for p in abs_paths]))
    sibling_root = images_root.parent / (images_root.name + f"_seg_map_{args.resize_mode}")
    print(f"\n[json_file mode]")
    print(f"  manifest    : {json_path}")
    print(f"  resize_mode : {args.resize_mode}")
    print(f"  images root : {images_root}")
    print(f"  maps root   : {sibling_root}   (sibling, mirrored structure)")

    def _out_path(src: str) -> Path:
        src = Path(src)
        rel = src.parent.relative_to(images_root)
        return sibling_root / rel / (src.stem + "_seg_map.png")

    # ---- collision guard is inside the core routine; compute all maps ------ #
    results = precompute_segmentation_maps(
        image_paths=[str(p) for p in abs_paths],
        output_dir=sibling_root,               # informational; out_path_fn wins
        size=args.size,
        batch_size=args.batch_size,
        model_name=args.model,
        device=args.device,
        skip_existing=not args.no_skip,
        local_files_only=args.local_files_only,
        out_path_fn=_out_path,
        resize_mode=args.resize_mode,
    )

    # ---- write the updated manifest with the STANDARD keys ----------------- #
    out_jsonl = json_path.parent / (json_path.stem + "_seg.jsonl")
    written = skipped = 0
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for entry, p in zip(entries, abs_paths):
            seg = results.get(str(p))
            if seg is None:                    # image failed -> excluded, loudly
                skipped += 1
                print(f"[WARN] no map for {p} — entry excluded from {out_jsonl.name}")
                continue
            f.write(json.dumps({
                "raw_image_path": p.as_posix(),
                "seg_path":       Path(seg).as_posix(),
                "prompt":         entry.get("prompt", ""),
            }) + "\n")
            written += 1
    print(f"\n  Written {written} entries -> {out_jsonl}" +
          (f"  ({skipped} entries FAILED and were excluded)" if skipped else ""))

    # Same last-line-of-defence verification the scan mode runs: every line's
    # files exist, keys present, pairing sane — fails loudly before training
    # or inference ever trusts this manifest. name_from="stem": json-mode maps
    # are named after the image STEM (see check 4b's docstring for why).
    _verify_scan_seg_training_jsonl(out_jsonl, "seg_path", "_seg_map",
                                    name_from="stem")


def main():
    parser = argparse.ArgumentParser(
        description="Pre-compute SegFormer-b5-Cityscapes segmentation label maps.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--data_dir", type=str, default=None,
        help="Folder with train.jsonl/val.jsonl/test.jsonl (e.g. data/). "
             "Any *.jsonl whose stem contains 'train'/'val'/'test' is also accepted. "
             "Activates seg-training JSONL mode -> data/seg_training/*.jsonl.",
    )
    parser.add_argument(
        "--raw_dir", type=str, default=None,
        help="(--data_dir mode) Pin the dataset root used to derive the sibling "
             "seg-map folder's location and mirrored structure -- e.g. "
             "--raw_dir data/dataset/extracted saves maps to "
             "data/dataset/extracted_seg_map_<mode>/, mirroring extracted/'s full "
             "structure underneath. Only used if EVERY referenced image actually "
             "lives under it; otherwise (or if omitted) the root is auto-derived "
             "as the deepest folder shared by all referenced images -- which can "
             "give a different, less predictable sibling location depending on "
             "which images the manifest happens to reference (e.g. a split file "
             "that only touches one city folder). Default: <data_dir>/raw.",
    )
    parser.add_argument(
        "--seg_dir", type=str, default=None,
        help="Legacy, --data_dir mode only: NOT used for saving (kept for CLI "
             "backward-compatibility). Use --output_root to redirect where the "
             "seg-ID PNGs actually go.",
    )
    parser.add_argument(
        "--output_root", type=str, default=None,
        help="--data_dir mode only: where to save the seg-ID PNGs themselves "
             "(mirrors grounded_sam_map_calculations.py's --output_root). "
             "Default (if omitted): a SIBLING of --data_dir, suffixed "
             "'_seg_map_<resize_mode>' -- e.g. --data_dir data/custome_dataset "
             "-> data/custome_dataset_seg_map_aspect. Pass this to save maps "
             "somewhere else entirely (a different disk/mount, a shared "
             "location, etc.) without changing --data_dir.",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="--input_dir mode ONLY: where to save the scanned PNGs "
             "(default: <input_dir's parent>/raw_seg_<resize_mode>). "
             "--data_dir/--dataset_dir modes use --output_root/--manifest_out instead.",
    )
    parser.add_argument(
        "--manifest_out", type=str, default=None,
        help="--data_dir/--dataset_dir modes: where to write train/val/test.jsonl "
             "(mirrors grounded_sam_map_calculations.py's --manifest_out). "
             "--data_dir mode default (if omitted): the SAME folder the seg-ID "
             "PNGs landed in (--output_root, or its own sibling-of-data_dir "
             "default) -- matching grounded_sam exactly. --dataset_dir scan mode "
             "default: <data_dir>/seg_training_<resize_mode> (unchanged, no "
             "grounded_sam equivalent to match for that mode).",
    )
    parser.add_argument(
        "--input_dir", type=str, default=None,
        help="Directory-mode: scan this folder for images instead of using JSONs.",
    )
    parser.add_argument(
        "--dry_run_n", type=int, default=None,
        help="Process only the first N entries per JSONL (sanity run before full dataset).",
    )
    parser.add_argument(
        "--size", type=int, default=None,
        help="Square size for seg maps, used as a fallback when --width/--height "
             "are not both given. If omitted (and --width/--height also omitted), "
             "defaults to this project's real dataset resolution, 1280x800 "
             "(non-square) -- NOT a generic square guess. Pass --size explicitly "
             "to opt into a square target instead.",
    )
    parser.add_argument(
        "--width", type=int, default=None,
        help="Non-square target width, e.g. 1280. REQUIRES --height and "
             "--resize_mode aspect. Overrides --size. Both must be divisible "
             "by 32 for SDXL (VAE /8, 2 UNet halvings /4 -- see the field "
             "guide's Lesson 4 for the /64 SD1.5 version of this same math) "
             "and chosen close to the source aspect ratio to keep distortion "
             "negligible -- e.g. 1280x800 matches this project's real "
             "dataset's native ratio (1280:800=1.6) exactly, no resize loss.",
    )
    parser.add_argument(
        "--height", type=int, default=None,
        help="Non-square target height, e.g. 320. REQUIRES --width and "
             "--resize_mode aspect. Overrides --size.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=None,
        help="Images per GPU batch. Default: auto-scaled from the detected GPU's "
             "VRAM (baseline 4 on a 12GB GPU; larger GPUs get a proportionally "
             "larger batch automatically). Pass a number to disable auto-scaling.",
    )
    parser.add_argument(
        "--model", type=str,
        default="checkpoints/local_models/segformer-b5-cityscapes",
        help=(
            "Seg model: local folder path (default, offline) OR a HF id like "
            f"{DEFAULT_SEG_MODEL} (use with --local_files_only False). "
            "Default: checkpoints/local_models/segformer-b5-cityscapes"
        ),
    )
    parser.add_argument(
        "--local_files_only", type=parse_bool, default=True,
        metavar="True|False",
        help=(
            "True (default) = load --model strictly from local disk (offline). "
            "False = allow download into checkpoints/local_models/. "
            "On first use, run with --local_files_only False to download the b5 model."
        ),
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device: 'cuda', 'cuda:N', or 'cpu'. Default: auto-detect a GPU and "
             "REFUSE to run if none is visible (never silently falls back to CPU). "
             "Pass --device cpu explicitly if you really want CPU.",
    )
    parser.add_argument(
        "--no_skip", action="store_true",
        help="Re-compute even if a seg PNG already exists.",
    )
    parser.add_argument(
        "--resize_mode", type=str, default="aspect",
        choices=["aspect"],
        help=(
            "Geometry technique applied to the RGB BEFORE SegFormer sees it. "
            "'aspect' (only mode supported) = NO pad, NO crop -- direct resize "
            "to an explicit non-square --width/--height chosen close to the "
            "source aspect ratio (e.g. 512x320 for a 1280x800 source). THIS "
            "GETS BAKED INTO THE SAVED MAP (unlike the grounded_sam branch) "
            "-- training/inference must be configured with the SAME "
            "width/height used here."
        ),
    )
    parser.add_argument(
        "--image_path", type=str, default="raw_image_path",
        help="Key in your input JSONLs that holds the image path. "
             "Default: 'raw_image_path' (the standard key used by "
             "raw_image_path / seg_path / prompt manifests, training and "
             "inference, both pipelines). Change to 'target'/'source' for "
             "manifests using those keys instead. "
             "Example: --image_path target",
    )
    parser.add_argument(
        "--image_root", type=str, default=None,
        help="Root folder prepended to RELATIVE image paths from the JSONL. "
             "Use this when images are NOT under the repo root. "
             "Example: --image_root /mnt/dataset  (then JSONL path "
             "'images/foo.jpg' resolves to /mnt/dataset/images/foo.jpg). "
             "Absolute paths in the JSONL are always used as-is.",
    )

    # ---- Single-image mode (map saved BESIDE the input image) -------------- #
    parser.add_argument(
        "--image", type=str, default=None,
        help="SINGLE-IMAGE mode: compute the seg map for exactly this one image "
             "and save it NEXT TO it as <stem>_seg_map_<resize_mode>.png (same "
             "folder). The printed 'Next:' line shows the ready-to-paste "
             "segformer_inference.py command for the pair. "
             "Example: --image data/new_frame.jpg",
    )

    # ---- Single-JSONL mode (sibling _seg_map folder + standard-key manifest) #
    parser.add_argument(
        "--json_file", type=str, default=None,
        help="SINGLE-JSONL mode: compute a seg map for every entry of this one "
             "manifest. Maps go to a SIBLING folder of the images' common root "
             "(<root>_seg_map_<resize_mode>/, mirrored structure, <stem>_seg_map.png) "
             "— the image tree is never written into. Writes <stem>_seg.jsonl beside "
             "the input with the standard keys raw_image_path/seg_path/prompt, "
             "then self-verifies it. Example: --json_file data/my_frames.jsonl",
    )

    # ---- Dataset-SCAN mode (find raw_image.jpg by scanning a folder) ------- #
    parser.add_argument(
        "--dataset_dir", type=str, default=None,
        help="Activates DATASET-SCAN mode. Recursively scans this folder for files "
             "named --image_name (default raw_image.jpg), computes a seg map for each, "
             "and saves it into a SIBLING folder next to --dataset_dir (e.g. "
             "<dataset_dir>_seg_map_<resize_mode>/), mirroring the internal folder "
             "structure, named <folder>_seg_map_<resize_mode>.png. The source dataset "
             "folder is never written into. Requires --data_dir (the folder holding "
             "train/val/test.jsonl) to recover each image's prompt + split. "
             "Example: --dataset_dir /data/custome_dataset --data_dir data/ --image_path target",
    )
    parser.add_argument(
        "--image_name", type=str, default="raw_image.jpg",
        help="(scan mode) Exact filename to process in each folder. Every other file "
             "is ignored — including *_depth_map.png / *_seg_map.png this pipeline writes. "
             "Default: raw_image.jpg",
    )
    args = parser.parse_args()

    _setup_logging(PROJECT_ROOT / "outputs" / "logs")

    if not any([args.dataset_dir, args.data_dir, args.input_dir,
                args.image, args.json_file]):
        parser.error(
            "Provide one of:\n"
            "  --image path/to/frame.jpg   (single image — map saved beside it as <stem>_seg_map_<mode>.png)\n"
            "  --json_file path/to/list.jsonl   (one manifest — maps to sibling _seg_map_<mode> folder + <stem>_seg.jsonl)\n"
            "  --dataset_dir /data/custome_dataset --data_dir data/  (scan mode — saves maps to a sibling folder)\n"
            "  --data_dir data/   (builds data/seg_training_<mode>/*.json from JSONL paths)\n"
            "  --input_dir data/raw   (directory mode — PNGs only, no JSON)\n"
            "All modes accept --resize_mode aspect (default, only mode supported)."
        )
    _n_modes = sum(bool(m) for m in
                   [args.dataset_dir, args.data_dir, args.input_dir,
                    args.image, args.json_file])
    if _n_modes > 1 and not (args.dataset_dir and args.data_dir):
        # (--dataset_dir legitimately REQUIRES --data_dir; every other pairing
        # is ambiguous — refuse instead of guessing which mode was meant.)
        parser.error("Pass only ONE mode flag (--image / --json_file / "
                     "--dataset_dir / --data_dir / --input_dir).")
    if args.data_dir and args.input_dir:
        parser.error("--data_dir and --input_dir are mutually exclusive.")
    if args.dataset_dir and not args.data_dir:
        parser.error(
            "--dataset_dir (scan mode) also needs --data_dir pointing at the folder that "
            "holds train/val/test.jsonl — that is where each image's prompt + split come from.\n"
            "  Example: --dataset_dir /data/custome_dataset --data_dir data/ --image_path target"
        )

    # ---- Resolve --size vs --width/--height into one `seg_size` value ------ #
    # int -> square target. (width, height) tuple -> non-square target.
    if (args.width is None) != (args.height is None):
        parser.error("--width and --height must be given together, or not at all.")
    if args.width is not None:
        if args.width % 32 or args.height % 32:
            parser.error(f"--width {args.width} and --height {args.height} must both be "
                          f"divisible by 32 (SDXL's VAE÷8 x 2 UNet halvings÷4 -- see field guide Lesson 4 "
                          f"for the SD1.5 /64 version of this same math).")
        seg_size = (args.width, args.height)
    elif args.size is not None:
        seg_size = args.size
    else:
        # Nothing passed at all -- default to THIS PROJECT's real resolution
        # (1280x800, matches the real dataset's native aspect ratio exactly,
        # divisible by 32 for SDXL) instead of a generic square guess. Found
        # by a real user hitting this: running with no --width/--height
        # silently produced 512x512 square maps -- distorted relative to the
        # ~1.6:1 real dataset, and not what anyone actually wants by default
        # on this project.
        seg_size = (1280, 800)
    # Overwrite in place: every downstream function below reads `args.size`
    # directly (not a separate parameter), so this one assignment propagates
    # the resolved int-or-(width,height) value everywhere without threading
    # a new argument through run_seg_directory_mode/run_single_image_mode/
    # run_json_file_mode/build_seg_training_from_scan/build_segmentation_training_jsons.
    args.size = seg_size

    # Belt-and-suspenders offline lock: set env vars so NOTHING touches the network,
    # on top of local_files_only=True being passed to from_pretrained.
    if args.local_files_only:
        os.environ["HF_HUB_OFFLINE"]      = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        # --model's default ("checkpoints/local_models/segformer-b5-cityscapes")
        # is relative -- only resolves correctly when launched from the repo
        # root. Anchor to PROJECT_ROOT so it works from any launch directory,
        # same fix already applied to grounded_sam_map_calculations.py.
        # PROJECT_ROOT / args.model is a no-op if --model was already passed
        # as an absolute path (pathlib silently discards the left operand
        # when the right one is absolute) -- safe either way. Only done in
        # this branch: --model is a HF Hub id, not a path, when
        # local_files_only=False, and must not be touched.
        args.model = str(PROJECT_ROOT / args.model)

    # Resolve device LOUDLY: prints full GPU diagnostics and raises a clear
    # error (instead of silently running on CPU) unless --device cpu was
    # explicitly passed. See src/utils.py resolve_device() for why this exists.
    args.device = resolve_device(args.device)

    # Auto-scale batch_size to the detected GPU's VRAM UNLESS the user passed
    # --batch_size explicitly (sentinel default is None). See src/utils.py
    # auto_batch_size() — verified on a ~12GB GPU, reasoned (not executed) for
    # much larger GPUs; override with --batch_size if it ever OOMs.
    if args.batch_size is None:
        args.batch_size = auto_batch_size(default=4, device=args.device)

    print(f"Device           : {args.device}")
    _size_str = f"{args.size}x{args.size}" if isinstance(args.size, int) else f"{args.size[0]}x{args.size[1]} (non-square, resize_mode=aspect)"
    print(f"Size             : {_size_str}")
    print(f"resize_mode      : {args.resize_mode}  (baked into the saved map)")
    print(f"Batch            : {args.batch_size}")
    print(f"Model            : {args.model}")
    print(f"local_files_only : {args.local_files_only}")

    if args.image:
        # SINGLE-IMAGE mode: one map, saved beside the image.
        run_single_image_mode(args)

    elif args.json_file:
        # SINGLE-JSONL mode: sibling _seg_map folder + standard-key manifest.
        run_json_file_mode(args)

    elif args.dataset_dir:
        # DATASET-SCAN mode: find raw_image.jpg by scanning, save maps to a
        # SIBLING folder (mirrored structure), rebuild
        # seg_training/{train,val,test}.jsonl from the original splits.
        dataset_dir = Path(args.dataset_dir).resolve()
        data_dir    = Path(args.data_dir).resolve()
        # Mode-named default: without this, running calc twice with different
        # resize_mode values would silently OVERWRITE the first run's
        # train/val/test.jsonl with the second's, even though the PNG maps
        # themselves are correctly mode-separated (sibling folder). An
        # explicit --manifest_out is trusted as-is. No grounded_sam
        # equivalent for THIS mode (scan mode has none), so this default
        # stays its own established formula rather than switching to
        # "same folder as the PNGs" the way --data_dir mode below does.
        out_dir = (Path(args.manifest_out).resolve() if args.manifest_out
                   else data_dir / f"seg_training_{args.resize_mode}")

        if not dataset_dir.exists():
            parser.error(f"--dataset_dir not found: {dataset_dir}")
        if args.dry_run_n:
            print(f"\n[DRY RUN] Scan capped at first {args.dry_run_n} images.")

        build_seg_training_from_scan(
            dataset_dir=dataset_dir,
            data_dir=data_dir,
            output_dir=out_dir,
            image_name=args.image_name,
            image_path=args.image_path,
            size=args.size,
            batch_size=args.batch_size,
            model_name=args.model,
            device=args.device,
            skip_existing=not args.no_skip,
            local_files_only=args.local_files_only,
            subset_n=args.dry_run_n,
            resize_mode=args.resize_mode,
        )

    elif args.data_dir:
        data_dir  = Path(args.data_dir).resolve()
        raw_dir   = Path(args.raw_dir).resolve()   if args.raw_dir  else data_dir / "raw"
        seg_dir   = Path(args.seg_dir).resolve()   if args.seg_dir  else data_dir / "raw_seg"
        if args.dry_run_n:
            print(f"\n[DRY RUN] First {args.dry_run_n} entries per JSON.")
        image_root = Path(args.image_root).resolve() if args.image_root else None
        if image_root:
            print(f"Image root : {image_root}  (prepended to relative image paths)")
        # output_root/manifest_out resolved to actual paths INSIDE
        # build_segmentation_training_jsons (None passed through as-is means
        # "use the default") -- unlike scan mode above, this mode's default
        # sibling location can't be computed here in main() without first
        # scanning the real images the manifest references (see that
        # function's own docstring), so there's nothing to pre-resolve.
        build_segmentation_training_jsons(
            data_dir=data_dir, raw_dir=raw_dir if args.raw_dir else (image_root or data_dir / "raw"),
            seg_dir=seg_dir,
            manifest_out=Path(args.manifest_out).resolve() if args.manifest_out else None,
            size=args.size, batch_size=args.batch_size, model_name=args.model,
            device=args.device,
            skip_existing=not args.no_skip, subset_n=args.dry_run_n,
            local_files_only=args.local_files_only,
            image_path=args.image_path,
            image_root=image_root,
            resize_mode=args.resize_mode,
            output_root=Path(args.output_root).resolve() if args.output_root else None,
        )
    else:
        run_seg_directory_mode(args)


if __name__ == "__main__":
    main()