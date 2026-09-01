"""
grounded_sam_map_calculations.py
----------------
Precompute Grounded-SAM segmentation maps (Grounding DINO box detection +
SAM mask segmentation, composited into a dense per-pixel class map -- see
src/encoders/grounded_sam_encoder.py) for a Cityscapes-layout dataset, and
write train/val/test JSONL manifests.

This file mirrors the NAMING and STRUCTURE conventions of the segformer
branch's own seg_map_calculations.py (one shared core per-image routine that
every entry-point mode delegates into via a path-lookup dict, a --data_dir
folder that auto-discovers train.jsonl/val.jsonl/test.jsonl inside it,
consistent verify-function naming, the same section style) -- but the LOGIC
inside is Grounded-SAM's own. Two things are deliberately NOT carried over
from segformer's version, because they're specific to SegFormer's
architecture, not a generic convention:

  - BATCHING. SegFormer is one dense classifier forward pass, so batching
    multiple images into one tensor is a real speedup there. Grounded-SAM's
    encoder runs Grounding DINO (box detection) + SAM (per-box mask decode)
    per image internally (see GroundedSamEncoder.label_ids/_label_ids_single
    -- box count varies per image, nothing here batches across images), so
    the core routine below is deliberately per-image, no batch_size.
  - resize_mode BAKED INTO OUTPUT NAMING. SegFormer's RGB preprocessing
    determines the map's final resolution, so a --resize_mode/--width/
    --height choice has to be baked into the output folder/filename to keep
    different-mode runs from colliding. GroundedSamEncoder resizes
    internally via its own `size` param -- there's no equivalent to bake in,
    so output naming here stays stem-based, exactly as it always was.

  - Each output map is a raw class-ID PNG (mode "L", 0..27 = this branch's own
    28-class CARLA vocabulary, IGNORE_ID=255 for unmatched pixels), NOT
    colour-coded -- colourisation only happens at load time via
    grounded_sam_encoder.py's own CARLA_PALETTE/carla_palette_tensor()
    (local_seg.py imports it as such), NOT segformer's 19-class
    SEG_CITYSCAPES_PALETTE -- the two branches' maps share the same raw-ID-PNG
    file convention (mode "L", IGNORE_ID=255), never the same class scheme.
  - Naming is stem-based (<stem>_seg_map_grounded_sam.png), mirroring the
    relative folder structure from a dataset-root anchor into a sibling
    output tree -- collision-safe for both a folder-per-sample layout and a
    many-files-per-folder layout (e.g. Cityscapes' native
    leftImg8bit/{city}/*.png, many differently-named files sharing one city
    folder).
  - Skip-existing by default: safe to re-run, only computes what's missing.
  - A bad/unreadable image never aborts the whole run: load and encode are
    each wrapped separately, so one bad file is logged and skipped instead
    of crashing a multi-hour batch partway through.
  - Every output manifest is self-verified after writing (keys present, seg
    PNG on disk, no two rows sharing a map file) -- same last-line-of-defence
    discipline as seg_map_calculations.py's verify functions.

Every manifest row also carries a "ground_truth" field (path to a GT PNG,
or "" if none is available for that sample). Resolution order, cheapest/
most-explicit first: (1) the input row's own "ground_truth" key, carried
through unmodified if present; (2) the input row's "seg_path" key -- hand-
built verification manifests (e.g. data/dataset/*.jsonl, built by pairing
real images against real Cityscapes labels, see outputs/verify_grounded_sam/)
store real ground truth there, not under a dedicated "ground_truth" key;
(3) "" if the input row has neither. Never guessed from a naming
convention -- a guessed path only holds for one specific local layout, and
would silently break on a different machine where the real dataset's files
live somewhere else entirely. Training itself never reads this field
(src/data/local_seg.py only reads raw_image_path/seg_path/prompt) -- it's
diagnostic-only, and always safe to be "".

INPUT: an EXISTING jsonl of {<image_path_key or "raw_image_path">, "prompt",
["ground_truth"]} entries per split, found one of two ways (mix freely):

  - --data_dir <folder>: auto-discovers <folder>/train.jsonl,
    <folder>/val.jsonl, <folder>/test.jsonl (exact name first, then any
    *.jsonl whose stem contains the split name -- see _find_split_jsonl).
    A split with no matching file is skipped with a warning, not an error.
  - --train_jsonl / --val_jsonl / --test_jsonl <path>: pin a specific split's
    manifest explicitly -- overrides --data_dir's auto-discovery for that
    split, or works standalone with no --data_dir at all.

Same --image_path key-fallback logic as seg_map_calculations.py's own
--data_dir mode (checks --image_path first, e.g. "target", then falls back
to "raw_image_path"), so both scripts can read the identical real-dataset
jsonl without any conversion step.

Output: a sibling of --data_dir (or --output_root if --data_dir wasn't
given), suffixed "_seg_map_grounded_sam" -- e.g. --data_dir .../custom_dataset
-> .../custom_dataset_seg_map_grounded_sam -- mirroring the relative folder
structure of the source images underneath, one raw class-ID PNG per image,
named <stem>_seg_map_grounded_sam.png. New train/val/test.jsonl are written
there too, each row carrying the SAME raw_image_path/prompt/ground_truth as
the input row, with seg_path updated to point at the newly computed map.

MULTI-GPU (SLURM/sbatch-driven, not managed by this script):
  This script is single-process/single-GPU per invocation -- it does NOT
  spawn its own worker processes. On a cluster, launch it once PER GPU (your
  sbatch script owns that, e.g. `srun --ntasks-per-node=4 ...`), and pass
  each invocation its own --rank/--world_size so every instance works a
  disjoint slice of the same image list and they all write into the same
  shared output tree (safe: every image's output path is unique by
  construction).

    # example sbatch snippet, one task per GPU:
    srun --ntasks-per-node=4 python grounded_sam_map_calculations.py \\
        --data_dir data/custom_dataset --width 1280 --height 800 \\
        --rank $SLURM_PROCID --world_size $SLURM_NTASKS

  --rank/--world_size default to 0/1 (i.e. process the whole list yourself)
  when omitted, so nothing changes for a normal single-GPU run.

LOGGING (mandatory):
  Every run writes a timestamped log file under outputs/logs/ (console
  output is a subset of what's logged, not a replacement for it). When
  --world_size > 1, the log file is rank-suffixed so parallel sbatch tasks
  never clobber each other's log.

Usage:
    # Folder auto-discovery -- looks for train.jsonl/val.jsonl/test.jsonl
    # inside data/custom_dataset/, writes data/custom_dataset_seg_map_grounded_sam/:
    python grounded_sam_map_calculations.py \\
        --data_dir data/custom_dataset --width 1280 --height 800

    # Explicit per-split paths, no --data_dir -- --output_root then has
    # nothing to default from and must be given explicitly:
    python grounded_sam_map_calculations.py \\
        --output_root data/custom_dataset_seg_map_grounded_sam \\
        --width 1280 --height 800 \\
        --train_jsonl data/train.jsonl --val_jsonl data/val.jsonl --image_path target

    # Pin one split explicitly while auto-discovering the rest from --data_dir:
    python grounded_sam_map_calculations.py \\
        --data_dir data/custom_dataset --test_jsonl data/custom_dataset/holdout.jsonl
"""

import argparse
import json
import logging
import os
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from src.encoders.grounded_sam_encoder import GroundedSamEncoder

PROJECT_ROOT = Path(os.path.abspath(__file__)).parent
SCRIPT_NAME = "grounded_sam_map_calc"

logger = logging.getLogger(SCRIPT_NAME)


def _setup_logging(log_dir: Path, rank: int | None) -> Path:
    """Mandatory console+file logging -- every run writes a timestamped log
    under outputs/logs/, rank-suffixed when a sbatch script runs several
    instances of this script in parallel (see module docstring)."""
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    suffix = f"_rank{rank}" if rank is not None else ""
    log_path = log_dir / f"{SCRIPT_NAME}_{ts}{suffix}.log"

    prefix = f"[rank {rank}] " if rank is not None else ""
    fmt = logging.Formatter(
        f"%(asctime)s | %(levelname)-7s | {prefix}%(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)

    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.info(f"Log file: {log_path}")
    return log_path


def _shard(items: list, rank: int, world_size: int) -> list:
    """Round-robin split: this invocation gets items[rank::world_size].
    Round-robin, not contiguous chunks, so any ordering-related skew in the
    source list spreads evenly across sbatch tasks instead of piling onto
    one."""
    return items[rank::world_size] if world_size > 1 else items


def _get_image_path(entry: dict, image_path_key: str = "raw_image_path") -> str:
    """Mirrors seg_map_calculations.py's own _get_image_path EXACTLY (same
    key, same fallback) -- lets both scripts read the identical real-dataset
    jsonl (e.g. data/train.jsonl, which uses "target") without any
    conversion step."""
    if image_path_key in entry:
        return entry[image_path_key]
    if "raw_image_path" in entry:
        return entry["raw_image_path"]
    raise KeyError(f"Entry has neither '{image_path_key}' nor 'raw_image_path'. Keys: {list(entry.keys())}")


def _is_relative_to(p: Path, other: Path) -> bool:
    try:
        p.relative_to(other)
        return True
    except ValueError:
        return False


def _resolve_abs(path_str: str) -> str:
    """"" stays "" (no ground_truth for this row); otherwise resolve relative
    to cwd, same treatment raw_image_path already gets above -- every path
    written into a manifest by this script must be absolute, or it breaks
    the moment something changes the process cwd (e.g. grounded_sam_training.py's
    hydra.job.chdir=true)."""
    if not path_str:
        return ""
    p = Path(path_str)
    return str(p if p.is_absolute() else (Path.cwd() / p).resolve())


def _find_split_jsonl(data_dir: Path, split: str) -> Path | None:
    """Locate the JSONL file for a given split in data_dir. Mirrors
    seg_map_calculations.py's own _find_split_jsonl exactly: exact match
    first (data_dir/{split}.jsonl), then any *.jsonl whose stem contains the
    split name; among those pick the shortest stem."""
    exact = data_dir / f"{split}.jsonl"
    if exact.exists():
        return exact
    candidates = sorted(
        [p for p in data_dir.glob("*.jsonl") if split.lower() in p.stem.lower()],
        key=lambda p: len(p.stem),
    )
    return candidates[0] if candidates else None


def _compute_one_map(encoder: GroundedSamEncoder, raw_image_path: Path, out_path: Path, device: str) -> bool:
    """Load raw_image_path, run it through encoder, save the class-ID PNG at
    out_path. Returns True on success, False on failure -- the caller skips
    the row rather than crashing the whole run.

    Split into two try/except blocks (load vs encode+save) so a failure says
    WHICH step broke -- same discipline as SegFormer's precompute_segmentation_maps.
    """
    try:
        img = Image.open(raw_image_path).convert("RGB")
    except Exception as e:
        exists = raw_image_path.exists()
        logger.warning(
            f"Image LOAD failed (Pillow could not open/decode) -- {raw_image_path} | "
            f"reason: {e} | path.exists()={exists}"
            + ("" if exists else "  -> the path itself is wrong (case, mount point, or a stale manifest entry)")
            + (f" | file size={raw_image_path.stat().st_size} bytes"
               "  -> 0 bytes or tiny usually means a broken symlink or failed copy" if exists else "")
        )
        return False

    try:
        arr = np.asarray(img).astype("float32") / 127.5 - 1.0  # [H,W,3] in [-1,1]
        img_tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)  # [1,3,H,W]
        with torch.no_grad():
            ids = encoder.label_ids(img_tensor)[0].cpu()  # [size_h, size_w] long
        out_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(ids.numpy().astype("uint8"), mode="L").save(out_path)
        return True
    except Exception as e:
        logger.warning(
            f"Encode/save FAILED (image loaded fine, Grounded-SAM or the PNG write threw) -- "
            f"{raw_image_path} -> {out_path} | image size/mode: {img.size} {img.mode} | reason: {e}"
        )
        return False


def precompute_grounded_sam_maps(
    encoder: GroundedSamEncoder,
    items: list[tuple[Path, Path]],
    device: str,
    skip_existing: bool,
    desc: str,
    rank: int = 0,
) -> dict[str, str]:
    """Core routine: given [(raw_image_path, out_path), ...] pairs already
    resolved by the caller, run each through the encoder and save a raw
    class-ID PNG. The "grounded_sam" in the name satisfies the visual-
    identity rule: you can tell at a glance this function belongs to THIS
    branch's pipeline, not segformer's or depth's.

    Mirrors segformer's own precompute_segmentation_maps in ROLE (a single
    shared driver every entry-point mode delegates into, returning a
    path-lookup dict the caller zips back against its own entries to build
    manifest rows) -- but deliberately NOT in batching: Grounded-SAM's
    Grounding DINO + SAM pipeline runs one image at a time internally (see
    GroundedSamEncoder.label_ids), so there is no batch_size here, unlike
    segformer's version.

    Returns {str(raw_image_path): str(out_path)} for every image that was
    processed OR found already cached (skip_existing) -- a missing raw image
    or a failed compute simply has no entry, which the caller treats as
    "skip this row" exactly the way segformer's build_segmentation_training_jsons
    treats a missing path_to_seg lookup.
    """
    path_to_seg: dict[str, str] = {}
    n_total = len(items)
    n_computed = n_skipped = n_failed = 0
    # tqdm's own bar is console/stderr-only (a live terminal, or nothing
    # useful once stdout is redirected to a file, e.g. under sbatch) -- it
    # never reaches the log FILE the logging module writes. Log a real
    # progress line into the file too, at a fixed number of checkpoints
    # regardless of how many images this run covers (6 or 6000), so a
    # `tail -f`/re-opened log always shows real, recent progress, not just
    # a startup banner and a final summary.
    log_every = max(1, n_total // 20)

    for i, (raw_image_path, out_path) in enumerate(tqdm(items, desc=desc, position=rank), start=1):
        if not raw_image_path.exists():
            continue
        if skip_existing and out_path.exists():
            path_to_seg[str(raw_image_path)] = str(out_path)
            n_skipped += 1
        elif _compute_one_map(encoder, raw_image_path, out_path, device):
            path_to_seg[str(raw_image_path)] = str(out_path)
            n_computed += 1
        else:
            n_failed += 1

        if i % log_every == 0 or i == n_total:
            logger.info(
                f"  [{desc}] {i}/{n_total} ({100 * i / n_total:.0f}%) -- "
                f"computed={n_computed} skipped(cached)={n_skipped} failed={n_failed}"
            )
    return path_to_seg


def _verify_grounded_sam_manifest(jsonl_path: Path) -> tuple[int, int]:
    """Last line of defence before training trusts this manifest -- same
    discipline as segformer's _verify_scan_seg_training_jsonl. The
    "grounded_sam" in the name satisfies the visual-identity rule.

    Checks:
      1. Required keys present (raw_image_path, seg_path, prompt).
      2. seg_path actually exists on disk (the PNG was really written).
      3. No two rows point at the SAME seg_path (a collision would silently
         pair the wrong image with a map at training time).
      4. No source image listed more than once (would be double-weighted).

    Returns (n_passed, n_failed) and logs a one-line PASS/FAIL summary.
    """
    with open(jsonl_path, "r", encoding="utf-8") as f:
        entries = [json.loads(line) for line in f if line.strip()]

    passed = failed = 0
    for i, entry in enumerate(entries):
        missing = [k for k in ("raw_image_path", "seg_path", "prompt") if k not in entry]
        if missing:
            logger.warning(f"  [FAIL] entry {i}: missing keys {missing}")
            failed += 1
            continue
        seg_p = Path(entry["seg_path"])
        if not seg_p.exists():
            logger.warning(f"  [FAIL] entry {i}: seg PNG not on disk: {seg_p}")
            failed += 1
            continue
        passed += 1

    seg_counts = Counter(os.path.normcase(str(Path(e["seg_path"]))) for e in entries if "seg_path" in e)
    img_counts = Counter(os.path.normcase(str(Path(e["raw_image_path"]))) for e in entries if "raw_image_path" in e)
    dup_segs = {p: n for p, n in seg_counts.items() if n > 1}
    dup_imgs = {p: n for p, n in img_counts.items() if n > 1}
    if dup_segs:
        worst_path, worst_n = max(dup_segs.items(), key=lambda kv: kv[1])
        logger.warning(
            f"  [FAIL] {len(dup_segs)} map file(s) referenced by MULTIPLE entries "
            f"(worst: {worst_n} entries -> {worst_path}). Every image must have its "
            f"OWN map -- re-run with --no_skip_existing if this looks like a stale run."
        )
        failed += sum(dup_segs.values())
        passed = max(0, passed - sum(dup_segs.values()))
    if dup_imgs:
        logger.warning(f"  [FAIL] {len(dup_imgs)} source image(s) listed more than once (double-weighted in training).")
        failed += sum(n - 1 for n in dup_imgs.values())

    status = "PASS" if failed == 0 else "FAIL"
    logger.info(f"  [{status}] {jsonl_path.name}: {passed}/{len(entries)} entries valid" + (f", {failed} FAILED" if failed else ""))
    return passed, failed


def build_grounded_sam_training_jsons(
    encoder: GroundedSamEncoder,
    input_jsonl: Path,
    output_dir: Path,
    device: str,
    image_path_key: str,
    limit: int | None,
    skip_existing: bool,
    raw_dir: Path | None = None,
    rank: int = 0,
    world_size: int = 1,
) -> list[dict]:
    """Read an EXISTING jsonl (one split's manifest) and compute a Grounded-
    SAM map for every image it references. "ground_truth" is carried through
    from the input row as-is (never guessed) -- "" if the input row doesn't
    have one.

    Naming mirrors segformer's build_segmentation_training_jsons (its
    --data_dir mode is the closest structural match: read an existing
    manifest, derive a dataset-root anchor, mirror the relative folder
    structure into a sibling map location, name each map after the image's
    own STEM). One real difference from segformer's version, because
    Grounded-SAM's reality is different: output naming here has no
    resize_mode suffix -- there's no resize-mode concept to keep separate
    runs from colliding (see module docstring).

    rank/world_size: when a sbatch script runs several instances of this
    script in parallel (see module docstring), the FULL manifest is still
    read and the collision guard still runs against the FULL manifest in
    every instance (cheap, and catches a real collision no matter how many
    instances are running) -- only the final per-image compute step is
    sharded.
    """
    with open(input_jsonl, "r", encoding="utf-8") as f:
        entries = [json.loads(line) for line in f if line.strip()]
    if limit is not None:
        entries = entries[:limit]

    abs_paths = []
    for entry in entries:
        p = Path(_get_image_path(entry, image_path_key))
        # Resolve to absolute HERE, not left relative -- a relative source
        # path written verbatim into raw_image_path would crash training the
        # moment Hydra changes the working directory to its own run folder.
        p = p if p.is_absolute() else (Path.cwd() / p).resolve()
        abs_paths.append(p)
        if not p.exists():
            logger.warning(f"missing {p}")

    existing = [p for p in abs_paths if p.exists()]
    if not existing:
        logger.error(f"no existing images in {input_jsonl} -- nothing to do.")
        return []

    if raw_dir is not None and raw_dir.exists() and all(_is_relative_to(p, raw_dir) for p in existing):
        dataset_root = raw_dir
    else:
        dataset_root = Path(os.path.commonpath([str(p) for p in existing]))
        if dataset_root.is_file():
            dataset_root = dataset_root.parent

    def _out_path_for(raw_image_path: Path) -> Path:
        rel_dir = raw_image_path.parent.relative_to(dataset_root)
        return output_dir / rel_dir / f"{raw_image_path.stem}_seg_map_grounded_sam.png"

    # ---- collision guard, before any GPU work -- runs against the FULL
    # manifest (not yet sharded), same discipline as segformer's
    # precompute_segmentation_maps: stop loudly on a real naming collision
    # instead of silently overwriting one image's map with another's. ----
    seen = {}
    for p in existing:
        out = _out_path_for(p)
        key = os.path.normcase(str(out))
        if key in seen:
            raise SystemExit(
                f"[FATAL] Two images would write the SAME seg map file:\n"
                f"    {seen[key]}\n    {p}\n  both -> {out}\n"
                f"  Each image needs its own map. Pass --raw_dir to pin a dataset-root "
                f"anchor if the auto-derived one is wrong for this manifest."
            )
        seen[key] = p

    pairs = _shard(list(zip(entries, abs_paths)), rank, world_size)
    desc = f"[rank {rank}] {input_jsonl.stem} ({input_jsonl.name})" if world_size > 1 else f"{input_jsonl.stem} ({input_jsonl.name})"

    # Phase 1: compute -- one shared call into the core routine, exactly the
    # way segformer's build_segmentation_training_jsons calls
    # precompute_segmentation_maps once and gets a path_to_seg dict back.
    # Filtered to existing paths BEFORE calling _out_path_for, matching the
    # original single-phase code exactly (it only ever called _out_path_for
    # after its own raw_image_path.exists() check) -- a missing image's
    # parent folder is not guaranteed to still resolve cleanly against
    # dataset_root, so this avoids a needless .relative_to() call on it.
    items = [
        (raw_image_path, _out_path_for(raw_image_path))
        for _entry, raw_image_path in pairs
        if raw_image_path.exists()
    ]
    path_to_seg = precompute_grounded_sam_maps(encoder, items, device, skip_existing, desc, rank)

    # Phase 2: build rows by zipping the ORIGINAL entries (prompt,
    # ground_truth) back against path_to_seg -- a missing lookup (raw image
    # didn't exist, or compute failed) means "skip this row", already logged
    # by precompute_grounded_sam_maps/_compute_one_map.
    rows = []
    for entry, raw_image_path in pairs:
        seg_str = path_to_seg.get(str(raw_image_path))
        if seg_str is None:
            continue
        # ground_truth: an explicit "ground_truth" key wins if the input row has
        # one; otherwise fall back to the input row's own "seg_path" -- verification
        # manifests like data/dataset/*.jsonl (built by pairing real images against
        # real hand-made Cityscapes labels, see outputs/verify_grounded_sam/) store
        # that real ground truth under "seg_path", not a dedicated "ground_truth"
        # key. Falls back to "" only if the input row has neither. Resolved to
        # absolute (_resolve_abs) same as every other path this script writes.
        rows.append({
            "raw_image_path": str(raw_image_path),
            "seg_path": seg_str,
            "prompt": entry.get("prompt", ""),
            "ground_truth": _resolve_abs(entry.get("ground_truth", entry.get("seg_path", ""))),
        })

    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_dir", default=None,
                    help="Folder with train.jsonl/val.jsonl/test.jsonl (e.g. data/custom_dataset). "
                         "Any *.jsonl whose stem contains 'train'/'val'/'test' is also accepted "
                         "(see _find_split_jsonl). A split with no matching file is skipped, not "
                         "an error. --train_jsonl/--val_jsonl/--test_jsonl override this per split.")
    p.add_argument("--train_jsonl", default=None, help="Explicit path for the train split (overrides --data_dir auto-discovery for it)")
    p.add_argument("--val_jsonl", default=None, help="Explicit path for the val split (overrides --data_dir auto-discovery for it)")
    p.add_argument("--test_jsonl", default=None, help="Explicit path for the test split (overrides --data_dir auto-discovery for it)")
    p.add_argument("--image_path", default="raw_image_path",
                    help="Key to read the image path from in each jsonl entry "
                         "(e.g. 'target' for data/train.jsonl) -- falls back to 'raw_image_path' if absent.")
    p.add_argument("--output_root", default=None,
                    help="Where to write {split}/<stem>_seg_map_grounded_sam.png. "
                         "Default (if omitted): a SIBLING of --data_dir, suffixed "
                         "'_seg_map_grounded_sam'. Only computable when --data_dir is given -- "
                         "explicit-paths-only mode (--train_jsonl/--val_jsonl/--test_jsonl with no "
                         "--data_dir) has no dataset root to be a sibling of, so --output_root is "
                         "required then.")
    p.add_argument("--manifest_out", default=None, help="Where to write train.jsonl/val.jsonl/test.jsonl (default: output_root)")
    p.add_argument("--raw_dir", default=None,
                    help="Pin the dataset-root anchor used to mirror the output folder structure. "
                         "Only used if EVERY referenced image actually lives under it; otherwise "
                         "the root is auto-derived as the deepest folder shared by all images in "
                         "that split's manifest.")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=800)
    # 0.15/0.15, not Grounding DINO's own README-recommended 0.3/0.25 -- measured
    # 92.6%/89.7% pixel coverage at 0.15 vs visible stuff-class gaps at 0.3/0.25
    # on real test images (field guide Lesson 21). Keep in sync with
    # GroundedSamEncoder's own default in src/encoders/grounded_sam_encoder.py.
    p.add_argument("--box_threshold", type=float, default=0.15)
    p.add_argument("--text_threshold", type=float, default=0.15)
    p.add_argument("--limit_train", type=int, default=None, help="cap on train images (e.g. half the dataset)")
    p.add_argument("--limit_val", type=int, default=None, help="cap on val images")
    p.add_argument("--limit_test", type=int, default=None, help="cap on test images")
    p.add_argument("--no_skip_existing", action="store_true")
    p.add_argument("--local_files_only", type=lambda s: s.lower() != "false", default=True)
    p.add_argument("--dino_model_path", default="checkpoints/local_models/grounding-dino-tiny",
                    help="Local folder for Grounding DINO -- only read when --local_files_only "
                         "(default). Same convention as configs/experiment/train_grounded_sam*.yaml's "
                         "dino_model_path.")
    p.add_argument("--dino_model_name", default="IDEA-Research/grounding-dino-tiny",
                    help="HF Hub id, only used when --local_files_only=false.")
    p.add_argument("--sam_model_path", default="checkpoints/local_models/sam-vit-base",
                    help="Local folder for SAM -- only read when --local_files_only (default).")
    p.add_argument("--sam_model_name", default="facebook/sam-vit-base",
                    help="HF Hub id, only used when --local_files_only=false.")
    p.add_argument("--device", default=None, help="e.g. 'cpu' to force CPU (auto-detects GPU if omitted). "
                                                    "On a multi-GPU node, pin per-task via CUDA_VISIBLE_DEVICES "
                                                    "in your sbatch script rather than this flag.")
    p.add_argument("--rank", type=int, default=0,
                    help="(multi-task / SLURM) This invocation's shard index, 0-based. Your "
                         "sbatch script sets this per task, e.g. --rank $SLURM_PROCID. "
                         "Default 0 (process the whole list) -- nothing changes for a "
                         "normal single-GPU run.")
    p.add_argument("--world_size", type=int, default=1,
                    help="(multi-task / SLURM) Total number of parallel instances of this "
                         "script sharding the work, e.g. --world_size $SLURM_NTASKS. "
                         "Default 1. See module docstring for a full sbatch example.")
    args = p.parse_args()

    if args.world_size < 1:
        p.error("--world_size must be >= 1.")
    if not (0 <= args.rank < args.world_size):
        p.error(f"--rank {args.rank} must be in [0, --world_size={args.world_size}).")

    log_dir = PROJECT_ROOT / "outputs" / "logs"
    _log_path = _setup_logging(log_dir, args.rank if args.world_size > 1 else None)

    if args.width % 32 or args.height % 32:
        p.error(f"--width {args.width} and --height {args.height} must both be divisible by 32 "
                 f"(SDXL's VAE/8 x 2 UNet halvings/4 -- see field guide Lesson 4/19).")

    if args.train_jsonl is None and args.val_jsonl is None and args.test_jsonl is None and args.data_dir is None:
        p.error("Need --data_dir (folder auto-discovery) or at least one of --train_jsonl/"
                 "--val_jsonl/--test_jsonl (explicit paths) -- see module docstring.")

    from src.utils import resolve_device
    device = resolve_device(args.device)

    data_dir = Path(args.data_dir) if args.data_dir else None

    # Sibling-of-dataset-root default: never written INTO the dataset folder,
    # always next to it, named after it. e.g. --data_dir data/custom_dataset
    # -> data/custom_dataset_seg_map_grounded_sam
    if args.output_root:
        output_root = Path(args.output_root).resolve()
    elif data_dir is not None:
        # .resolve() here too -- a relative --data_dir (e.g. "data/dataset")
        # would otherwise leave output_root relative, and every seg_path
        # written into the manifest inherits that relativity. Harmless until
        # something changes the process cwd -- but grounded_sam_training.py's
        # hydra.job.chdir=true does exactly that for every run, so a relative
        # seg_path resolves against the run's own output folder instead of
        # the repo root and local_seg.py's Image.open(seg_path) throws
        # FileNotFoundError on the very first batch. raw_image_path already
        # gets this same absolute-resolution treatment above -- seg_path
        # (via output_root) needs it for the identical reason.
        output_root = (data_dir.parent / (data_dir.name + "_seg_map_grounded_sam")).resolve()
    else:
        p.error("--output_root is required when --data_dir is not given -- explicit-paths-only "
                 "mode (--train_jsonl/--val_jsonl/--test_jsonl) has no dataset root to default a "
                 "sibling folder from.")
    manifest_out = Path(args.manifest_out) if args.manifest_out else output_root
    manifest_out.mkdir(parents=True, exist_ok=True)

    # Resolve each split's input jsonl: an explicit --{split}_jsonl always
    # wins; otherwise auto-discover it from --data_dir (_find_split_jsonl);
    # otherwise this split simply isn't run (no error -- a real dataset
    # might legitimately have no test split, e.g.).
    explicit = {"train": args.train_jsonl, "val": args.val_jsonl, "test": args.test_jsonl}
    split_jsonl_path: dict[str, Path | None] = {}
    for split in ("train", "val", "test"):
        if explicit[split]:
            split_jsonl_path[split] = Path(explicit[split])
        elif data_dir is not None:
            split_jsonl_path[split] = _find_split_jsonl(data_dir, split)
        else:
            split_jsonl_path[split] = None

    logger.info("=" * 60)
    logger.info("  Grounded-SAM segmentation map calculation")
    logger.info(f"  Device      : {device}"
                + (f"  (rank {args.rank}/{args.world_size})" if args.world_size > 1 else ""))
    logger.info(f"  Size        : {args.width}x{args.height} (width x height)")
    logger.info(f"  Data dir    : {data_dir}  (auto-discovery fallback)")
    for split in ("train", "val", "test"):
        logger.info(f"  {split:<5} jsonl : {split_jsonl_path[split] or '(none found -- split skipped)'}")
    logger.info(f"  Output      : {output_root}")
    logger.info("=" * 60)

    # ── Pick LOCAL model folders vs HUB ids from the local_files_only flag ──
    # Exact same pattern as grounded_sam_training.py/grounded_sam_inference.py:
    # local_files_only=True alone does NOT redirect where a model loads from,
    # it only blocks network fallback. GroundedSamEncoder's own dino_model/
    # sam_model defaults are HF Hub repo ids ("IDEA-Research/grounding-dino-
    # tiny", "facebook/sam-vit-base") -- passing local_files_only=True with
    # THOSE ids still resolves against the default ~/.cache/huggingface/hub/
    # cache-by-repo-id layout, never checkpoints/local_models/, so a fully
    # offline node with nothing in that default cache fails outright. Local
    # paths are made absolute from PROJECT_ROOT (this script's own
    # non-Hydra equivalent of training/inference's _root = get_original_cwd())
    # so this still works no matter what directory it's launched from.
    if args.local_files_only:
        os.environ["HF_HUB_OFFLINE"]      = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        dino_model = str(PROJECT_ROOT / args.dino_model_path)
        sam_model  = str(PROJECT_ROOT / args.sam_model_path)
    else:
        dino_model = args.dino_model_name
        sam_model  = args.sam_model_name

    encoder = GroundedSamEncoder(
        size=(args.width, args.height),
        dino_model=dino_model,
        sam_model=sam_model,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
        local_files_only=args.local_files_only,
    ).to(device)
    encoder.eval()

    t_start = time.time()
    split_counts = {}
    limits = {"train": args.limit_train, "val": args.limit_val, "test": args.limit_test}

    for split in ("train", "val", "test"):
        input_jsonl = split_jsonl_path[split]
        if input_jsonl is None:
            logger.warning(f"no jsonl found/given for split '{split}' -- skipping")
            continue
        if not input_jsonl.exists():
            logger.warning(f"jsonl for split '{split}' does not exist: {input_jsonl} -- skipping")
            continue

        rows = build_grounded_sam_training_jsons(
            encoder, input_jsonl, output_root, device,
            image_path_key=args.image_path, limit=limits[split],
            skip_existing=not args.no_skip_existing,
            raw_dir=Path(args.raw_dir).resolve() if args.raw_dir else None,
            rank=args.rank, world_size=args.world_size,
        )

        # Multi-task runs: each instance writes its OWN rank-suffixed manifest
        # (never a shared file -- N processes writing the same path would
        # race/clobber). Merge them yourself once every task is done, e.g.:
        #   cat train_rank*.jsonl > train.jsonl
        suffix = f"_rank{args.rank}" if args.world_size > 1 else ""
        out_jsonl = manifest_out / f"{split}{suffix}.jsonl"
        with open(out_jsonl, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        logger.info(f"  Written {len(rows)} entries -> {out_jsonl}")
        _verify_grounded_sam_manifest(out_jsonl)
        split_counts[split] = len(rows)

    elapsed = time.time() - t_start
    logger.info(f"\nDone. {elapsed:.1f}s elapsed. {split_counts}")
    logger.info(f"Log file: {_log_path}")


if __name__ == "__main__":
    main()
