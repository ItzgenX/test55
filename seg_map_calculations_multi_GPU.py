"""
seg_map_calculations_multi_GPU.py
----------------------------------
Normal python script -- `python seg_map_calculations_multi_GPU.py --data_dir
data/dataset --width 1280 --height 800`, nothing else. Automatically uses
every GPU allocated to the process (torch.cuda.device_count()) -- 1, 4,
whatever -- with no rank/world_size flag or concept anywhere.

Only supports --data_dir mode (the one that matters for real-scale dataset
precompute) -- for --image/--json_file/--dataset_dir/--input_dir single-item
or scan modes, use seg_map_calculations.py directly, single-GPU is already
instant for those.

WHY THIS NEEDED MORE THAN A THIN WRAPPER (unlike grounded_sam's per-image
design): seg_map_calculations.py's build_segmentation_training_jsons derives
its shared output folder (dataset_root -> sibling_root) via
os.path.commonpath() over the image list EACH TIME IT RUNS -- a real,
already-documented fragility in that file (see its own --raw_dir docstring):
two runs against different subsets of one dataset can derive two DIFFERENT
sibling locations. Naively splitting the image list across GPU workers and
having each one call that function independently would hit exactly this --
a worker whose shard happens to cover fewer folders could derive a
different, deeper root than the others, silently splitting maps across
mismatched folder trees. Fixed by resolving dataset_root/sibling_root/every
image's output path ONCE, centrally, in this process, BEFORE any worker
starts -- then handing each worker its shard via
precompute_segmentation_maps's own out_path_fn hook (a callable(image_path)
-> Path it already supports "when given, it FULLY decides where each seg
PNG is saved") instead of letting each worker derive paths on its own.

Every worker calls precompute_segmentation_maps -- SegFormer's own batched
core routine -- UNCHANGED, on its shard, with out_path_fn pointing at the
centrally pre-resolved paths. No compute logic is duplicated here.

Progress: each worker prints its OWN tqdm bar (precompute_segmentation_maps's
own, unmodified) -- with N workers these interleave in the shared terminal.
Accepted trade-off: combining them into one bar would mean either modifying
or duplicating that function's internal batching loop, which this script
deliberately avoids. Real progress still lands in each worker's own summary
line once its shard finishes, and in this script's own log file.

NOT TESTED ON HARDWARE this was written for: the author's dev machine has
one GPU. Real execution DID verify the whole pipeline on that one GPU
(central path resolution, worker startup, result routing, manifest
writing/merging) -- true parallelism across several GPUs needs hardware
this machine doesn't have.
"""

import argparse
import json
import logging
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

from seg_map_calculations import (
    PROJECT_ROOT,
    DEFAULT_SEG_MODEL,
    _get_image_path,
    _find_split_jsonl,
    _verify_scan_seg_training_jsonl,
    precompute_segmentation_maps,
)

SCRIPT_NAME = "seg_map_calc_multi_gpu"
logger = logging.getLogger(SCRIPT_NAME)


def _setup_logging(log_dir: Path) -> Path:
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"{SCRIPT_NAME}_{ts}.log"
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
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


def _is_relative_to(p: Path, other: Path) -> bool:
    try:
        p.relative_to(other)
        return True
    except ValueError:
        return False


def _resolve_all_splits(data_dir, raw_dir, image_path_key, subset_n):
    """Central, one-time resolution (parent process, before any worker
    starts): read every split's entries, resolve absolute image paths, pool
    them ALL to derive ONE dataset_root/sibling_root -- see module docstring
    for why this must happen once, not per-worker."""
    split_entries, split_images, all_images = {}, {}, []
    for split in ("train", "val", "test"):
        src = _find_split_jsonl(data_dir, split)
        if src is None:
            logger.warning(f"no jsonl for split '{split}' -- skipping")
            continue
        with open(src, "r", encoding="utf-8") as f:
            entries = [json.loads(line) for line in f if line.strip()]
        if subset_n:
            entries = entries[:subset_n]
        abs_paths = []
        for entry in entries:
            p = Path(_get_image_path(entry, image_path_key))
            abs_p = p if p.is_absolute() else (Path.cwd() / p).resolve()
            abs_paths.append(abs_p)
            if not abs_p.exists():
                logger.warning(f"missing {abs_p}")
        split_entries[split] = entries
        split_images[split] = abs_paths
        all_images.extend(abs_paths)

    if not all_images:
        raise SystemExit("[FATAL] No images found in any split -- nothing to do.")

    if raw_dir is not None and raw_dir.exists() and all(_is_relative_to(p, raw_dir) for p in all_images):
        dataset_root = raw_dir
    else:
        dataset_root = Path(os.path.commonpath([str(p) for p in all_images]))
        if dataset_root.is_file():
            dataset_root = dataset_root.parent

    return split_entries, split_images, dataset_root


def _gpu_worker(gpu_id, shard, out_paths, sibling_root, width, height, model_name, batch_size, local_files_only, result_queue):
    """One process, one GPU. shard: list of (split, image_path_str) this
    worker owns. Calls precompute_segmentation_maps -- SegFormer's own
    batched core routine -- UNCHANGED, with out_path_fn reading from the
    CENTRALLY pre-resolved out_paths dict (never derives paths itself)."""
    # try/finally: guarantees worker_done always reaches the parent no
    # matter what fails in between, so main()'s result_queue.get() loop can
    # never hang waiting for a message that was never sent (see
    # grounded_sam_map_calculations_multi_GPU.py -- same fix, found there
    # via a real deadlock during its own verification).
    try:
        image_strs = [img for _split, img in shard]
        path_to_seg = precompute_segmentation_maps(
            image_paths=image_strs,
            output_dir=sibling_root,
            size=(width, height),
            batch_size=batch_size,
            model_name=model_name,
            device=f"cuda:{gpu_id}",
            skip_existing=False,  # already filtered centrally before sharding
            out_path_fn=lambda p: out_paths[str(Path(p))],
            resize_mode="aspect",
        )
        result_queue.put(("result", gpu_id, path_to_seg))
    except Exception as e:
        result_queue.put(("worker_failed", gpu_id, f"{type(e).__name__}: {e}"))
    finally:
        result_queue.put(("worker_done", gpu_id, None))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_dir", required=True)
    p.add_argument("--raw_dir", default=None)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=800)
    p.add_argument("--batch_size", type=int, default=4, help="Per-GPU batch size.")
    p.add_argument("--model", default="checkpoints/local_models/segformer-b5-cityscapes")
    p.add_argument("--local_files_only", type=lambda s: s.lower() != "false", default=True)
    p.add_argument("--image_path", default="raw_image_path")
    p.add_argument("--dry_run_n", type=int, default=None)
    p.add_argument("--no_skip_existing", action="store_true")
    p.add_argument(
        "--output_root", type=str, default=None,
        help="Where to save the seg-ID PNGs (mirrors seg_map_calculations.py's "
             "own --output_root / grounded_sam_map_calculations.py's --output_root). "
             "Default (if omitted): a SIBLING of --data_dir, suffixed "
             "'_seg_map_aspect'.",
    )
    p.add_argument(
        "--manifest_out", type=str, default=None,
        help="Where to write train/val/test.jsonl (mirrors seg_map_calculations.py's "
             "own --manifest_out). Default (if omitted): the SAME folder the "
             "seg-ID PNGs landed in (--output_root, or its own sibling default).",
    )
    args = p.parse_args()

    _log_path = _setup_logging(PROJECT_ROOT / "outputs" / "logs")

    if args.width % 32 or args.height % 32:
        p.error(f"--width {args.width}/--height {args.height} must both be divisible by 32.")

    import torch
    if not torch.cuda.is_available():
        p.error("No CUDA GPU visible -- this script is GPU-only.")
    num_gpus = torch.cuda.device_count()

    if args.local_files_only:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        model_name = str(PROJECT_ROOT / args.model)
    else:
        model_name = DEFAULT_SEG_MODEL

    data_dir = Path(args.data_dir).resolve()
    raw_dir = Path(args.raw_dir).resolve() if args.raw_dir else None

    logger.info("=" * 60)
    logger.info(f"  Segmentation map calculation -- {num_gpus} GPU(s) (auto-detected)")
    logger.info(f"  Size : {args.width}x{args.height}")
    logger.info("=" * 60)

    split_entries, split_images, dataset_root = _resolve_all_splits(
        data_dir, raw_dir, args.image_path, args.dry_run_n,
    )
    suffix = "_seg_map_aspect"
    # Explicit --output_root wins outright (resolved, same reasoning as
    # seg_map_calculations.py's own --output_root): otherwise the existing
    # sibling-of-dataset_root default, unchanged.
    sibling_root = (Path(args.output_root).resolve() if args.output_root
                     else dataset_root.parent / (dataset_root.name + suffix))
    sibling_root.mkdir(parents=True, exist_ok=True)
    logger.info(f"  Dataset root : {dataset_root}")
    logger.info(f"  Seg maps     : {sibling_root}"
                + ("" if args.output_root else "  (sibling folder, mirrored structure)"))

    def out_fn(img: Path) -> Path:
        rel_dir = img.parent.relative_to(dataset_root)
        return sibling_root / rel_dir / (img.stem + suffix + ".png")

    # ---- collision guard, before any GPU work, against every image in every split ----
    seen = {}
    out_paths = {}
    for split, imgs in split_images.items():
        for img in imgs:
            if not img.exists():
                continue
            out = out_fn(img)
            out_paths[str(img)] = out
            key = os.path.normcase(str(out))
            if key in seen:
                raise SystemExit(
                    f"[FATAL] Two images would write the SAME seg map file:\n"
                    f"    {seen[key]}\n    {img}\n  both -> {out}\n"
                    f"  Pass --raw_dir to pin the dataset-root anchor if the auto-derived one is wrong."
                )
            seen[key] = img

    # ---- split cached (skip_existing) vs to-compute, tagged with split ----
    skip_existing = not args.no_skip_existing
    all_shard, cached_by_split = [], {s: set() for s in split_images}
    for split, imgs in split_images.items():
        for img in imgs:
            if not img.exists():
                continue
            out = out_paths[str(img)]
            if skip_existing and out.exists():
                cached_by_split[split].add(str(img))
            else:
                all_shard.append((split, str(img)))
    logger.info(f"  To compute: {len(all_shard)}  |  Cached: {sum(len(v) for v in cached_by_split.values())}")

    computed_by_gpu = {}
    if all_shard:
        ctx = mp.get_context("spawn")
        result_queue = ctx.Queue()
        shards = [all_shard[i::num_gpus] for i in range(num_gpus)]
        workers = [
            ctx.Process(target=_gpu_worker,
                        args=(gpu_id, shards[gpu_id], out_paths, sibling_root, args.width, args.height,
                              model_name, args.batch_size, args.local_files_only, result_queue))
            for gpu_id in range(num_gpus) if shards[gpu_id]
        ]
        for w in workers:
            w.start()

        active = len(workers)
        failed = []
        while active > 0:
            # worker_done is the ONLY message that decrements active -- the
            # worker's finally block guarantees exactly one of these per
            # worker no matter what happens inside it. worker_failed is
            # purely informational (logged), sent in addition to, never
            # instead of, worker_done.
            kind, gpu_id, payload = result_queue.get()
            if kind == "result":
                computed_by_gpu[gpu_id] = payload
                logger.info(f"  GPU {gpu_id}: {len(payload)}/{len(shards[gpu_id])} computed")
            elif kind == "worker_failed":
                failed.append(gpu_id)
                logger.error(f"  GPU {gpu_id}: FAILED -- {payload}")
            elif kind == "worker_done":
                active -= 1
        for w in workers:
            w.join()
        if not computed_by_gpu and failed:
            raise SystemExit(f"[FATAL] All GPU worker(s) failed: {failed}")

    all_computed = {}
    for d in computed_by_gpu.values():
        all_computed.update(d)

    # ---- write + verify unified manifests ----
    split_counts = {}
    # manifest_out: explicit value wins outright; otherwise defaults to
    # sibling_root itself -- the SAME folder the PNGs just landed in --
    # matching seg_map_calculations.py's --data_dir mode default (which in
    # turn matches grounded_sam_map_calculations.py's own default) exactly.
    out_dir = (Path(args.manifest_out).resolve() if args.manifest_out else sibling_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"  Manifest     : {out_dir}"
                + ("" if args.manifest_out else "  (same folder as the seg maps)"))
    for split, entries in split_entries.items():
        imgs = split_images[split]
        out_entries = []
        for entry, img in zip(entries, imgs):
            key = str(img)
            if key in cached_by_split[split]:
                seg_str = str(out_paths[key])
            else:
                seg_str = all_computed.get(key)
            if seg_str is None:
                continue
            _gt = entry.get("ground_truth", "")
            if _gt and not Path(_gt).is_absolute():
                _gt = (Path.cwd() / _gt).resolve().as_posix()
            out_entries.append({
                "raw_image_path": img.as_posix(),
                "seg_path": Path(seg_str).resolve().as_posix(),
                "prompt": entry["prompt"],
                "ground_truth": _gt,
            })
        out_path = out_dir / f"{split}.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for row in out_entries:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        logger.info(f"  Written {len(out_entries)} entries -> {out_path}")
        _verify_scan_seg_training_jsonl(out_path, "seg_path", suffix, name_from="stem")
        split_counts[split] = len(out_entries)

    logger.info(f"\nDone. {split_counts}")
    logger.info(f"Log file: {_log_path}")


if __name__ == "__main__":
    main()
