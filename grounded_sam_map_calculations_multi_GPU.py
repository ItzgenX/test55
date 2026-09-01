"""
grounded_sam_map_calculations_multi_GPU.py
-------------------------------------------
Normal python script -- `python grounded_sam_map_calculations_multi_GPU.py
--data_dir ... --width 1280 --height 800`, nothing else. Automatically uses
every GPU allocated to the process (torch.cuda.device_count()) -- 1, 4,
whatever -- with no rank/world_size flag or concept anywhere, hidden or
otherwise. Give it 4 GPUs, it uses 4; give it 1, it uses 1; same command
either way.

HOW: spawns one worker process per GPU (multiprocessing, 'spawn' context --
required for CUDA safety), all pulling from ONE shared task queue covering
every split at once, so per-image cost variance (SAM's mask-decode time
scales with detected box count, which varies a lot) can't strand one GPU on
a "hard" slice while another sits idle. Results and progress route back
through one queue to this process, which owns the one combined progress bar
and writes the final, already-unified train/val/test.jsonl directly -- no
per-GPU files to merge afterward.

Every actual per-image routine (image loading, encoding, PNG saving,
manifest-row building, verification) is IMPORTED from
grounded_sam_map_calculations.py, not copied -- this file only adds the
multi-GPU orchestration around it, so a fix there never has to be
duplicated here too.

NOT TESTED ON HARDWARE this was written for: the author's dev machine has
one GPU. Real execution DID verify the whole pipeline on that one GPU
(worker startup, task distribution, progress/result routing, manifest
writing) -- what's unverified is true parallelism across several GPUs at
once, which needs hardware this machine doesn't have.

Every other flag (--train_jsonl, --limit_train, --box_threshold, ...) works
exactly like grounded_sam_map_calculations.py's own -- see that script's
docstring/--help for the full list.
"""

import argparse
import json
import logging
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

from tqdm import tqdm

from grounded_sam_map_calculations import (
    PROJECT_ROOT,
    _compute_one_map,
    _find_split_jsonl,
    _get_image_path,
    _is_relative_to,
    _resolve_abs,
    _verify_grounded_sam_manifest,
)

SCRIPT_NAME = "grounded_sam_map_calc_multi_gpu"
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


def _out_path_for(raw_image_path: Path, dataset_root: Path, output_dir: Path) -> Path:
    rel_dir = raw_image_path.parent.relative_to(dataset_root)
    return output_dir / rel_dir / f"{raw_image_path.stem}_seg_map_grounded_sam.png"


def _build_tasks_for_split(split, input_jsonl, output_dir, image_path_key, limit, skip_existing, raw_dir):
    """CPU-only: resolve every image's absolute + output path, run the
    collision guard (before any GPU work), split into cached rows (no GPU
    work needed) vs tasks still needing computing. Same logic as
    grounded_sam_map_calculations.py's build_grounded_sam_training_jsons,
    just split into "figure out the work" (here) vs "do the work" (the
    worker processes) so it can run once in this process before any GPU
    spins up."""
    with open(input_jsonl, "r", encoding="utf-8") as f:
        entries = [json.loads(line) for line in f if line.strip()]
    if limit is not None:
        entries = entries[:limit]

    abs_paths = []
    for entry in entries:
        p = Path(_get_image_path(entry, image_path_key))
        p = p if p.is_absolute() else (Path.cwd() / p).resolve()
        abs_paths.append(p)
        if not p.exists():
            logger.warning(f"missing {p}")

    existing = [p for p in abs_paths if p.exists()]
    if not existing:
        logger.error(f"no existing images in {input_jsonl} -- nothing to do for split '{split}'.")
        return [], []

    if raw_dir is not None and raw_dir.exists() and all(_is_relative_to(p, raw_dir) for p in existing):
        dataset_root = raw_dir
    else:
        dataset_root = Path(os.path.commonpath([str(p) for p in existing]))
        if dataset_root.is_file():
            dataset_root = dataset_root.parent

    seen = {}
    for p in existing:
        out = _out_path_for(p, dataset_root, output_dir)
        key = os.path.normcase(str(out))
        if key in seen:
            raise SystemExit(
                f"[FATAL] Two images would write the SAME seg map file:\n"
                f"    {seen[key]}\n    {p}\n  both -> {out}\n"
                f"  Pass --raw_dir to pin the dataset-root anchor if the auto-derived one is wrong."
            )
        seen[key] = p

    tasks, cached_rows = [], []
    for entry, raw_image_path in zip(entries, abs_paths):
        if not raw_image_path.exists():
            continue
        out_path = _out_path_for(raw_image_path, dataset_root, output_dir)
        prompt = entry.get("prompt", "")
        ground_truth = _resolve_abs(entry.get("ground_truth", entry.get("seg_path", "")))
        if skip_existing and out_path.exists():
            cached_rows.append({"raw_image_path": str(raw_image_path), "seg_path": str(out_path),
                                 "prompt": prompt, "ground_truth": ground_truth})
        else:
            tasks.append((split, raw_image_path, out_path, prompt, ground_truth))
    return tasks, cached_rows


def _gpu_worker(gpu_id, task_queue, result_queue, width, height, dino_model, sam_model,
                 box_threshold, text_threshold, local_files_only):
    """One process, one GPU. Pulls tasks until it sees the sentinel None.
    Never touches the terminal/log file/tqdm directly -- routes everything
    through result_queue so output from several processes never
    interleaves. Imports happen here, not at module level, so this
    (spawned, no inherited CUDA state) process does its own clean import."""
    import torch
    from src.encoders.grounded_sam_encoder import GroundedSamEncoder

    torch.cuda.set_device(gpu_id)
    device = f"cuda:{gpu_id}"
    try:
        encoder = GroundedSamEncoder(
            size=(width, height), dino_model=dino_model, sam_model=sam_model,
            box_threshold=box_threshold, text_threshold=text_threshold,
            local_files_only=local_files_only,
        ).to(device)
        encoder.eval()
    except Exception as e:
        result_queue.put(("worker_failed", gpu_id, f"{type(e).__name__}: {e}"))
        return

    result_queue.put(("worker_ready", gpu_id))
    # try/finally around the ENTIRE loop, not just the per-task compute
    # call: run_multi_gpu()'s result_queue.get() loop blocks forever
    # waiting for exactly num_gpus worker_done/worker_failed messages -- if
    # ANY exception anywhere in this loop (not just inside
    # _compute_one_map) escaped uncaught, this process would exit silently
    # with no message ever sent, and the parent would hang forever with no
    # indication why. The finally guarantees worker_done always fires.
    try:
        while True:
            task = task_queue.get()
            if task is None:
                break
            split, raw_image_path, out_path, prompt, ground_truth = task
            # _compute_one_map returns a plain bool and logs its own
            # warnings internally (against grounded_sam_map_calculations.py's
            # own logger, unconfigured in this fresh child process -- its
            # messages fall through to Python logging's default stderr
            # handler, which is fine, just not routed into this script's own
            # structured log file).
            try:
                ok = _compute_one_map(encoder, raw_image_path, out_path, device)
            except Exception as e:
                result_queue.put(("log", "warning",
                                   f"_compute_one_map raised {type(e).__name__}: {e} -- {raw_image_path}"))
                ok = False
            result_queue.put(("progress", ok))
            if ok:
                result_queue.put(("result", split, {"raw_image_path": str(raw_image_path), "seg_path": str(out_path),
                                                     "prompt": prompt, "ground_truth": ground_truth}))
    finally:
        result_queue.put(("worker_done", gpu_id))


def run_multi_gpu(tasks, num_gpus, width, height, dino_model, sam_model, box_threshold, text_threshold, local_files_only):
    """Spawns num_gpus workers sharing ONE task queue, drains their combined
    progress/results/logs into ONE tqdm bar, returns {split: [row, ...]}."""
    ctx = mp.get_context("spawn")  # 'fork' is unsafe with CUDA
    task_queue, result_queue = ctx.Queue(), ctx.Queue()
    for t in tasks:
        task_queue.put(t)
    for _ in range(num_gpus):
        task_queue.put(None)

    workers = [
        ctx.Process(target=_gpu_worker,
                    args=(gpu_id, task_queue, result_queue, width, height, dino_model, sam_model,
                          box_threshold, text_threshold, local_files_only))
        for gpu_id in range(num_gpus)
    ]
    for w in workers:
        w.start()

    manifest_rows = {"train": [], "val": [], "test": []}
    n_computed = n_failed = n_seen = ready = 0
    active = num_gpus
    failed_workers = []
    n_total = len(tasks)
    log_every = max(1, n_total // 20) if n_total else 1

    pbar = tqdm(total=n_total, desc="Computing Grounded-SAM maps (all splits, all GPUs)")
    while active > 0:
        kind, *rest = result_queue.get()
        if kind == "worker_ready":
            ready += 1
            logger.info(f"  GPU {rest[0]}: worker ready")
        elif kind == "worker_failed":
            gpu_id, reason = rest
            failed_workers.append(gpu_id)
            logger.error(f"  GPU {gpu_id}: FAILED TO START -- {reason}")
            active -= 1
        elif kind == "worker_done":
            active -= 1
            logger.info(f"  GPU {rest[0]}: done")
        elif kind == "log":
            level, text = rest
            getattr(logger, level)(text)
        elif kind == "progress":
            n_seen += 1
            n_computed += 1 if rest[0] else 0
            n_failed += 0 if rest[0] else 1
            pbar.update(1)
            if n_seen % log_every == 0 or n_seen == n_total:
                logger.info(f"  {n_seen}/{n_total} ({100 * n_seen / max(n_total, 1):.0f}%) -- "
                            f"computed={n_computed} failed={n_failed}")
        elif kind == "result":
            split, row = rest
            manifest_rows[split].append(row)
    pbar.close()
    for w in workers:
        w.join()

    if ready == 0:
        raise SystemExit(f"[FATAL] All {num_gpus} GPU worker(s) failed to start -- see errors above.")
    if failed_workers:
        logger.warning(f"  {len(failed_workers)}/{num_gpus} GPU(s) failed to start (GPUs {failed_workers}) -- "
                        f"the remaining {ready} completed all {n_total} tasks between them, just slower.")
    if n_seen != n_total:
        raise SystemExit(f"[FATAL] Expected {n_total} results, got {n_seen} -- a worker likely crashed "
                          f"mid-task. Re-run with --no_skip_existing to recompute what's missing.")
    return manifest_rows


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_dir", default=None)
    p.add_argument("--train_jsonl", default=None)
    p.add_argument("--val_jsonl", default=None)
    p.add_argument("--test_jsonl", default=None)
    p.add_argument("--image_path", default="raw_image_path")
    p.add_argument("--output_root", default=None)
    p.add_argument("--manifest_out", default=None)
    p.add_argument("--raw_dir", default=None)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=800)
    p.add_argument("--box_threshold", type=float, default=0.15)
    p.add_argument("--text_threshold", type=float, default=0.15)
    p.add_argument("--limit_train", type=int, default=None)
    p.add_argument("--limit_val", type=int, default=None)
    p.add_argument("--limit_test", type=int, default=None)
    p.add_argument("--no_skip_existing", action="store_true")
    p.add_argument("--local_files_only", type=lambda s: s.lower() != "false", default=True)
    p.add_argument("--dino_model_path", default="checkpoints/local_models/grounding-dino-tiny")
    p.add_argument("--sam_model_path", default="checkpoints/local_models/sam-vit-base")
    args = p.parse_args()

    log_dir = PROJECT_ROOT / "outputs" / "logs"
    _log_path = _setup_logging(log_dir)

    if args.width % 32 or args.height % 32:
        p.error(f"--width {args.width} and --height {args.height} must both be divisible by 32.")
    if args.train_jsonl is None and args.val_jsonl is None and args.test_jsonl is None and args.data_dir is None:
        p.error("Need --data_dir or at least one of --train_jsonl/--val_jsonl/--test_jsonl.")

    import torch
    if not torch.cuda.is_available():
        p.error("No CUDA GPU visible -- this script is GPU-only.")
    num_gpus = torch.cuda.device_count()  # auto: whatever is actually allocated to this process

    if args.local_files_only:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    data_dir = Path(args.data_dir) if args.data_dir else None
    if args.output_root:
        output_root = Path(args.output_root).resolve()
    elif data_dir is not None:
        output_root = (data_dir.parent / (data_dir.name + "_seg_map_grounded_sam")).resolve()
    else:
        p.error("--output_root is required when --data_dir is not given.")
    manifest_out = Path(args.manifest_out) if args.manifest_out else output_root
    manifest_out.mkdir(parents=True, exist_ok=True)

    explicit = {"train": args.train_jsonl, "val": args.val_jsonl, "test": args.test_jsonl}
    split_jsonl_path = {}
    for split in ("train", "val", "test"):
        if explicit[split]:
            split_jsonl_path[split] = Path(explicit[split])
        elif data_dir is not None:
            split_jsonl_path[split] = _find_split_jsonl(data_dir, split)
        else:
            split_jsonl_path[split] = None

    logger.info("=" * 60)
    logger.info(f"  Grounded-SAM map calculation -- {num_gpus} GPU(s) (auto-detected)")
    logger.info(f"  Size   : {args.width}x{args.height}")
    logger.info(f"  Output : {output_root}")
    logger.info("=" * 60)

    t_start = time.time()
    limits = {"train": args.limit_train, "val": args.limit_val, "test": args.limit_test}
    raw_dir = Path(args.raw_dir).resolve() if args.raw_dir else None
    skip_existing = not args.no_skip_existing

    all_tasks, manifest_rows = [], {"train": [], "val": [], "test": []}
    for split in ("train", "val", "test"):
        input_jsonl = split_jsonl_path[split]
        if input_jsonl is None or not input_jsonl.exists():
            logger.warning(f"no jsonl for split '{split}' -- skipping")
            continue
        tasks, cached_rows = _build_tasks_for_split(
            split, input_jsonl, output_root, args.image_path, limits[split], skip_existing, raw_dir,
        )
        all_tasks.extend(tasks)
        manifest_rows[split].extend(cached_rows)
        logger.info(f"  [{split}] {len(cached_rows)} cached, {len(tasks)} to compute")

    if all_tasks:
        # Local-path resolution happens once here (PROJECT_ROOT anchors it
        # to this launch-directory-independent process), then passed
        # explicitly into every worker -- same swap grounded_sam_map_
        # calculations.py's own main() does for its single encoder.
        if args.local_files_only:
            dino_model = str(PROJECT_ROOT / args.dino_model_path)
            sam_model = str(PROJECT_ROOT / args.sam_model_path)
        else:
            dino_model = "IDEA-Research/grounding-dino-tiny"
            sam_model = "facebook/sam-vit-base"
        computed = run_multi_gpu(all_tasks, num_gpus, args.width, args.height, dino_model, sam_model,
                                  args.box_threshold, args.text_threshold, args.local_files_only)
        for split in ("train", "val", "test"):
            manifest_rows[split].extend(computed[split])
    else:
        logger.info("  Nothing to compute -- every referenced image was already cached.")

    split_counts = {}
    for split in ("train", "val", "test"):
        rows = manifest_rows[split]
        if not rows:
            continue
        out_jsonl = manifest_out / f"{split}.jsonl"
        with open(out_jsonl, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        logger.info(f"  Written {len(rows)} entries -> {out_jsonl}")
        _verify_grounded_sam_manifest(out_jsonl)
        split_counts[split] = len(rows)

    logger.info(f"\nDone. {time.time() - t_start:.1f}s elapsed. {split_counts}")
    logger.info(f"Log file: {_log_path}")


if __name__ == "__main__":
    main()
