"""
grounded_sam_map_calculations_multi_GPU.py
-------------------------------------------
Normal python script -- run it directly (`python
grounded_sam_map_calculations_multi_GPU.py ...`), no separate launcher
command needed. Uses every visible GPU by default.

WHAT IT ACTUALLY DOES (deliberately simple, not clever): launches N ordinary
OS subprocesses of the existing, already-working
grounded_sam_map_calculations.py -- one per GPU -- each with its own
--rank/--world_size/--device already filled in for you. You never type rank
or world_size yourself; this wrapper does it internally. Once every
subprocess finishes, it merges their per-GPU manifests
(train_rank0.jsonl + train_rank1.jsonl + ...) into the normal unified
train.jsonl/val.jsonl/test.jsonl -- no manual `cat` step needed either.

No new compute logic here at all -- image loading, encoding, PNG saving,
manifest-row building, verification are 100% grounded_sam_map_calculations.py's
own code, unchanged, run once per GPU. This file is just a launcher + a
merge step.

NOT TESTED ON HARDWARE this was written for: the author's dev machine has
one GPU. What IS verified for real (see below): running this with
--num_gpus 1 goes through the exact same subprocess-launch + wait + merge
code path as a real N-GPU run, just spawning one subprocess instead of N --
it does not exercise true parallelism, but it does exercise every other
moving part.

USAGE (identical to grounded_sam_map_calculations.py, plus --num_gpus --
every other flag passes straight through unchanged, see that script's own
--help for the full list):
    python grounded_sam_map_calculations_multi_GPU.py \\
        --data_dir data/custom_dataset --width 1280 --height 800 --num_gpus 4

    # Uses every visible GPU (torch.cuda.device_count()) if --num_gpus is
    # omitted. Pass --num_gpus 1 to sanity-check this wrapper on a single
    # GPU before trusting it on the real multi-GPU box.
"""

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
WORKER_SCRIPT = PROJECT_ROOT / "grounded_sam_map_calculations.py"


def _resolve_manifest_out(passthrough: list[str]) -> Path:
    """Same output_root/manifest_out resolution grounded_sam_map_calculations.py's
    own main() uses, just enough of it to know where the per-rank manifests
    will land so we can merge them afterward."""
    peek = argparse.ArgumentParser(add_help=False)
    peek.add_argument("--data_dir", default=None)
    peek.add_argument("--output_root", default=None)
    peek.add_argument("--manifest_out", default=None)
    peeked, _ = peek.parse_known_args(passthrough)

    if peeked.output_root:
        output_root = Path(peeked.output_root).resolve()
    elif peeked.data_dir:
        data_dir = Path(peeked.data_dir)
        output_root = (data_dir.parent / (data_dir.name + "_seg_map_grounded_sam")).resolve()
    else:
        raise SystemExit(
            "[FATAL] Need --data_dir or --output_root, same requirement as "
            "grounded_sam_map_calculations.py itself."
        )
    return Path(peeked.manifest_out).resolve() if peeked.manifest_out else output_root


def _merge_rank_manifests(manifest_out: Path, num_gpus: int) -> None:
    print("\n[merge] combining per-GPU manifests into unified train/val/test.jsonl")
    for split in ("train", "val", "test"):
        rank_files = [manifest_out / f"{split}_rank{r}.jsonl" for r in range(num_gpus)]
        rank_files = [f for f in rank_files if f.exists()]
        if not rank_files:
            continue
        merged_path = manifest_out / f"{split}.jsonl"
        n = 0
        with open(merged_path, "w", encoding="utf-8") as out:
            for rf in rank_files:
                with open(rf, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            out.write(line)
                            n += 1
        print(f"  [{split}] merged {len(rank_files)} rank file(s) -> {n} entries -> {merged_path}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--num_gpus", type=int, default=None,
                    help="Number of GPUs to use, one subprocess each. Default: every visible "
                         "GPU (torch.cuda.device_count()). Every other flag passes straight "
                         "through to grounded_sam_map_calculations.py -- see that script's --help.")
    args, passthrough = p.parse_known_args()

    import torch
    if not torch.cuda.is_available():
        p.error("No CUDA GPU visible -- this script is multi-GPU only. Use "
                 "grounded_sam_map_calculations.py --device cpu for a CPU run.")
    n_visible = torch.cuda.device_count()
    num_gpus = args.num_gpus if args.num_gpus is not None else n_visible
    if num_gpus < 1:
        p.error("--num_gpus must be >= 1.")
    if num_gpus > n_visible:
        p.error(f"--num_gpus {num_gpus} exceeds the {n_visible} GPU(s) actually visible to this "
                 f"process (torch.cuda.device_count()). Check CUDA_VISIBLE_DEVICES if unexpected.")

    manifest_out = _resolve_manifest_out(passthrough)

    print("=" * 60)
    print(f"  Grounded-SAM map calculation -- {num_gpus} GPU(s), {num_gpus} subprocess(es)")
    print(f"  Each subprocess: {WORKER_SCRIPT.name} --rank <i> --world_size {num_gpus} --device cuda:<i>")
    print("=" * 60)

    procs = []
    for rank in range(num_gpus):
        cmd = [
            sys.executable, str(WORKER_SCRIPT), *passthrough,
            "--rank", str(rank), "--world_size", str(num_gpus), "--device", f"cuda:{rank}",
        ]
        print(f"[launch] GPU {rank}: {' '.join(cmd)}")
        procs.append(subprocess.Popen(cmd))

    exit_codes = [proc.wait() for proc in procs]
    failed = [rank for rank, code in enumerate(exit_codes) if code != 0]
    if failed:
        raise SystemExit(
            f"[FATAL] GPU worker(s) {failed} exited with a non-zero code -- check their "
            f"rank-suffixed log files under outputs/logs/ for the real error. Not merging "
            f"manifests: a failed rank's partial/missing output would silently under-count."
        )

    _merge_rank_manifests(manifest_out, num_gpus)
    print("\nDone.")


if __name__ == "__main__":
    main()
