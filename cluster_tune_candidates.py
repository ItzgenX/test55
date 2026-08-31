"""
cluster_tune_candidates.py
---------------------------
Writes ready-to-submit SBATCH SCRIPTS for short CALIBRATION runs -- Stage
1/2 of the tuning process (see CLUSTER_COMMANDS.md / project decision log):
a few hundred real steps per candidate, just enough to read the loss trend,
real s/step, and GPU memory headroom before committing the full 10h budget
to one configuration.

This script only WRITES .sh files to --out_dir -- it never submits or runs
anything itself. Run it anywhere (this dev box is fine, it's just a text
generator) to produce the files, then `sbatch` each one ON THE CLUSTER --
GPUs there are only reachable through a submitted sbatch job, not a direct
interactive command.

Each generated script uses the SAME #SBATCH header as this project's real
allocation (CLUSTER_COMMANDS.md / the cluster config files' own header
comment: 1 node, 4 tasks, 12 CPUs/task, 4 GPUs) -- only --time is shorter
by default, since a 300-step calibration run doesn't need the full 10h.

One variable varies per stage -- everything else pinned at the real
cluster config's own default (configs/experiment/train_seg_cluster.yaml /
train_grounded_sam_cluster.yaml) -- so a difference in the result is
actually attributable to the thing being varied, not confounded by two
knobs moving at once.

Candidate values are not invented here: LR bracket matches the cluster
config's own inline comment ("back off toward 1e-4 ... or push toward
3e-4"); batch/rank brackets are a half/default/double spread around each
branch's real current default.

Usage:
    python cluster_tune_candidates.py --branch segformer --stage lr
    python cluster_tune_candidates.py --branch grounded_sam --stage batch
    python cluster_tune_candidates.py --branch segformer --stage rank
    python cluster_tune_candidates.py --branch grounded_sam --stage all --out_dir cluster_jobs --time 01:00:00

Then, ON THE CLUSTER (never here -- this script only writes the files):
    sbatch cluster_jobs/calib_lr_1e-4.sh
    sbatch cluster_jobs/calib_lr_2e-4.sh
    ...
"""

import argparse
from pathlib import Path

# Real values from configs/experiment/{train_seg,train_grounded_sam}_cluster.yaml,
# confirmed by reading the actual files, not assumed.
BRANCH = {
    "segformer": {
        "script": "segformer_training.py",
        "experiment": "train_seg_cluster",
        "default_lr": "2e-4",
        "default_batch": 8,
        "default_rank": 128,
    },
    "grounded_sam": {
        "script": "grounded_sam_training.py",
        "experiment": "train_grounded_sam_cluster",
        "default_lr": "2e-4",
        "default_batch": 8,
        "default_rank": 128,
    },
}

# Real allocation shape, matches CLUSTER_COMMANDS.md / the cluster config
# files' own header comment. --time is the one field this script shortens
# for calibration jobs (see --time CLI arg) -- everything else is the same
# resource shape a full training job would request.
SBATCH_HEADER = """#!/bin/bash
#SBATCH --job-name={tag}
#SBATCH --output={tag}_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:4
#SBATCH --time={time}
"""

# Short enough to be a real signal (loss trend, s/step, one validation grid
# to eyeball) without spending meaningful wall-clock -- NOT a full run.
CALIB_STEPS = 300
CALIB_VAL_STEPS = 150

LR_CANDIDATES = ["1e-4", "2e-4", "3e-4"]          # matches the cluster config's own documented bracket
BATCH_CANDIDATES = [8, 16, 24]                     # default, 2x, 3x -- actual ceiling depends on real nvidia-smi headroom
RANK_CANDIDATES = [64, 128, 256]                   # half, default, double


def _write_job(branch: str, tag: str, overrides: list[str], out_dir: Path, time: str) -> Path:
    b = BRANCH[branch]
    override_str = " \\\n    ".join(overrides)
    srun_line = (
        f"srun accelerate launch \\\n"
        f"  --multi_gpu --num_processes=4 --num_machines=1 --mixed_precision=bf16 \\\n"
        f"  {b['script']} experiment={b['experiment']} \\\n"
        f"    {override_str} \\\n"
        f"    max_train_steps={CALIB_STEPS} val_steps={CALIB_VAL_STEPS} ckpt_steps={CALIB_VAL_STEPS} \\\n"
        f"    tag={tag}\n"
    )
    script = SBATCH_HEADER.format(tag=tag, time=time) + "\n" + srun_line
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{tag}.sh"
    path.write_text(script, encoding="utf-8", newline="\n")
    return path


def stage_lr(branch: str, out_dir: Path, time: str) -> None:
    print(f"\n{'='*70}\n  STAGE: learning rate  ({branch})\n{'='*70}")
    print("Fixed: batch_size (default), rank (default). Varying: learning_rate.")
    print(f"WATCH: loss trend over these {CALIB_STEPS} steps -- NaN/spiking = too high,")
    print("flat-and-stable-but-not-moving = maybe too low (but also just noisy at this")
    print("step count -- don't over-read a flat 300-step trace). Compare the three")
    print("tensorboard loss curves side by side, not one run in isolation.\n")
    for lr in LR_CANDIDATES:
        default_marker = "  (cluster config's current default)" if lr == BRANCH[branch]["default_lr"] else ""
        tag = f"calib_lr_{lr}"
        path = _write_job(branch, tag, [f"learning_rate={lr}"], out_dir, time)
        print(f"  {lr}{default_marker}  ->  sbatch {path.as_posix()}")


def stage_batch(branch: str, out_dir: Path, time: str) -> None:
    default_batch = BRANCH[branch]["default_batch"]
    print(f"\n{'='*70}\n  STAGE: batch size  ({branch})\n{'='*70}")
    print("Fixed: learning_rate (default), rank (default). Varying: data.batch_size.")
    print("WATCH: nvidia-smi memory headroom AND real s/step for each -- a bigger batch")
    print("that OOMs or barely fits isn't a real option. If a larger batch is stable,")
    print("its effective batch changed (batch x accum x 4 GPUs) -- the LR that was")
    print("tuned for the default effective batch may no longer be right; re-check LR")
    print("at the new batch size before trusting a direct comparison.\n")
    for batch in BATCH_CANDIDATES:
        default_marker = "  (cluster config's current default)" if batch == default_batch else ""
        tag = f"calib_batch_{batch}"
        path = _write_job(branch, tag, [f"data.batch_size={batch}"], out_dir, time)
        print(f"  {batch}{default_marker}  ->  sbatch {path.as_posix()}")


def stage_rank(branch: str, out_dir: Path, time: str) -> None:
    default_rank = BRANCH[branch]["default_rank"]
    print(f"\n{'='*70}\n  STAGE: LoRA rank  ({branch})\n{'='*70}")
    print("LAST resort, not first -- only worth running once a genuinely long run at")
    print(f"the default rank ({default_rank}) has already shown mIoU plateauing somewhere")
    print("unsatisfying. Rank changes model capacity, not stability -- a 300-step")
    print("calibration run tells you almost nothing about capacity (mIoU is near-random")
    print("at this step count regardless of rank, per every real run in this repo so")
    print("far). Running this stage is mostly useful for confirming the run doesn't")
    print("crash/OOM at a different rank, NOT for judging quality yet.\n")
    for rank in RANK_CANDIDATES:
        default_marker = "  (cluster config's current default)" if rank == default_rank else ""
        tag = f"calib_rank_{rank}"
        path = _write_job(branch, tag, [f"lora.struct.config.rank={rank}"], out_dir, time)
        print(f"  {rank}{default_marker}  ->  sbatch {path.as_posix()}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--branch", required=True, choices=list(BRANCH.keys()))
    p.add_argument("--stage", required=True, choices=["lr", "batch", "rank", "all"])
    p.add_argument("--out_dir", default="cluster_jobs", help="Where to write the .sh files. Default: cluster_jobs/")
    p.add_argument("--time", default="01:00:00",
                    help="SBATCH --time for each calibration job (D-HH:MM:SS or HH:MM:SS). "
                         "Default 1h -- generous for 300 steps + one validation at unmeasured "
                         "cluster speed; tighten once you have a real s/step number.")
    args = p.parse_args()
    out_dir = Path(args.out_dir)

    stages = {"lr": stage_lr, "batch": stage_batch, "rank": stage_rank}
    if args.stage == "all":
        for fn in stages.values():
            fn(args.branch, out_dir, args.time)
    else:
        stages[args.stage](args.branch, out_dir, args.time)

    print(f"\n{'='*70}")
    print(f"Written to {out_dir}/ -- this script only WROTE these files, it did not")
    print("submit or run anything. On the CLUSTER, GPUs are only reachable through a")
    print("submitted job -- submit each with:")
    print(f"    sbatch {out_dir}/<name>.sh")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
