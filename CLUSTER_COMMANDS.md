# Cluster commands — grounded_sam branch

Reference commands for running this branch on the cluster (1 node / 4 GPU /
12 CPU-per-task / 10h wall, per your sbatch header). Copy-paste, replace the
REPLACE-marked paths first. Every command here is verified against the real
CLI flags in this repo's scripts, not guessed.

## 0. Prerequisite check

`configs/experiment/train_grounded_sam_cluster.yaml` still has placeholder
paths (`base_model_path`, `vae_local_path`, `data.train_jsonl`,
`data.val_jsonl`) marked `# REPLACE`. Fill those in before any command below
will actually run.

## 1. Precompute — small slice first (verify the pipeline on real images)

```
python grounded_sam_map_calculations.py \
  --data_dir /path/to/Custom_dataset \
  --width 1280 --height 800 \
  --limit_train 30 --limit_val 10
```

`--output_root` is omitted deliberately — it now defaults to a SIBLING of
`--data_dir`, suffixed `_seg_map_grounded_sam` (e.g.
`.../Custom_dataset_seg_map_grounded_sam`), same convention as your own
`calc.py`. Pass `--output_root` explicitly only if you want somewhere else.

**Cost warning, stated honestly:** each image costs a Grounding-DINO
detection pass PLUS a SAM mask pass — this has only ever run on 20-75
images in this repo, so there is no real measured per-image time to
extrapolate from. Time THIS 30-image run before committing to the full
41,679.

## 2. Precompute — full dataset

```
python grounded_sam_map_calculations.py \
  --data_dir /path/to/Custom_dataset \
  --width 1280 --height 800
```

Skip-existing by default (`--no_skip_existing` forces recompute), so this
is safe to resume across separate jobs. **This is very likely more than one
10h allocation on its own** — from step 1's measured per-image time,
`41679 x seconds_per_image` tells you how many jobs to plan for. This is a
SEPARATE sbatch job from training, not something to run inside the 10h
training window.

## 3. Smoke test — small dataset, ~10 minutes, everything aligned and working

Trains on the 30-image slice from step 1 for a handful of steps with real
validation, to confirm config paths / data loading / checkpoint saving / the
4-GPU launch all work together — not a quality check, a correctness check.

```
srun accelerate launch \
  --multi_gpu --num_processes=4 --num_machines=1 --mixed_precision=bf16 \
  grounded_sam_training.py experiment=train_grounded_sam_cluster \
  data.train_jsonl=<output_root from step 1>/train.jsonl \
  data.val_jsonl=<output_root from step 1>/val.jsonl \
  max_train_steps=5 val_steps=5 val_batches=1 val_num_inference_steps=15 \
  tag=grounded_sam_cluster_check
```

**On the "~10 minutes" sizing — honest, not a guess presented as fact:**
`max_train_steps=5` is sized from the ONLY real number this repo has:
~100-130 s/step measured on a single 12GB dev GPU (no gradient_checkpointing
on the cluster, 4x parallel GPUs, and batch 8 vs 1 all change this in ways I
can't predict without your hardware). 5 steps + 1 validation (~15 sampling
steps) should land in single-digit minutes, but READ the actual `s/it` off
this run's own log — that's the real number, use it to size step 4 below,
don't trust the 5 above as calibrated.

Check
`outputs/train/grounded_sam_cluster_check/runs/.../val_grids/step000005.png`
afterward — if a PNG exists there and isn't corrupt, the full save/load/
validate pipeline is confirmed working end to end. The seg map panel should
render in the 28-class CARLA palette (purple road, blue cars, etc.) — if it's
mostly black, the taxonomy isn't loading right and needs looking at before
spending real GPU time.

## 4. Size the real run from the measured s/step

Once step 3 confirms everything works, read its actual `s/it` off the log
and size the real run from that:

```
steps_that_fit = floor(36000 seconds / measured_s_per_step)
```

Compare against `max_train_steps: 20000` in `train_grounded_sam_cluster.yaml`
— if `steps_that_fit` is lower, either lower `max_train_steps` to match, or
plan on the resume path below across multiple 10h jobs.

## 5. Full training launch

```
srun accelerate launch \
  --multi_gpu --num_processes=4 --num_machines=1 --mixed_precision=bf16 \
  grounded_sam_training.py experiment=train_grounded_sam_cluster
```

As an sbatch script:

```bash
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:4
#SBATCH --time=10:00:00

srun accelerate launch \
  --multi_gpu --num_processes=4 --num_machines=1 --mixed_precision=bf16 \
  grounded_sam_training.py experiment=train_grounded_sam_cluster
```

Checkpoints land at
`outputs/train/grounded_sam_cluster/runs/<date>/<time>/checkpoint-<N>/`.

## 6. Resume from the last checkpoint

Find the highest `checkpoint-<N>` folder under the previous run's output
dir, then:

```
srun accelerate launch \
  --multi_gpu --num_processes=4 --num_machines=1 --mixed_precision=bf16 \
  grounded_sam_training.py experiment=train_grounded_sam_cluster \
  lora.struct.ckpt_path=outputs/train/grounded_sam_cluster/runs/<date>/<time>/checkpoint-<N>
```

Point at the `checkpoint-<N>` folder itself, NOT `.../checkpoint-<N>/struct`
— `/struct` is appended automatically (verified: `src/utils.py:86`).

**Verified gotcha, not a guess:** the step counter does NOT carry over.
`global_step = 0` is set unconditionally regardless of `ckpt_path`
(`grounded_sam_training.py`), so the resumed run's own checkpoints restart
numbering from 1 in a NEW output folder — "checkpoint-500 in run 2" is
really cumulative step ~2000 if run 1 stopped at 1500. Track the real total
yourself; the tooling doesn't. Optimizer momentum also resets each resume
(only weights are saved, see `src/utils.py:save_checkpoint`) — harmless
here since `lr_scheduler: constant` means no schedule is lost, just a short
re-warm in gradient dynamics.
