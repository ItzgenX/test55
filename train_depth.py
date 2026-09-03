"""
train_depth.py
---------------
Train an SD1.5 depth-conditioned LoRA from precomputed depth maps
(compute_depth.py's cache). DEPTH-BRANCH-architecture.md RULE 0 applies:
derives from the original repo + paper only -- SD1.5, matching the
original repo's own released depth checkpoint
(checkpoints/sd15-depth-128-only-res), not SDXL (an earlier variant of
this branch used SDXL; switched back to SD1.5, see train_depth.yaml's own
comment for the full reasoning). Reuses train.py's loop shape (hydra
composition, accelerate, AdamW, the same checkpoint/validation cadence)
because that's this repo's own generic training loop, not anything
segmentation-specific -- and follows the same monitoring conventions
(fixed vs fresh validation scenes, a best_model/ folder, one grid image
per checkpoint) used elsewhere in this project for readability parity
across branches.

Differences from train.py:
  1. Data comes from DepthJsonDataset -- "jpg" (RGB, VAE target) and "depth"
     (the precomputed depth map, expanded to 3ch at load time), not a live
     encoder run on the image.
  2. Every model call passes skip_encode=True, so model.forward_easy() feeds
     batch["depth"] straight to FixedStructureMapper15 instead of running a
     live MiDaS pass (src/model.py's skip_encode branch, fixed by this
     branch -- defect #2 in the architecture doc).
  3. SD1.5 has no SDXL-style time_ids/micro-conditioning mechanism at all --
     model.forward_easy() takes no such argument, unlike SDXL.forward().
  4. Dropout zeroes the cached depth tensor directly (src/model.py's
     `c[dropout_mask] = 0`, unmodified) -- never re-runs an annotator.
  5. GATE 0 (DEPTH-BRANCH-architecture.md §8) runs at startup and hard-fails
     before any training step if any of the five checks fail.

MONITORING -- outputs/train/depth/runs/<date>/<time>/:
  training_params.txt          resolved config for this run
  best_model/                  lowest val/loss so far (overwritten)
    struct/{lora,mapper}-checkpoint.pt
    sample_00.jpg               ORIGINAL | DEPTH | PREDICTED, one row per scene
    prompts.txt
    info.txt                    step/epoch/loss when this became best
  checkpoint-<step>/
    struct/{lora,mapper}-checkpoint.pt
    sample_00.jpg ...
    prompts.txt
  logs/tensorboard/

  n_grid_images val scenes per checkpoint, split roughly 50/50:
    FIXED half -- drawn once at startup, reused at every checkpoint, so you
                  watch the same scenes improve.
    FRESH half -- redrawn each checkpoint, a generalization peek.
"""

import json
import math
import os
import random
import signal
import sys
import traceback
from pathlib import Path

import einops
import hydra
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from accelerate import Accelerator
from accelerate.logging import get_logger
from diffusers.optimization import get_scheduler
from omegaconf import OmegaConf
from tqdm.auto import tqdm

from src.model import SD15
from src.utils import DataProvider, add_lora_from_config, save_checkpoint

torch.set_float32_matmul_precision("high")

# GATE 0 check [2] expected ranges -- verified against a real run of THIS
# exact configuration (SD1.5 + FixedStructureMapper15 + rank=64,
# only_res_conv, train_depth.yaml): real lora_params=14,999,552,
# mapper_params=1,245,072, matched_layers=18 (SD1.5's UNet has 4 down_blocks
# x 2 resnets + 4 up_blocks x 2 resnets + mid_block x 2 resnets = 18 conv1
# layers -- more than SDXL's 14, since SD1.5's UNet has an extra down/up
# stage SDXL's reduced-stage design doesn't). Re-verify with a real run if
# rank, adaption_mode, or the base model ever change.
LORA_PARAMS_MIN, LORA_PARAMS_MAX = 14_500_000, 15_500_000
MAPPER_PARAMS_MIN, MAPPER_PARAMS_MAX = 1_150_000, 1_350_000
EXPECTED_MATCHED_LAYERS = 18

stop_training = False


def signal_handler(sig, frame):
    global stop_training
    stop_training = True
    print("got stop signal")


# =============================================================================
# GATE 0 -- architecture verification, before any training step (§8)
# =============================================================================


def _matched_only_res_conv_layers(pre_wiring_state_dict_keys) -> list[str]:
    """Read-only re-derivation of which module paths add_lora_to_unet's
    only_res_conv filter matches, for the GATE 0 report. Mirrors the filter
    condition in src.model.ModelBase.add_lora_to_unet exactly. MUST be run
    against the unet's state_dict keys from BEFORE add_lora_to_unet wired
    anything in -- after wiring, each matched conv is a NewStructLoRAConv
    with 5 sub-tensors (W/A/B/beta/gamma), not the original single conv, so
    counting post-wiring keys overcounts 5x."""
    matched = []
    for path in pre_wiring_state_dict_keys:
        if "bias" in path:
            continue
        if "0.conv1" in path or "1.conv1" in path:
            target_path = ".".join(path.split(".")[:-1])
            if target_path not in matched:
                matched.append(target_path)
    return matched


def compute_frozen_reference(model: SD15, device, seed: int):
    """Must be called BEFORE add_lora_to_unet wires anything in: with
    model.mappers/encoders/dps all empty, model.forward_easy()'s per-condition
    loop runs zero times, so this is a plain, unmodified unet forward pass.
    Returns (frozen_pred, imgs, prompts) -- imgs/prompts are reused for the
    adapted pass afterwards so both runs see bit-identical VAE-encoded
    latents, noise and timesteps given the same seed."""
    B = 2
    imgs = torch.rand(B, 3, 768, 1280, device=device, dtype=torch.float32) * 2 - 1
    prompts = ["a photo of a road" for _ in range(B)]
    torch.manual_seed(seed)
    random.seed(seed)
    with torch.no_grad():
        frozen_pred, _, _, _ = model.forward_easy(imgs, prompts, cs=[], cfg_mask=[], skip_encode=True)
    return frozen_pred, imgs, prompts


def run_gate_0(model: SD15, mapper, matched_layers, frozen_pred, ref_imgs, ref_prompts, device, logger, seed: int = 0) -> bool:
    logger.info("=" * 70)
    logger.info("GATE 0 -- architecture verification")
    logger.info("=" * 70)
    all_pass = True

    # --- 1. shape check --------------------------------------------------
    # FixedStructureMapper15 returns 4 outputs (out0..out3), matching SD1.5's
    # 4-stage UNet -- one more than the SDXL-era DepthStructureMapperXL's 3,
    # since SD1.5's UNet has an extra down/up stage SDXL's reduced-stage
    # design doesn't. Expected shapes verified by a real run, not derived by
    # hand (see the class's own down/block1/block2/block3 stride-2 stages).
    c = torch.rand(2, 3, 768, 1280, device=device, dtype=torch.float32)
    with torch.no_grad():
        out0, out1, out2, out3 = mapper(c)
    shapes = {
        "out0": (tuple(out0.shape[-2:]), (96, 160)),
        "out1": (tuple(out1.shape[-2:]), (48, 80)),
        "out2": (tuple(out2.shape[-2:]), (24, 40)),
        "out3": (tuple(out3.shape[-2:]), (12, 20)),
    }
    shape_ok = all(got == exp for got, exp in shapes.values())
    logger.info(f"[1] shape check: {shapes} -> {'PASS' if shape_ok else 'FAIL'}")
    all_pass &= shape_ok

    # --- 2. param count ----------------------------------------------------
    lora_params = sum(p.numel() for p in model.params_to_optimize)
    mapper_params = sum(p.numel() for p in mapper.parameters())
    # Ranges verified against a real run of THIS SD1.5 + FixedStructureMapper15
    # + rank=64 configuration (train_depth.yaml) -- not the SDXL-era
    # DepthStructureMapperXL numbers this file used before the SD1.5 switch.
    lora_ok = LORA_PARAMS_MIN < lora_params < LORA_PARAMS_MAX
    mapper_ok = MAPPER_PARAMS_MIN < mapper_params < MAPPER_PARAMS_MAX
    layers_ok = len(matched_layers) == EXPECTED_MATCHED_LAYERS
    logger.info(f"[2] only_res_conv matched {len(matched_layers)} layers (expect {EXPECTED_MATCHED_LAYERS}): {matched_layers}")
    logger.info(f"[2] lora_params={lora_params:,} (expect {LORA_PARAMS_MIN:,}-{LORA_PARAMS_MAX:,}) -> {'PASS' if lora_ok else 'FAIL'}")
    logger.info(f"[2] mapper_params={mapper_params:,} (expect {MAPPER_PARAMS_MIN:,}-{MAPPER_PARAMS_MAX:,}) -> {'PASS' if mapper_ok else 'FAIL'}")
    logger.info(f"[2] layer count -> {'PASS' if layers_ok else 'FAIL'}")
    all_pass &= lora_ok and mapper_ok and layers_ok

    # --- shared fixed inputs for checks 3 & 4 ------------------------------
    # Reuse the SAME imgs/prompts as the frozen reference pass (see
    # compute_frozen_reference) so VAE-encoded latents/noise/timesteps are
    # bit-identical given the same seed -- only the depth condition differs.
    imgs, prompts = ref_imgs, ref_prompts
    depth = torch.rand(imgs.shape[0], 3, 768, 1280, device=device, dtype=torch.float32)

    def seeded_forward(cs, cfg_mask, skip_encode):
        torch.manual_seed(seed)
        random.seed(seed)
        with torch.no_grad():
            model_pred, _, _, _ = model.forward_easy(
                imgs, prompts, cs=cs, cfg_mask=cfg_mask, skip_encode=skip_encode
            )
        return model_pred

    # --- 3. zero-init no-op --------------------------------------------
    adapted_pred_scale1 = seeded_forward(cs=[depth], cfg_mask=[True], skip_encode=True)
    zero_init_ok = torch.allclose(frozen_pred, adapted_pred_scale1, atol=1e-5)
    max_diff = (frozen_pred - adapted_pred_scale1).abs().max().item()
    logger.info(f"[3] zero-init no-op: max|frozen-adapted|={max_diff:.3e} -> {'PASS' if zero_init_ok else 'FAIL'}")
    all_pass &= zero_init_ok

    # --- 4. lambda=0 invariant -------------------------------------------
    original_scales = []
    for layer in model.lora_layers["struct"]:
        original_scales.append(layer.lora_scale)
        layer.lora_scale = 0.0
    adapted_pred_scale0 = seeded_forward(cs=[depth], cfg_mask=[True], skip_encode=True)
    for layer, s in zip(model.lora_layers["struct"], original_scales):
        layer.lora_scale = s
    lambda0_ok = torch.allclose(frozen_pred, adapted_pred_scale0, atol=1e-5)
    max_diff0 = (frozen_pred - adapted_pred_scale0).abs().max().item()
    logger.info(f"[4] lambda=0 invariant: max|frozen-adapted|={max_diff0:.3e} -> {'PASS' if lambda0_ok else 'FAIL'}")
    all_pass &= lambda0_ok

    logger.info("=" * 70)
    logger.info(f"GATE 0 checks 1-4: {'ALL PASS' if all_pass else 'FAILURE -- see above'}")
    logger.info("=" * 70)

    return all_pass


def run_gate_0_overfit(model: SD15, train_dataset, device, logger, n_samples=8, n_steps=300, lr=1e-3) -> bool:
    logger.info("=" * 70)
    logger.info(f"GATE 0 [5] tiny-batch overfit -- {n_samples} samples, {n_steps} steps")
    logger.info("=" * 70)

    idxs = list(range(min(n_samples, len(train_dataset))))
    if len(idxs) < n_samples:
        logger.info(f"[5] FAIL: dataset only has {len(idxs)} samples, need {n_samples}")
        return False

    batch = [train_dataset[i] for i in idxs]
    imgs = torch.stack([b["jpg"] for b in batch]).to(device)
    depth = torch.stack([b["depth"] for b in batch]).to(device)
    prompts = [b["caption"] for b in batch]

    optimizer = torch.optim.AdamW(model.params_to_optimize + list(model.mappers[0].parameters()), lr=lr)

    losses = []
    model.unet.train()
    model.mappers[0].train()
    for step in range(n_steps):
        # This probe runs in plain fp32 (no accelerate/bf16 autocast, no
        # gradient accumulation) -- on a memory-constrained dev GPU, PyTorch's
        # caching allocator can fragment across repeated same-shape
        # alloc/free cycles even with gradient checkpointing on, OOMing on a
        # later step despite step 0 succeeding (observed on a 12GB card at
        # 8x1280x768). Real cluster GPUs (95GB) won't need this; harmless
        # and cheap to keep since this loop is small (n_steps default 300).
        if device != "cpu" and torch.cuda.is_available():
            torch.cuda.empty_cache()
        optimizer.zero_grad()
        _, loss, _, _ = model.forward_easy(imgs, prompts, cs=[depth.clone()], cfg_mask=[True], skip_encode=True)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if step % 50 == 0 or step == n_steps - 1:
            logger.info(f"[5] step {step}: loss={loss.item():.5f}")

    collapsed = losses[-1] < 0.3 * losses[0]
    logger.info(f"[5] loss[0]={losses[0]:.5f} -> loss[-1]={losses[-1]:.5f} -> {'PASS' if collapsed else 'FAIL'}")
    logger.info("=" * 70)
    return collapsed


# =============================================================================
# Grid generation
# =============================================================================


def make_grid_row(original: torch.Tensor, depth: torch.Tensor, predicted) -> np.ndarray:
    """original, depth: [3,H,W] in [-1,1]/[0,1] resp. predicted: PIL.Image.
    Returns an HWC uint8 numpy array: ORIGINAL | DEPTH | PREDICTED."""
    orig_01 = ((original.cpu() + 1) / 2).clamp(0, 1)
    depth_01 = depth.cpu().clamp(0, 1)
    pred_t = TF.to_tensor(predicted.resize((orig_01.shape[-1], orig_01.shape[-2])))
    row = torch.cat([orig_01, depth_01, pred_t], dim=2)  # concat along width
    return (row.permute(1, 2, 0).numpy() * 255).astype(np.uint8)


def save_sample_grid(rows: list[np.ndarray], prompts: list[str], out_dir: Path, name: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    grid = np.concatenate(rows, axis=0)
    from PIL import Image

    Image.fromarray(grid).save(out_dir / f"{name}.jpg", quality=92)
    with open(out_dir / "prompts.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(prompts))


@hydra.main(config_path="configs", config_name="train")
def main(cfg):
    signal.signal(signal.SIGINT, signal_handler)

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    output_path = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)

    accelerator = Accelerator(
        project_dir=output_path / "logs",
        log_with="tensorboard",
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        mixed_precision="bf16",
    )
    logger = get_logger(__name__)
    logger.info(f"output dir: {output_path}")

    with open(output_path / "training_params.txt", "w", encoding="utf-8") as f:
        f.write(str(cfg))

    # hydra.job.chdir=true (configs/train.yaml) moves the process cwd to the
    # run's output dir before this runs -- a relative data.train_jsonl/
    # val_jsonl (the config default: data/dataset/depth_cache/...) then
    # resolves against THAT dir, not the repo root, and DepthJsonDataModule
    # fails with FileNotFoundError. Same bug, same fix, as
    # grounded_sam_training.py/segformer_training.py.
    _root = hydra.utils.get_original_cwd()
    if cfg.data.get("train_jsonl") and not os.path.isabs(cfg.data.train_jsonl):
        cfg.data.train_jsonl = os.path.join(_root, cfg.data.train_jsonl)
    if cfg.data.get("val_jsonl") and not os.path.isabs(cfg.data.val_jsonl):
        cfg.data.val_jsonl = os.path.join(_root, cfg.data.val_jsonl)

    # Instantiate everything EXCEPT `data` first -- GATE 0 checks 1-4 need
    # only the model + mapper, and must be able to run before compute_depth.py
    # has ever been run (dataset not existing yet is not a reason to skip them).
    data_cfg = cfg.data
    cfg_no_data = OmegaConf.create({k: v for k, v in cfg.items() if k != "data"})
    resolved_cfg = hydra.utils.instantiate(cfg_no_data)
    model: SD15 = resolved_cfg.model
    model = model.to(accelerator.device)
    model.pipe.to(accelerator.device, torch.float32)

    # MUST run before add_lora_to_unet: with model.mappers/encoders/dps still
    # empty, model.forward_easy()'s per-condition loop runs zero times, giving
    # a plain unmodified unet forward pass to compare the adapted one against.
    frozen_pred, ref_imgs, ref_prompts = compute_frozen_reference(model, accelerator.device, cfg.seed)

    pre_wiring_keys = list(model.unet.state_dict().keys())
    cfg_mask = add_lora_from_config(model, resolved_cfg, accelerator.device)
    mapper = model.mappers[0]
    # add_lora_to_unet creates each replacement LoRA module fresh (on CPU by
    # default) and setattr's it in -- model.to(device) above ran BEFORE these
    # existed, so without this they silently stay on CPU until accelerate's
    # own .prepare() would eventually move them (too late for GATE 0, which
    # must run in plain fp32 before that).
    model.unet.to(accelerator.device)
    matched_layers = _matched_only_res_conv_layers(pre_wiring_keys)

    if cfg.get("gradient_checkpointing", False):
        model.unet.enable_gradient_checkpointing()

    # ---- GATE 0, checks 1-4 (synthetic, no data needed) --------------------
    gate_1_4_pass = run_gate_0(
        model, mapper, matched_layers, frozen_pred, ref_imgs, ref_prompts, accelerator.device, logger, seed=cfg.seed
    )
    if not gate_1_4_pass:
        logger.info("GATE 0 checks 1-4 FAILED. Not proceeding to training.")
        sys.exit(1)

    try:
        dm = hydra.utils.instantiate(data_cfg)
    except Exception as e:
        # hydra.utils.instantiate wraps the DepthJsonDataset's own
        # FileNotFoundError in its own InstantiationException -- a plain
        # `except FileNotFoundError` here never actually catches it, so
        # catch broadly and surface the real cause explicitly.
        cause = e.__cause__ or e
        logger.info("=" * 70)
        logger.info(f"GATE 0 [5] cannot run -- dataset not ready: {cause}")
        logger.info("Run compute_depth.py first. GATE 0 checks 1-4 above already PASSED.")
        logger.info("=" * 70)
        sys.exit(1)
    train_dataloader = dm.train_dataloader()
    val_dataloader = dm.val_dataloader()

    # ---- GATE 0, check 5 (tiny-batch overfit, needs real cached data) ------
    # Snapshot trainable state first: the overfit probe runs 300 real gradient
    # steps on model.params_to_optimize + the mapper, and we want the actual
    # training run below to start from true zero-init, not from that probe.
    import copy

    lora_state_before = model.get_lora_state_dict()
    mapper_state_before = copy.deepcopy(mapper.state_dict())

    gate_5_pass = run_gate_0_overfit(model, dm.train_dataset, accelerator.device, logger)
    if not gate_5_pass:
        logger.info("GATE 0 check [5] FAILED. Not proceeding to training.")
        sys.exit(1)

    model.unet.load_state_dict(lora_state_before["struct"], strict=False)
    mapper.load_state_dict(mapper_state_before)
    logger.info("GATE 0 [5] PASS -- restored zero-init state before starting the real run.")
    logger.info("GATE 0: ALL FIVE CHECKS PASS. Proceeding to training.")

    # =========================================================================
    # Training loop
    # =========================================================================
    mappers_params = list(filter(lambda p: p.requires_grad, mapper.parameters()))
    optimizer = torch.optim.AdamW(model.params_to_optimize + mappers_params, lr=cfg.learning_rate)
    lr_scheduler = get_scheduler(cfg.lr_scheduler, optimizer=optimizer)

    logger.info(f"trainable LoRA params: {sum(p.numel() for p in model.params_to_optimize):,}")
    logger.info(f"trainable mapper params: {sum(p.numel() for p in mappers_params):,}")

    if accelerator.is_main_process:
        accelerator.init_trackers("tensorboard")

    unet, mapper, optimizer, train_dataloader, val_dataloader, lr_scheduler = accelerator.prepare(
        model.unet, mapper, optimizer, train_dataloader, val_dataloader, lr_scheduler
    )
    model.unet = unet
    model.mappers = [mapper]

    # fixed validation scenes: drawn once, reused at every checkpoint
    val_dataset = dm.val_dataset
    n_grid = cfg.get("n_grid_images", 4)
    n_fixed = max(1, n_grid // 2)
    n_fresh = max(1, n_grid - n_fixed)
    fixed_idxs = random.Random(cfg.seed).sample(range(len(val_dataset)), min(n_fixed, len(val_dataset)))

    def build_grid(idxs, generator):
        rows, prompts = [], []
        for idx in idxs:
            sample = val_dataset[idx]
            img_b = sample["jpg"].unsqueeze(0).to(accelerator.device)
            depth_b = sample["depth"].unsqueeze(0).to(accelerator.device)
            prompt = sample["caption"]
            preds = model.sample(
                prompt=[prompt],
                num_images_per_prompt=1,
                cs=[depth_b],
                generator=generator,
                cfg_mask=[True],
                skip_encode=True,
                num_inference_steps=30,
            )
            rows.append(make_grid_row(sample["jpg"], sample["depth"], preds[0]))
            prompts.append(prompt)
        return rows, prompts

    def save_checkpoint_and_grids(step: int, is_best: bool):
        generator = torch.Generator(device=accelerator.device).manual_seed(cfg.seed)
        fresh_idxs = random.sample(range(len(val_dataset)), min(n_fresh, len(val_dataset)))
        rows, prompts = build_grid(fixed_idxs, generator)
        fresh_rows, fresh_prompts = build_grid(fresh_idxs, generator)

        for tag, dest in [("checkpoint", output_path / f"checkpoint-{step}")]:
            save_checkpoint(
                model.get_lora_state_dict(accelerator.unwrap_model(unet)),
                [accelerator.unwrap_model(mapper).state_dict()],
                None,
                dest,
            )
            save_sample_grid(rows + fresh_rows, prompts + fresh_prompts, dest, "sample_00")

        if is_best:
            best_dir = output_path / "best_model"
            save_checkpoint(
                model.get_lora_state_dict(accelerator.unwrap_model(unet)),
                [accelerator.unwrap_model(mapper).state_dict()],
                None,
                best_dir,
            )
            save_sample_grid(rows + fresh_rows, prompts + fresh_prompts, best_dir, "sample_00")
            with open(best_dir / "info.txt", "w", encoding="utf-8") as f:
                f.write(f"step={step}\n")

        for tracker in accelerator.trackers:
            if tracker.name == "tensorboard":
                np_grid = np.concatenate(rows + fresh_rows, axis=0)
                tracker.writer.add_image("validation", np_grid, step, dataformats="HWC")

    global_step = 0
    best_val_loss = float("inf")
    logger.info("start training")

    try:
        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / cfg.gradient_accumulation_steps)
        max_train_steps = cfg.get("max_train_steps", None) or cfg.epochs * num_update_steps_per_epoch
    except Exception:
        max_train_steps = 10_000_000

    progress_bar = tqdm(range(max_train_steps), disable=not accelerator.is_main_process)
    progress_bar.set_description("Steps")

    for epoch in range(cfg.epochs):
        unet.train()
        mapper.train()

        for batch in train_dataloader:
            with accelerator.accumulate(unet, mapper):
                imgs = batch["jpg"].to(accelerator.device).clip(-1.0, 1.0)
                depth = batch["depth"].to(accelerator.device)
                prompts = batch["caption"]

                _, loss, _, _ = model.forward_easy(
                    imgs, prompts, cs=[depth], cfg_mask=[True], skip_encode=True, batch=batch
                )

                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}, step=global_step)
                progress_bar.set_postfix(loss=loss.detach().item())

                if global_step % cfg.val_steps == 0 or stop_training:
                    unet.eval()
                    mapper.eval()
                    val_losses = []
                    with torch.no_grad():
                        for i, val_batch in enumerate(val_dataloader):
                            if i >= cfg.get("val_batches", 4):
                                break
                            v_imgs = val_batch["jpg"].to(accelerator.device).clip(-1.0, 1.0)
                            v_depth = val_batch["depth"].to(accelerator.device)
                            v_prompts = val_batch["caption"]
                            _, v_loss, _, _ = model.forward_easy(
                                v_imgs, v_prompts, cs=[v_depth], cfg_mask=[True], skip_encode=True
                            )
                            val_losses.append(v_loss.item())
                    mean_val_loss = float(np.mean(val_losses)) if val_losses else float("inf")
                    accelerator.log({"val_loss": mean_val_loss}, step=global_step)
                    logger.info(f"step {global_step}: val_loss={mean_val_loss:.5f}")

                    is_best = mean_val_loss < best_val_loss
                    best_val_loss = min(best_val_loss, mean_val_loss)

                    if accelerator.is_main_process and (global_step % cfg.ckpt_steps == 0 or stop_training):
                        try:
                            save_checkpoint_and_grids(global_step, is_best)
                        except Exception as e:
                            logger.info(f"validation/grid generation error: {e}")
                            logger.info(traceback.format_exc())

                    unet.train()
                    mapper.train()

            if stop_training or global_step >= max_train_steps:
                break
        if stop_training or global_step >= max_train_steps:
            break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        save_checkpoint(
            model.get_lora_state_dict(accelerator.unwrap_model(unet)),
            [accelerator.unwrap_model(mapper).state_dict()],
            None,
            output_path / f"checkpoint-{global_step}",
        )
    logger.info(f"done. final checkpoint: {output_path / f'checkpoint-{global_step}'}")


if __name__ == "__main__":
    main()
