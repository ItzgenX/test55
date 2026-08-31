"""
grounded_sam_training.py
---------------
Train the segmentation-conditioned LoRAdapter on PRE-SAVED Grounded-SAM class-ID
maps (Grounding DINO + SAM, this branch's own 28-class CARLA vocabulary --
see grounded_sam_map_calculations.py / src/encoders/grounded_sam_encoder.py).

QUICK COMMANDS (run from repo root with conda loradapter env active):
  # --- Smoke test (a few images, 3 short epochs) ---
  python grounded_sam_training.py experiment=train_grounded_sam epochs=3 data.batch_size=1 gradient_accumulation_steps=1 val_steps=5 ckpt_steps=10

  # --- Full training run ---
  python grounded_sam_training.py experiment=train_grounded_sam

  # --- Full training -- 4-GPU cluster ---
  accelerate launch --num_processes=4 grounded_sam_training.py experiment=train_grounded_sam_cluster

  # --- Resume from checkpoint ---
  python grounded_sam_training.py experiment=train_grounded_sam "lora.struct.ckpt_path=outputs/train/grounded_sam/runs/YYYY-MM-DD/HH-MM-SS/checkpoint-epoch1/step1000"

GPU / HARDWARE:
  data.batch_size, gradient_accumulation_steps, and gradient_checkpointing in
  configs/experiment/train_grounded_sam*.yaml are plain, fixed values --
  nothing here rescales them at runtime. What's written in the YAML is
  exactly what runs. This script REFUSES to run on CPU (no silent fallback)
  -- see print_gpu_diagnostics()/the device check right after Accelerator() below.

OUTPUT: outputs/train/<tag>/runs/YYYY-MM-DD/HH-MM-SS/
  training_params.txt  <- plain-text snapshot of the FULL resolved config this
                          run used (prompts, batch size, epochs, every setting)
                          -- read this to know what settings produced the
                          checkpoints in this folder.
  best_model/           <- weights of the best val/loss checkpoint
  checkpoint-epochN/    <- per-epoch + per-ckpt_steps checkpoints with sample images
  logs/tensorboard/     <- TensorBoard event files
TENSORBOARD: tensorboard --logdir outputs/train/<tag>/runs/

STRUCTURE: this file mirrors the segformer branch's own segformer_training.py
in NAMING and STRUCTURE (startup banner, best_model tracking, decoupled
val_steps/ckpt_steps, labeled checkpoint-monitoring images, training_params.txt,
LR warmup, gradient clipping, early stopping) -- but the CONDITIONING DATA is
NOT copied from segformer's architecture, it stays this branch's own:

  1. batch["seg"] here is a [B,H,W] Long CLASS-ID map (1..28 real classes,
     0=NULL/CFG-dropout, 29=IGNORE/void -- see local_seg.py's _load_seg_ids
     and SegIDStructureMapperXL's docstring), NOT a [B,3,H,W] RGB colour map.
     There is no colorize/decolorize round-trip anywhere in the conditioning
     path -- colourisation only ever happens for a DISPLAY panel, never for
     what the model actually trains on.
  2. skip_encode=True bypasses the encoder during training exactly as
     segformer does, but the DEFAULT encoder slot here is torch.nn.Identity
     (configs/lora/encoder/identity.yaml), not a loaded model -- Grounding
     DINO + SAM cost real VRAM/startup time neither training loss nor most
     validation needs, since the conditioning map is already precomputed.
  3. Because of (2), the mIoU controllability metric (see
     _save_checkpoint_grounded_sam_images below) only runs when the
     configured encoder actually has a live inference path
     (getattr(encoder, "live_available", False) -- defaults False here,
     unlike segformer's default-True, because Identity has no such path at
     all). Swap lora.struct.encoder to configs/lora/encoder/grounded_sam.yaml
     if you want live mIoU scoring during training -- accept the extra
     DINO+SAM VRAM/load cost deliberately, don't get it by accident.
  4. Targets SDXL (configs/model/sdxl.yaml), whose 3-stage UNet only needs
     H/W divisible by 32 -- your native 1280x800 trains directly, no resize.

KEY GROUNDED-SAM-SPECIFIC POINTS:
  - Monitoring panel labels: "SEG MAP" (colourised on the fly from the raw
    class-id map, CARLA palette), "RAW SEG GEN".
  - Model loading: cfg.dino_model_path/dino_model_name + cfg.sam_model_path/
    sam_model_name (two separate models), only rewritten when the configured
    encoder actually has those keys (Identity doesn't).
"""

import hydra
import math
import os
import random
import signal
import time
import traceback
from datetime import datetime
from functools import reduce
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as TF
from accelerate import Accelerator
from accelerate.logging import get_logger
from diffusers.optimization import get_scheduler
from hydra.utils import get_original_cwd

_module_logger = get_logger(__name__)
from PIL import Image, ImageDraw
from tqdm.auto import tqdm

from src.model import ModelBase
from src.utils import add_lora_from_config, save_checkpoint, print_gpu_diagnostics, write_training_params_txt, compute_miou
from src.encoders.seg_encoder import seg_colorize_ids
from src.encoders.grounded_sam_encoder import carla_palette_tensor, CARLA_CLASSES
from src.data.transforms import normalize_size


torch.set_float32_matmul_precision("high")

stop_training = False


def signal_handler(sig, frame):
    global stop_training
    stop_training = True
    print("got stop signal")


# ── Checkpoint-monitoring images ───────────────────────────────────────────────
# Every time a checkpoint is saved, that SAME checkpoint generates N validation
# images so you can judge it by eye. Each scene is saved as its OWN labeled file
# (a 3-panel "explained" image: ORIGINAL | SEG MAP | PREDICTED) INSIDE the
# checkpoint's own folder, next to its weights. One prompts.txt lists all N
# prompts. Mirrors segformer_training.py's structure exactly.

def _gsam_label_bar(width: int, text: str, bar_h: int = 24) -> np.ndarray:
    """Dark bar with centred yellow text label. Returns [bar_h, width, 3] uint8.
    Identical to segformer_training.py's _seg_label_bar -- pure image-drawing
    code, no dependency on the conditioning data format."""
    bar = Image.new("RGB", (width, bar_h), color=(25, 25, 25))
    draw = ImageDraw.Draw(bar)
    bbox = draw.textbbox((0, 0), text)
    draw.text(((width - bbox[2]) // 2, 4), text, fill=(255, 220, 60))
    return np.asarray(bar)


def _gsam_colorize_for_display(seg_ids: torch.Tensor, palette: torch.Tensor) -> torch.Tensor:
    """[H,W] or [B,H,W] Long class-id map (1..28 real, 0=NULL, 29=IGNORE) ->
    [B,3,H,W] float RGB in [0,1], for DISPLAY ONLY. The actual conditioning
    signal fed to the model never goes through this -- it stays raw class ids
    the whole way (see SegIDStructureMapperXL). 0 and 29 both fall outside the
    28-class palette -- mask to a safe index before the lookup, paint those
    pixels black after (same guard GroundedSamEncoder.forward() uses)."""
    if seg_ids.dim() == 2:
        seg_ids = seg_ids.unsqueeze(0)
    void_mask = (seg_ids == 0) | (seg_ids >= 29)
    real_ids = torch.where(void_mask, torch.zeros_like(seg_ids), seg_ids - 1)
    colour = seg_colorize_ids(real_ids, palette)  # [B,3,H,W] in [0,1]
    return torch.where(void_mask.unsqueeze(1), torch.zeros_like(colour), colour)


def _gsam_scene_image(
    orig_11: torch.Tensor,     # [3,H,W] in [-1,1]  -- raw validation image
    seg_ids: torch.Tensor,     # [H,W] Long          -- raw class-id conditioning map
    pred_pil: Image.Image,     # PIL -- generation WITH the text prompt
    size,
    palette: torch.Tensor,
    raw_pil: Image.Image | None = None,  # PIL or None -- generation WITHOUT prompt
) -> Image.Image:
    """One EXPLAINED image for a single validation scene -- labeled panels side
    by side: ORIGINAL | SEG MAP | PREDICTED (and | RAW SEG GEN if raw_pil given).

    Mirrors segformer_training.py's _seg_scene_image, adapted for class-id
    input: seg_ids is the raw [H,W] Long map (not an already-RGB [3,H,W]
    tensor) -- colourised here via _gsam_colorize_for_display, purely for
    this panel."""
    size_w, size_h = normalize_size(size)
    target = (size_w, size_h)
    orig_np = np.asarray(
        TF.to_pil_image(((orig_11.float() + 1) / 2).clamp(0, 1).cpu())
        .resize(target).convert("RGB")
    )
    seg_colour = _gsam_colorize_for_display(seg_ids.cpu(), palette.cpu())[0]  # [3,H,W] in [0,1]
    seg_np = np.asarray(
        TF.to_pil_image(seg_colour.float().clamp(0, 1))
        .resize(target).convert("RGB")
    )
    pred_np = np.asarray(pred_pil.resize(target).convert("RGB"))

    texts = ["ORIGINAL", "SEG MAP", "PREDICTED"]
    columns = [orig_np, seg_np, pred_np]

    if raw_pil is not None:
        texts.append("RAW SEG GEN")
        columns.append(np.asarray(raw_pil.resize(target).convert("RGB")))

    labels = np.concatenate([_gsam_label_bar(size_w, t) for t in texts], axis=1)
    panels = np.concatenate(columns, axis=1)
    return Image.fromarray(np.concatenate([labels, panels], axis=0))


def _gsam_target_ids_native(seg_ids: torch.Tensor) -> torch.Tensor:
    """[H,W] Long stored-scheme class ids (1..28 real, 0=NULL, 29=IGNORE) ->
    GroundedSamEncoder's own NATIVE scheme (0..27 real, 255=IGNORE_ID), so
    they're directly comparable to what encoder.label_ids() returns on a
    freshly generated image. Unlike segformer's mIoU (which has to invert an
    RGB colour map back to ids via seg_ids_from_colormap), this is a plain
    index shift -- our conditioning was never colourised to begin with."""
    void_mask = (seg_ids == 0) | (seg_ids >= 29)
    return torch.where(void_mask, torch.full_like(seg_ids, 255), seg_ids - 1)


def _save_checkpoint_grounded_sam_images(
    model, val_dataset, idxs, kinds, n_loras, cfg, cfg_mask, device, out_dir, include_empty, palette
):
    """Generate + save the monitoring images for one checkpoint into out_dir
    (the SAME folder as that checkpoint's weights). One labeled file per
    scene, plus a single prompts.txt. Returns (prompts, [np_images], metrics).

    Mirrors segformer_training.py's _save_checkpoint_segmentation_images
    structurally (fixed+fresh scene split, per-scene labeled image, metrics
    dict), with grounded_sam's own data:
      - item["seg"] is [H,W] Long class ids, not [3,H,W] RGB.
      - skip_encode=True (pre-saved map -> mapper, same as training).
      - mIoU only computed if the configured encoder has a live inference
        path (see module docstring point 3) -- gracefully skipped otherwise,
        logged once per call, never crashes.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    prompts, images = [], []
    mious = []  # controllability metric, FIXED scenes only (comparable checkpoint-to-checkpoint)
    size_w, size_h = normalize_size(cfg.size)

    _enc0 = getattr(model.encoders[0], "module", model.encoders[0])
    _live_seg = getattr(_enc0, "live_available", False)

    n_steps = int(cfg.get("val_num_inference_steps", 50))
    _module_logger.info(
        f"[grounded_sam grid] generating {len(idxs)} scene(s), "
        f"{n_steps} inference steps each, {size_w}x{size_h}, include_empty={include_empty}"
    )

    for n, (idx, kind) in enumerate(zip(idxs, kinds)):
        item = val_dataset[idx]
        seg_ids = item["seg"].unsqueeze(0).to(device)   # [1,H,W] Long
        cs = [seg_ids] * n_loras
        prompt = cfg.prompt if cfg.get("prompt") else item["caption"]

        _t0 = time.time()
        pred = model.sample(
            prompt=[prompt], num_images_per_prompt=1, cs=cs,
            generator=torch.Generator(device=device).manual_seed(cfg.seed),
            cfg_mask=cfg_mask, skip_encode=True,
            height=size_h, width=size_w, num_inference_steps=n_steps,
        )[0]
        _module_logger.info(
            f"[grounded_sam grid] scene {n + 1}/{len(idxs)} ({kind}) pred sample "
            f"done in {time.time() - _t0:.1f}s"
        )

        raw = None
        if include_empty:
            _t0 = time.time()
            raw = model.sample(
                prompt=[""], num_images_per_prompt=1, cs=cs,
                generator=torch.Generator(device=device).manual_seed(cfg.seed),
                cfg_mask=cfg_mask, skip_encode=True,
                height=size_h, width=size_w, num_inference_steps=n_steps,
            )[0]
            _module_logger.info(
                f"[grounded_sam grid] scene {n + 1}/{len(idxs)} ({kind}) raw sample "
                f"done in {time.time() - _t0:.1f}s"
            )

        img = _gsam_scene_image(item["jpg"], seg_ids[0].cpu(), pred, cfg.size, palette, raw_pil=raw)
        img.save(out_dir / f"sample_{n:02d}_{kind}.jpg", quality=95)
        prompts.append(prompt)
        images.append(np.asarray(img))

        if kind == "fixed" and _live_seg:
            target_ids = _gsam_target_ids_native(seg_ids[0].cpu())
            gen_t = (TF.to_tensor(pred.resize((size_w, size_h)).convert("RGB"))
                      .unsqueeze(0).to(device) * 2.0 - 1.0)
            pred_ids = _enc0.label_ids(gen_t)[0].cpu()
            mious.append(compute_miou(pred_ids, target_ids, num_classes=len(CARLA_CLASSES)))

    (out_dir / "prompts.txt").write_text(
        "\n".join(f"[{n}] [{k}] {p}" for n, (k, p) in enumerate(zip(kinds, prompts))),
        encoding="utf-8",
    )
    metrics = {}
    if mious:
        metrics["val/miou_fixed"] = float(np.mean(mious))
    elif not _live_seg:
        # Logged once per checkpoint, not per scene -- avoids spamming the log
        # every single validation when the encoder is simply Identity (today's
        # default). See module docstring point 3 for how to enable this.
        print("[grounded_sam grid] mIoU skipped -- configured encoder has no live "
              "inference path (lora.struct.encoder is Identity). Swap to "
              "configs/lora/encoder/grounded_sam.yaml for live mIoU scoring.")
    return prompts, images, metrics


def _grounded_sam_validation_loss(
    model, val_dataloader, n_loras, cfg, cfg_mask, accelerator, max_batches
):
    """Standard validation: run the SAME denoising loss as training on
    HELD-OUT validation data WITHOUT backprop, averaged over up to
    max_batches batches. Mirrors segformer_training.py's
    _segmentation_validation_loss exactly -- this function is opaque to
    whether batch["seg"] is RGB or class-id, so nothing here needed adapting.

    Gives you: (a) a val/loss curve to watch for overfitting, (b) an
    objective best_model criterion (lowest val/loss), not a biased
    training-loss average.
    """
    device = accelerator.device

    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    torch.manual_seed(cfg.seed)

    model.unet.eval()
    for m in model.mappers:
        m.eval()
    for e in model.encoders:
        e.eval()

    total = torch.tensor(0.0, device=device)
    count = torch.tensor(0.0, device=device)
    with torch.no_grad():
        for i, batch in enumerate(val_dataloader):
            if i >= max_batches:
                break
            imgs = batch["jpg"].to(device).clip(-1.0, 1.0)
            B = imgs.shape[0]
            seg = batch["seg"].to(device)
            cs = [seg] * n_loras
            prompts = [cfg.prompt] * B if cfg.get("prompt") else batch["caption"]
            _, loss, _, _ = model.forward_easy(
                imgs, prompts, cs,
                cfg_mask=[True for _ in cfg_mask],
                skip_encode=True,
                batch=batch,
            )
            total += loss.detach()
            count += 1

    model.unet.train()
    for m in model.mappers:
        m.train()
    for e in model.encoders:
        e.train()

    torch.set_rng_state(cpu_rng)
    if cuda_rng is not None:
        torch.cuda.set_rng_state_all(cuda_rng)

    total = accelerator.reduce(total, reduction="sum")
    count = accelerator.reduce(count, reduction="sum")
    return (total / torch.clamp(count, min=1.0)).item()
# ─────────────────────────────────────────────────────────────────────────────


@hydra.main(config_path="configs", config_name="train", version_base=None)
def main(cfg):
    global stop_training
    if hasattr(signal, "SIGUSR1"):  # POSIX-only; this repo also runs on Windows dev boxes
        signal.signal(signal.SIGUSR1, signal_handler)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    output_path = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)

    # (width, height) -- grounded_sam_map_calculations.py's convention. diffusers
    # wants height first, so keep both spellings straight from the start.
    size_w, size_h = normalize_size(cfg.size)

    # grounded_sam_map_calculations.py hard-enforces this for the offline
    # precompute step, but `size:` here is a separate config value with
    # nothing keeping it in sync -- an edited/overridden size that isn't
    # divisible by 32 would otherwise silently misalign the mapper's output
    # resolution against the UNet's real feature-map resolution instead of
    # failing loudly (SDXL: VAE/8 x 2 UNet halvings/4 = /32 total, verified
    # against the actual loaded UNet/VAE, not just this comment).
    if size_w % 32 or size_h % 32:
        raise ValueError(
            f"cfg.size = ({size_w}, {size_h}) -- both width and height must be divisible "
            f"by 32 for SDXL (VAE/8 x 2 UNet halvings/4). See grounded_sam_map_calculations.py's "
            f"--width/--height guard for the same check on the offline precompute side."
        )

    # hydra.job.chdir=true (configs/train.yaml) moves the process cwd to the
    # run's output dir before this runs -- a relative data.train_jsonl/
    # val_jsonl (the config default) then resolves against THAT dir, not the
    # repo root, and SegJsonDataModule fails with FileNotFoundError.
    _root = get_original_cwd()
    if cfg.get("data", {}).get("train_jsonl") and not os.path.isabs(cfg.data.train_jsonl):
        cfg.data.train_jsonl = os.path.join(_root, cfg.data.train_jsonl)
    if cfg.get("data", {}).get("val_jsonl") and not os.path.isabs(cfg.data.val_jsonl):
        cfg.data.val_jsonl = os.path.join(_root, cfg.data.val_jsonl)

    # ── LOCAL MODELS ONLY when local_files_only=true ────────────────────────
    # Same base_model_path contract as grounded_sam_inference.py
    # (configs/inference_grounded_sam.yaml). The encoder slot has TWO models
    # (Grounding DINO + SAM), not one -- only rewritten when the configured
    # encoder actually has dino_model/sam_model keys (Identity, today's
    # training default, has neither -- this simply no-ops for it).
    _enc_has_dino_sam = "dino_model" in cfg.lora.struct.get("encoder", {}) and "sam_model" in cfg.lora.struct.get("encoder", {})
    if cfg.get("local_files_only", False):
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        if cfg.get("base_model_path"):
            cfg.model.model_name = os.path.join(_root, cfg.base_model_path)
        if _enc_has_dino_sam and cfg.get("dino_model_path") and cfg.get("sam_model_path"):
            cfg.lora.struct.encoder.dino_model = os.path.join(_root, cfg.dino_model_path)
            cfg.lora.struct.encoder.sam_model = os.path.join(_root, cfg.sam_model_path)
        if cfg.get("vae_local_path") and cfg.model.get("vae_path"):
            cfg.model.vae_path = os.path.join(_root, cfg.vae_local_path)

    accelerator = Accelerator(
        project_dir=output_path / "logs",
        log_with="tensorboard",
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        mixed_precision="bf16",
    )

    # Accelerate silently uses CPU if no GPU is visible -- print full GPU
    # diagnostics and refuse to train on CPU, so a misconfigured environment
    # fails LOUDLY here instead of training unnoticeably slowly.
    if accelerator.is_main_process:
        print_gpu_diagnostics()
    if accelerator.device.type != "cuda":
        raise RuntimeError(
            f"accelerate selected device {accelerator.device!r}, not a GPU. Training on "
            f"CPU is never intended for this pipeline. Check `nvidia-smi` and that this "
            f"conda env's torch build has CUDA support "
            f"(python -c \"import torch; print(torch.version.cuda)\")."
        )

    logger = get_logger(__name__)
    logger.info("==================================")
    logger.info(cfg)
    logger.info(output_path)

    # Snapshot the parameters this run uses into a plain-text file next to
    # the checkpoints (full resolved config -- prompts, batch size, epochs,
    # everything -- so a result folder is always self-documenting).
    if accelerator.is_main_process:
        write_training_params_txt(cfg, output_path, str(accelerator.device), original_cwd=_root)

    cfg = hydra.utils.instantiate(cfg)
    model: ModelBase = cfg.model

    model = model.to(accelerator.device)
    model.pipe.to(accelerator.device)
    n_loras = len(cfg.lora.keys())

    cfg_mask = add_lora_from_config(model, cfg, accelerator.device)

    if cfg.get("gradient_checkpointing", False):
        model.unet.enable_gradient_checkpointing()

    dm = cfg.data
    train_dataloader = dm.train_dataloader()
    val_dataloader = dm.val_dataloader()

    mappers_params = list(
        filter(lambda p: p.requires_grad, reduce(lambda x, y: x + list(y.parameters()), model.mappers, []))
    )
    encoder_params = list(
        filter(lambda p: p.requires_grad, reduce(lambda x, y: x + list(y.parameters()), model.encoders, []))
    )
    optimizer = torch.optim.AdamW(
        model.params_to_optimize + mappers_params + encoder_params,
        lr=cfg.learning_rate,
    )

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / cfg.gradient_accumulation_steps)
    max_train_steps = cfg.epochs * num_update_steps_per_epoch
    if cfg.get("max_train_steps", None) is not None:
        max_train_steps = cfg.max_train_steps

    lr_scheduler = get_scheduler(
        cfg.lr_scheduler, optimizer=optimizer,
        num_warmup_steps=cfg.get("lr_warmup_steps", 0) * accelerator.num_processes,
        num_training_steps=max_train_steps * accelerator.num_processes,
    )

    logger.info(f"Mapper params:  {sum(p.numel() for p in mappers_params):,}")
    logger.info(f"Encoder params: {sum(p.numel() for p in encoder_params):,}")
    logger.info(f"LoRA params:    {sum(p.numel() for p in model.params_to_optimize):,}")

    # ── GROUNDED-SAM TRAINING STARTUP BANNER ──────────────────────────────────
    if accelerator.is_main_process:
        tb_dir = output_path / "logs" / "tensorboard"
        logger.info("")
        logger.info("=" * 64)
        logger.info("  PIPELINE   :  GROUNDED-SAM  (Grounding DINO + SAM, 28-class CARLA conditioning)")
        logger.info(f"  resize_mode:  {cfg.get('resize_mode', 'aspect')}")
        logger.info(f"  Output     :  {output_path}")
        logger.info(f"  TensorBoard:  tensorboard --logdir \"{tb_dir}\"")
        logger.info(f"  Train      :  {len(dm.train_dataset):,} images  |  Val: {len(dm.val_dataset):,} images")
        logger.info(f"  Schedule   :  {cfg.epochs} epochs x {num_update_steps_per_epoch} steps = {max_train_steps} total optimizer steps")
        logger.info(f"  LR         :  {cfg.learning_rate}  scheduler={cfg.lr_scheduler}  warmup={cfg.get('lr_warmup_steps', 0)} steps")
        logger.info(f"  val_steps  :  {cfg.val_steps}  (val/loss + best_model update)")
        logger.info(f"  ckpt_steps :  {cfg.ckpt_steps}  (weights + {cfg.get('n_grid_images', 10)} grid images saved)")
        logger.info(f"  TensorBoard scalars: train/loss  train/lr  train/grad_norm  train/epoch  val/loss")
        logger.info(f"  TensorBoard images : val/sample_00 ... val/sample_{cfg.get('n_grid_images', 10) - 1:02d}  (ORIGINAL | SEG MAP | PREDICTED)")
        logger.info("=" * 64)
        logger.info("")

    if accelerator.is_main_process:
        import socket as _socket
        _socket.gethostname = lambda: str(cfg.get("tag", "loradapter"))
        accelerator.init_trackers("tensorboard")

    prepared = accelerator.prepare(
        *model.mappers, *model.encoders, model.unet,
        optimizer, train_dataloader, val_dataloader, lr_scheduler,
    )
    mappers = prepared[: len(model.mappers)]
    encoders = prepared[len(model.mappers): len(model.mappers) + len(model.encoders)]
    (unet, optimizer, train_dataloader, val_dataloader, lr_scheduler) = prepared[
        len(model.mappers) + len(model.encoders):
    ]
    model.unet = unet
    model.mappers = mappers
    model.encoders = encoders

    _display_palette = carla_palette_tensor()  # classid conditioning's val-grid display panel only

    global_step = 0
    progress_bar = tqdm(range(global_step, max_train_steps), disable=not accelerator.is_main_process)
    progress_bar.set_description("Steps")

    # ── Best-model + early-stop tracking ────────────────────────────────────
    best_loss = float("inf")
    best_epoch = None
    best_step = None
    best_epoch_frac = None

    # ── Checkpoint-monitoring images setup ──────────────────────────────────
    # n_grid_images grounded_sam scenes per checkpoint, split 50/50: FIXED half
    # drawn once with OS entropy at the start of this run (watch the SAME
    # scenes improve over training), FRESH half re-drawn at each checkpoint.
    # SOURCE = validation set only.
    n_grid_images = max(2, min(int(cfg.get("n_grid_images", 10)), len(dm.val_dataset)))
    include_empty = bool(cfg.get("grid_include_empty_prompt", False))
    n_fixed = n_grid_images // 2
    n_random = n_grid_images - n_fixed

    _fixed_val_idxs = random.Random().sample(range(len(dm.val_dataset)), n_fixed)
    logger.info(
        f"Grounded-SAM grid: {n_grid_images} val scenes = {n_fixed} fixed (OS entropy) "
        f"{_fixed_val_idxs} + {n_random} re-randomized each checkpoint "
        f"(include_empty={include_empty}, val size={len(dm.val_dataset)})"
    )

    def save_gsam_ckpt_and_grid(stem, is_best=False, info_lines=None):
        """Save the CURRENT model as a checkpoint AND its monitoring images
        together. Mirrors segformer_training.py's save_seg_ckpt_and_grid.
        Grid generation is best-effort: errors are logged but never block
        the checkpoint save. Main process only."""
        if not accelerator.is_main_process:
            return

        _t_ckpt0 = time.time()
        ckpt_dir = (output_path / "best_model") if is_best else (output_path / stem)
        save_checkpoint(
            model.get_lora_state_dict(accelerator.unwrap_model(unet)),
            [accelerator.unwrap_model(m).state_dict() for m in mappers],
            None, ckpt_dir,
        )
        logger.info(f"[grounded_sam grid] {stem}: weights saved in {time.time() - _t_ckpt0:.1f}s -- generating monitoring images now")

        try:
            unet.eval()
            for m in mappers:
                m.eval()
            for e in encoders:
                e.eval()

            pool = [i for i in range(len(dm.val_dataset)) if i not in set(_fixed_val_idxs)]
            new_idxs = random.sample(pool, min(n_random, len(pool)))
            idxs = list(_fixed_val_idxs) + new_idxs
            kinds = ["fixed"] * len(_fixed_val_idxs) + ["new"] * len(new_idxs)

            with torch.no_grad():
                prompts, images, metrics = _save_checkpoint_grounded_sam_images(
                    model, dm.val_dataset, idxs, kinds, n_loras, cfg, cfg_mask,
                    accelerator.device, ckpt_dir, include_empty, _display_palette,
                )

            if is_best:
                metric_lines = [f"{k}: {v:.4f}" for k, v in metrics.items()]
                (ckpt_dir / "info.txt").write_text(
                    "\n".join((info_lines or []) + metric_lines), encoding="utf-8"
                )

            for tracker in accelerator.trackers:
                if tracker.name == "tensorboard":
                    for n, img in enumerate(images):
                        tracker.writer.add_image(f"val/sample_{n:02d}", img, global_step, dataformats="HWC")
                    tracker.writer.add_text("val/prompts", " | ".join(prompts), global_step)
                    for k, v in metrics.items():
                        tracker.writer.add_scalar(k, v, global_step)

            if metrics:
                logger.info(f"[grounded_sam metric] {stem}: " + "  ".join(f"{k}={v:.4f}" for k, v in metrics.items()))
            logger.info(f"[grounded_sam grid] {stem}: {len(prompts)} scene images -> {ckpt_dir}")

        except Exception as e:
            print("!!! ERROR generating grounded_sam checkpoint images !!!")
            print(e)
            print(traceback.format_exc())
        finally:
            unet.train()
            for m in mappers:
                m.train()
            for e in encoders:
                e.train()

    def do_grounded_sam_validation(label, epoch_num, epoch_frac):
        """VALIDATION ONLY -- the CHEAP half, run every val_steps: compute
        val/loss, log it, update best_model/ when it improves. Does NOT save
        a regular checkpoint -- that's decoupled, controlled by ckpt_steps."""
        nonlocal best_loss, best_epoch, best_step, best_epoch_frac
        val_loss = _grounded_sam_validation_loss(
            model, val_dataloader, n_loras, cfg, cfg_mask,
            accelerator, int(cfg.get("val_batches", 8)),
        )
        accelerator.log({"val/loss": val_loss}, step=global_step)
        if accelerator.is_main_process:
            since = "" if best_epoch is None else (
                f"  | best: epoch{best_epoch} step{best_step} ({best_loss:.6f})"
                f"  | {epoch_num - best_epoch} epoch(s) since improvement")
            logger.info(f"[grounded_sam val] {label}: val/loss = {val_loss:.6f}{since}")
        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch_num
            best_step = global_step
            best_epoch_frac = epoch_frac
            info = [
                f"from:        {label}",
                f"epoch:       {epoch_num}",
                f"epoch_frac:  {epoch_frac:.2f}",
                f"global_step: {global_step}",
                f"val/loss:    {val_loss:.6f}",
                f"timestamp:   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                f"resize_mode: {cfg.get('resize_mode', 'aspect')}",
            ]
            save_gsam_ckpt_and_grid("best_model", is_best=True, info_lines=info)
            if accelerator.is_main_process:
                logger.info(f"*** NEW BEST (grounded_sam) *** epoch {epoch_num} ({epoch_frac:.2f}), "
                            f"step {global_step}, val/loss={val_loss:.6f}")

    # ── Training loop ──────────────────────────────────────────────────────
    logger.info("[GROUNDED_SAM] start training")
    for epoch in range(cfg.epochs):
        logger.info(f"[GROUNDED_SAM] Epoch {epoch + 1}/{cfg.epochs} started  (global_step={global_step})")
        unet.train()
        for m in mappers:
            m.train()
        for e in encoders:
            e.train()

        for step, batch in enumerate(train_dataloader):
            _grad_norm = None

            with accelerator.accumulate(unet, *mappers, *encoders):
                imgs = batch["jpg"].to(accelerator.device).clip(-1.0, 1.0)
                B = imgs.shape[0]

                # precomputed class-id conditioning map, not the training image itself
                seg = batch["seg"].to(accelerator.device)
                cs = [seg] * n_loras

                prompts = (
                    [cfg.prompt] * B
                    if cfg.get("prompt", None) is not None
                    else batch["caption"]
                )

                model_pred, loss, x0, _ = model.forward_easy(
                    imgs, prompts, cs,
                    cfg_mask=[True for _ in cfg_mask],
                    skip_encode=True,  # cs is already a finished conditioning map, don't re-encode it
                    batch=batch,
                )

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    all_params = [p for g in optimizer.param_groups for p in g["params"]]
                    _g = accelerator.clip_grad_norm_(all_params, max_norm=1.0)
                    _grad_norm = float(_g)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            loss_val = loss.detach().item()
            lr_val = lr_scheduler.get_last_lr()[0]
            epoch_frac = epoch + step / max(len(train_dataloader), 1)
            log_dict = {
                "train/loss": loss_val,
                "train/lr": lr_val,
                "train/epoch": epoch_frac,
            }
            if _grad_norm is not None:
                log_dict["train/grad_norm"] = _grad_norm
            progress_bar.set_postfix(loss=loss_val, lr=f"{lr_val:.2e}", gnorm=f"{_grad_norm or 0:.3f}", refresh=False)
            accelerator.log(log_dict, step=global_step)

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if global_step >= max_train_steps:
                    stop_training = True

                if global_step % int(cfg.get("log_every_steps", 50)) == 0:
                    logger.info(
                        f"[train] step {global_step}/{max_train_steps}  "
                        f"loss={loss_val:.4f}  lr={lr_val:.2e}  "
                        f"grad_norm={_grad_norm if _grad_norm is not None else float('nan'):.3f}  "
                        f"epoch={epoch_frac:.2f}")

                # ── DECOUPLED step-level triggers ───────────────────────────
                # val_steps  = how often to compute val/loss (cheap; also
                #              updates best_model when val/loss improves).
                # ckpt_steps = how often to write a checkpoint to disk (heavy:
                #              weights + N monitoring images). INDEPENDENT --
                #              e.g. validate every 500 but save every 1000.
                if global_step % cfg.val_steps == 0 or stop_training:
                    do_grounded_sam_validation(f"step{global_step}", epoch + 1, epoch_frac)
                if global_step % cfg.ckpt_steps == 0 or stop_training:
                    save_gsam_ckpt_and_grid(f"checkpoint-epoch{epoch + 1}/step{global_step}")

            if stop_training:
                break

        # ── END-OF-EPOCH: ALWAYS validate AND save this epoch's checkpoint ──
        do_grounded_sam_validation(f"epoch{epoch + 1}", epoch + 1, float(epoch + 1))
        save_gsam_ckpt_and_grid(f"checkpoint-epoch{epoch + 1}/checkpoint-epoch{epoch + 1}")

        if stop_training:
            break

        # ── EARLY STOPPING (optional; early_stop_patience epochs, 0 = off) ──
        patience = int(cfg.get("early_stop_patience", 0))
        if patience > 0 and best_epoch is not None:
            epochs_since_best = (epoch + 1) - best_epoch
            if accelerator.is_main_process:
                logger.info(f"[early-stop] {epochs_since_best} epoch(s) since best "
                            f"(epoch {best_epoch}, val/loss={best_loss:.6f}); patience={patience}")
            if epochs_since_best >= patience:
                if accelerator.is_main_process:
                    logger.info(f"[early-stop] no improvement for {patience} epoch(s) -- stopping. "
                                f"Best model: epoch {best_epoch}, step {best_step}, "
                                f"val/loss={best_loss:.6f} (see best_model/info.txt).")
                break

    # ── Final snapshot on early-stop only ───────────────────────────────────
    # On normal completion the end-of-last-epoch block above already ran
    # save_gsam_ckpt_and_grid for this exact folder. On early-stop (max_train_steps
    # hit mid-epoch, or a signal) the epoch loop breaks BEFORE that block runs,
    # so this is the only place capturing the interrupted epoch's final state.
    if stop_training:
        accelerator.wait_for_everyone()
        save_gsam_ckpt_and_grid(f"checkpoint-epoch{epoch + 1}/checkpoint-epoch{epoch + 1}")


if __name__ == "__main__":
    main()
