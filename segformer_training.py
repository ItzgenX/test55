"""Train a segmentation-conditioned LoRA using precomputed class-ID maps.

Same training loop shape as train.py, adapted for segmentation's
precomputed-map workflow (see seg_map_calculations.py / local_seg.py):

  1. Data comes from SegJsonDataset -- "jpg" (RGB, VAE target) and "seg"
     (the colourised conditioning map, loaded + coloured from a raw
     class-ID PNG at __getitem__ time), not a live encoder run on the image.
  2. Every model call passes skip_encode=True, so model.forward() feeds
     batch["seg"] straight to the mapper instead of running a live encoder
     on it (src/model.py's skip_encode branch).
  3. Targets SDXL (configs/model/sdxl.yaml), whose 3-stage UNet only needs
     H/W divisible by 32 -- your native 1280x800 trains directly, no resize.

MONITORING (what this run leaves behind, per checkpoint):

    outputs/train/<tag>/runs/<date>/<time>/
      training_params.txt              full resolved config of this run
      best_model/                      lowest val/loss so far (overwritten)
        struct/{lora,mapper}-checkpoint.pt
        sample_00_fixed.jpg ...        ORIGINAL | SEG MAP | PREDICTED
        prompts.txt                    one line per scene, [fixed]/[new] tagged
        info.txt                       when + how good (epoch/step/loss/metrics)
      checkpoint-epoch1/
        step500/                       periodic save (ckpt_steps)
          struct/... + samples + prompts.txt
        epoch_end/                     end-of-epoch save
      logs/tensorboard/

  n_grid_images scenes per checkpoint, split 50/50:
    FIXED half  -- drawn once at startup, reused at EVERY checkpoint, so you
                   watch the same scenes improve. Metrics use these only.
    FRESH half  -- redrawn each checkpoint, a generalization peek.
  Source is the VALIDATION set only, never train.

  Metrics on fixed scenes (fixed seed) -> comparable checkpoint-to-checkpoint:
    val/psnr_fixed, val/ssim_fixed  -- always
    val/miou_fixed                  -- only when val_miou=true (see below)

  val_miou: computing mIoU means segmenting the GENERATED image, which needs a
  live segmenter in VRAM. Training itself does NOT load one (skip_encode reads
  precomputed maps; the encoder slot is torch.nn.Identity), so this flag opts
  into loading ONE SegFormer on the main process only. Default false: a 12GB
  dev box already runs near its VRAM ceiling at 1280x800, and the metric is
  only meaningful on real cluster-scale runs anyway. Fails CLOSED -- if the
  segmenter cannot be built, mIoU is skipped and everything else still runs.

Usage:
    python segformer_training.py experiment=train_seg
"""

import hydra
from hydra.utils import get_original_cwd
import math
from src.model import ModelBase
from src.data.transforms import normalize_size
from diffusers.optimization import get_scheduler
import torch
from accelerate import Accelerator
from accelerate.utils import InitProcessGroupKwargs
from tqdm.auto import tqdm
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import torchvision.transforms.functional as TF
from PIL import Image, ImageDraw
from accelerate.logging import get_logger

_module_logger = get_logger(__name__)
import random
import signal
import os
import time
import traceback
from functools import reduce

from src.utils import (
    add_lora_from_config,
    save_checkpoint,
    print_gpu_diagnostics,
    write_training_params_txt,
    compute_psnr_ssim,
    compute_miou,
)
from src.encoders.seg_encoder import seg_colorize_ids


torch.set_float32_matmul_precision("high")


stop_training = False


def signal_handler(sig, frame):
    global stop_training
    stop_training = True
    print("got stop signal")


# ── Checkpoint-monitoring images ─────────────────────────────────────────────
# Every checkpoint generates N validation scenes so it can be judged by eye.
# Each scene is its OWN labeled file inside the checkpoint's folder, next to
# that checkpoint's weights -- so a checkpoint and the images it produced can
# never drift apart.


def _seg_label_bar(width: int, text: str, bar_h: int = 24) -> np.ndarray:
    """Dark bar with centred yellow caption. Returns [bar_h, width, 3] uint8.
    Same construction as segformer_inference.py's own label bar, so a training
    monitoring image and an inference grid are visually consistent."""
    bar = Image.new("RGB", (width, bar_h), color=(25, 25, 25))
    draw = ImageDraw.Draw(bar)
    bbox = draw.textbbox((0, 0), text)
    draw.text((max(0, (width - (bbox[2] - bbox[0])) // 2), 4), text, fill=(255, 220, 60))
    return np.asarray(bar)


def _seg_scene_image(
    orig_11: torch.Tensor,      # [3,H,W] in [-1,1] -- the real validation photo
    seg_01: torch.Tensor,       # [3,H,W] in [0,1]  -- the conditioning colour map
    pred_pil: Image.Image,      # generation WITH the text prompt
    size,
    raw_pil: Image.Image | None = None,  # generation WITHOUT a prompt, or None
) -> Image.Image:
    """One labeled image for a single validation scene:

        ORIGINAL | SEG MAP | PREDICTED   (+ | RAW SEG GEN when raw_pil given)

    Saved as its own file rather than tiled into one grid, so each panel stays
    large enough to actually judge. RAW SEG GEN (empty prompt) shows structure
    adherence with no text to lean on -- the honest test of whether the LoRA
    learned anything, since a strong prompt alone can produce a plausible image
    with zero contribution from the conditioning.
    """
    size_w, size_h = normalize_size(size)
    target = (size_w, size_h)  # PIL .resize() order: (width, height)

    orig_np = np.asarray(
        TF.to_pil_image(((orig_11.float() + 1) / 2).clamp(0, 1).cpu()).resize(target).convert("RGB")
    )
    seg_np = np.asarray(
        TF.to_pil_image(seg_01.float().clamp(0, 1).cpu()).resize(target).convert("RGB")
    )
    pred_np = np.asarray(pred_pil.resize(target).convert("RGB"))

    texts = ["ORIGINAL", "SEG MAP", "PREDICTED"]
    columns = [orig_np, seg_np, pred_np]
    if raw_pil is not None:
        texts.append("RAW SEG GEN")
        columns.append(np.asarray(raw_pil.resize(target).convert("RGB")))

    labels = np.concatenate([_seg_label_bar(size_w, t) for t in texts], axis=1)
    panels = np.concatenate(columns, axis=1)
    return Image.fromarray(np.concatenate([labels, panels], axis=0))


def _build_metric_segmenter(cfg, seg_model_path: str, size, device):
    """Build the SEPARATE live segmenter used only to score val/miou_fixed.

    Deliberately separate from model.encoders[0]: at training time that slot is
    torch.nn.Identity (skip_encode=True means no encoder is ever called), and
    it has additionally been through accelerator.prepare(), so reaching into it
    would mean unwrapping a possibly-DDP-wrapped module to call a method it
    does not even have. Building our own instance sidesteps both problems.

    Returns None (never raises) when disabled or unbuildable -- mIoU is a
    monitoring nicety, and failing to load it must never take down a training
    run that is otherwise fine. FAILS CLOSED: no segmenter -> no mIoU, rather
    than a fail-open guess that would throw mid-checkpoint and cost the whole
    monitoring image set for that checkpoint.
    """
    if not cfg.get("val_miou", False):
        return None
    try:
        from src.encoders.seg_encoder import SegmentationEncoder

        enc = SegmentationEncoder(
            size=size,
            model=seg_model_path,
            local_files_only=bool(cfg.get("local_files_only", True)),
        )
        enc = enc.to(device).eval()
        print(f"[val_miou] metric segmenter loaded on {device}: {seg_model_path}")
        return enc
    except Exception as e:
        print(f"[val_miou] WARNING: could not build the metric segmenter -- "
              f"mIoU will be skipped, training continues. Reason: {e}")
        return None


def _save_checkpoint_segmentation_images(
    model, val_dataset, idxs, kinds, n_loras, cfg, cfg_mask, device, out_dir,
    include_empty, metric_segmenter, num_inference_steps,
):
    """Generate + save this checkpoint's monitoring images into out_dir (the
    same folder as its weights). Returns (prompts, [np images], metrics dict).

    Metrics are computed on FIXED scenes only. Fresh scenes change every
    checkpoint, so including them would mix scene difficulty into the trend and
    make two checkpoints incomparable -- the opposite of what the metric is for.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    prompts, images = [], []
    psnrs, ssims, mious = [], [], []

    # (width, height). model.sample() MUST get these explicitly: the underlying
    # diffusers pipeline otherwise defaults to a SQUARE size derived from the
    # UNet config, silently mismatching a non-square conditioning map.
    size_w, size_h = normalize_size(cfg.size)
    # The dataset's own palette -- taking it from there (rather than importing a
    # fresh one) guarantees it is the exact table the maps were colourised with.
    palette = val_dataset.palette.to(device)

    for n, (idx, kind) in enumerate(zip(idxs, kinds)):
        item = val_dataset[idx]
        # classid conditioning (this branch's default, see configs/data/
        # segformer_jsonl.yaml): [H,W] Long, values 1..19 (0=NULL/padding_idx,
        # src/data/local_seg.py._load_seg_ids). NOT an RGB colour map -- that
        # only exists as a display-only copy below, never fed to the model.
        seg = item["seg"].unsqueeze(0).to(device)  # [1,H,W] Long, 1..19
        cs = [seg] * n_loras
        prompt = cfg.prompt if cfg.get("prompt") else item["caption"]
        # Colourised copy for the SEG MAP display panel only -- seg_colorize_ids
        # expects raw 0..num_classes-1 ids, so undo the dataset's own +1 NULL
        # shift first (same convention _load_seg_ids documents).
        seg_display = seg_colorize_ids((seg[0] - 1).clamp(min=0).cpu(), palette.cpu())[0]  # [3,H,W] in [0,1]

        # Fixed per-scene seed: the ONLY thing that differs between the same
        # scene at two checkpoints is the weights, so a visible change is a real
        # change rather than a different noise draw.
        #
        # Timing logged around every call: an earlier smoke test on this
        # codebase's grounded_sam branch sat silent for 26+ minutes during
        # exactly this kind of call with no way to tell "slow" from "stuck" --
        # the tqdm bar never reaches the log file, only real logger.info calls
        # do. Same fix applied here preventively, not after hitting the same
        # problem twice.
        _t0 = time.time()
        pred = model.sample(
            prompt=[prompt], num_images_per_prompt=1, cs=cs,
            generator=torch.Generator(device=device).manual_seed(cfg.seed),
            cfg_mask=cfg_mask, skip_encode=True,
            height=size_h, width=size_w, num_inference_steps=num_inference_steps,
        )[0]
        _module_logger.info(
            f"[seg grid] scene {n + 1}/{len(idxs)} ({kind}) pred sample "
            f"done in {time.time() - _t0:.1f}s"
        )

        raw = None
        if include_empty:
            _t0 = time.time()
            raw = model.sample(
                prompt=[""], num_images_per_prompt=1, cs=cs,
                generator=torch.Generator(device=device).manual_seed(cfg.seed),
                cfg_mask=cfg_mask, skip_encode=True,
                height=size_h, width=size_w, num_inference_steps=num_inference_steps,
            )[0]
            _module_logger.info(
                f"[seg grid] scene {n + 1}/{len(idxs)} ({kind}) raw sample "
                f"done in {time.time() - _t0:.1f}s"
            )

        img = _seg_scene_image(item["jpg"], seg_display, pred, cfg.size, raw_pil=raw)
        img.save(out_dir / f"sample_{n:02d}_{kind}.jpg", quality=95)
        prompts.append(prompt)
        images.append(np.asarray(img))

        if kind != "fixed":
            continue

        orig_u8 = np.asarray(
            TF.to_pil_image(((item["jpg"].float() + 1) / 2).clamp(0, 1).cpu())
            .resize((size_w, size_h)).convert("RGB")
        )
        pred_u8 = np.asarray(pred.resize((size_w, size_h)).convert("RGB"))
        p_, s_ = compute_psnr_ssim(orig_u8, pred_u8)
        psnrs.append(p_)
        ssims.append(s_)

        # Controllability: the structure the model was TOLD to follow (the
        # conditioning ids themselves, +1-shift undone -- a plain index shift,
        # no colormap inversion needed since this conditioning was never
        # colourised to begin with; seg_ids_from_colormap would crash here on
        # the shape mismatch, it expects [3,H,W] RGB and seg[0] is [H,W] Long
        # ids) vs. the structure it actually produced (the metric segmenter
        # re-run on the generated image). Skipped entirely when no segmenter
        # was built.
        if metric_segmenter is not None:
            target_ids = (seg[0] - 1).clamp(min=0).cpu()                 # [H,W], 0..18
            gen_t = (
                TF.to_tensor(pred.resize((size_w, size_h)).convert("RGB"))
                .unsqueeze(0).to(device) * 2.0 - 1.0
            )                                                            # [1,3,H,W] in [-1,1]
            pred_ids = metric_segmenter.label_ids(gen_t)[0].cpu()        # [H,W]
            mious.append(
                compute_miou(pred_ids, target_ids, num_classes=int(palette.shape[0]))
            )

    (out_dir / "prompts.txt").write_text(
        "\n".join(f"[{n}] [{k}] {p}" for n, (k, p) in enumerate(zip(kinds, prompts))),
        encoding="utf-8",
    )

    metrics = {}
    if psnrs:
        metrics["val/psnr_fixed"] = float(np.mean(psnrs))
        metrics["val/ssim_fixed"] = float(np.mean(ssims))
    if mious:
        metrics["val/miou_fixed"] = float(np.mean(mious))
    return prompts, images, metrics


def _segmentation_validation_loss(model, val_dataloader, n_loras, cfg, cfg_mask, accelerator, max_batches):
    """The SAME denoising loss as training, on held-out data, without backprop.

    Gives two things training loss cannot: an overfitting signal (val/loss
    diverging from train/loss), and an objective best_model criterion.

    The RNG snapshot/seed/restore around the loop is what makes the number
    comparable BETWEEN checkpoints: loss here depends heavily on which random
    timesteps and noise get drawn, so without pinning them, two checkpoints
    would differ by their noise draw as much as by their weights. Training's
    own RNG stream is restored afterward so validating does not perturb it.
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
        # skip encoders with zero trainable params (e.g. SegmentationEncoder,
        # which self-freezes AND self-evals its SegFormer-B5 weights at
        # construction) -- .train() doesn't touch requires_grad, but it does
        # re-enable Dropout and let BatchNorm/LayerNorm running stats drift
        # away from their pretrained calibration, degrading its own
        # segmentation quality as training progresses
        if any(p.requires_grad for p in e.parameters()):
            e.train()

    torch.set_rng_state(cpu_rng)
    if cuda_rng is not None:
        torch.cuda.set_rng_state_all(cuda_rng)

    total = accelerator.reduce(total, reduction="sum")
    count = accelerator.reduce(count, reduction="sum")
    return (total / torch.clamp(count, min=1.0)).item()


@hydra.main(config_path="configs", config_name="train")
def main(cfg):
    global stop_training
    if hasattr(signal, "SIGUSR1"):  # POSIX-only; this repo also runs on Windows dev boxes
        signal.signal(signal.SIGUSR1, signal_handler)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    output_path = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)

    # (width, height) -- seg_map_calculations.py's convention. diffusers wants
    # height first, so keep both spellings straight from the start.
    size_w, size_h = normalize_size(cfg.size)

    # seg_map_calculations.py hard-enforces this for the offline precompute
    # step, but `size:` here is a separate config value with nothing keeping
    # it in sync -- an edited/overridden size that isn't divisible by 32
    # would otherwise silently misalign the mapper's output resolution
    # against the UNet's real feature-map resolution instead of failing
    # loudly (SDXL: VAE/8 x 2 UNet halvings/4 = /32 total).
    if size_w % 32 or size_h % 32:
        raise ValueError(
            f"cfg.size = ({size_w}, {size_h}) -- both width and height must be divisible "
            f"by 32 for SDXL (VAE/8 x 2 UNet halvings/4). See seg_map_calculations.py's "
            f"--width/--height guard for the same check on the offline precompute side."
        )

    # hydra.job.chdir=true (configs/train.yaml) moves the process cwd to the
    # run's output dir before this runs -- a relative data.train_jsonl/
    # val_jsonl (the config default) then resolves against THAT dir, not the
    # repo root, and the dataset fails with FileNotFoundError. Unlike
    # base_model_path/seg_model_path/vae_local_path below, this isn't about
    # offline mode, so it's resolved unconditionally, not gated on
    # local_files_only. (Same bug, same fix, as grounded_sam_training.py.)
    _root = get_original_cwd()
    if cfg.get("data", {}).get("train_jsonl") and not os.path.isabs(cfg.data.train_jsonl):
        cfg.data.train_jsonl = os.path.join(_root, cfg.data.train_jsonl)
    if cfg.get("data", {}).get("val_jsonl") and not os.path.isabs(cfg.data.val_jsonl):
        cfg.data.val_jsonl = os.path.join(_root, cfg.data.val_jsonl)

    # ── LOCAL MODELS ONLY when local_files_only=true ────────────────────────
    # Same base_model_path/seg_model_path contract as segformer_inference.py
    # (configs/inference_seg.yaml) -- swap hub ids for local checkpoint
    # folders (see download_models.py) and hard-block any network fallback,
    # so a missing local file fails loudly instead of silently hitting HF.
    if cfg.get("local_files_only", False):
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        if cfg.get("base_model_path"):
            cfg.model.model_name = os.path.join(_root, cfg.base_model_path)
        if cfg.get("seg_model_path") and "model" in cfg.lora.struct.get("encoder", {}):
            cfg.lora.struct.encoder.model = os.path.join(_root, cfg.seg_model_path)
        if cfg.get("vae_local_path") and cfg.model.get("vae_path"):
            cfg.model.vae_path = os.path.join(_root, cfg.vae_local_path)

    # Resolve the metric segmenter's checkpoint path as a plain STRING now,
    # before instantiate() replaces config nodes with built objects. The
    # training encoder slot is Identity and carries no model path, so this has
    # to come from the top-level seg_model_path/seg_model_name keys.
    if cfg.get("local_files_only", False) and cfg.get("seg_model_path"):
        _metric_seg_path = os.path.join(_root, cfg.seg_model_path)
    else:
        _metric_seg_path = cfg.get("seg_model_name", "nvidia/segformer-b5-finetuned-cityscapes-1024-1024")

    # Print the ACTUAL resolved value for every model right before it gets
    # loaded -- mirrors the same diagnostic added to grounded_sam_training.py.
    # Training never had this, which is exactly why "why is it hitting HF
    # Hub instead of my local mount" was hard to diagnose from the log alone:
    # each swap above is gated on `if cfg.get("base_model_path"):` etc, so a
    # missing/empty config key silently no-ops the swap and falls through to
    # a HF Hub id -- this makes that fall-through visible immediately instead
    # of only showing up as a from_pretrained failure with no context.
    _vae_path_display = cfg.model.get("vae_path") or "(none -- using base model's own VAE)"
    print(f"[model] base = {cfg.model.model_name}")
    print(f"[model] vae_path = {_vae_path_display}")
    print(f"[model] seg encoder (metric only, not loaded at train time) = {_metric_seg_path}")
    print(f"[model] local_files_only = {cfg.get('local_files_only', False)}")

    # Rank 0 generates every monitoring image while the other ranks sit blocked
    # in the next gradient all-reduce. With n_grid_images scenes x
    # num_inference_steps each, that stall can exceed NCCL's default 30-minute
    # collective timeout and kill an otherwise healthy multi-GPU job. Raise the
    # timeout to match how long monitoring can actually take. (Never exercised
    # on multi-GPU in this repo yet -- this is preventative, so the first real
    # 4-GPU run doesn't discover it the hard way.)
    _ddp_timeout = int(cfg.get("ddp_timeout_seconds", 5400))
    accelerator = Accelerator(
        project_dir=output_path / "logs",
        log_with="tensorboard",
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        mixed_precision="bf16",
        kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(seconds=_ddp_timeout))],
    )

    # Accelerate falls back to CPU silently when no GPU is visible. Print full
    # diagnostics (main process only -- 4 ranks would interleave 4 copies) and
    # then refuse on EVERY rank, so a single bad rank still fails loudly.
    if accelerator.is_main_process:
        print_gpu_diagnostics()
    if accelerator.device.type != "cuda":
        raise RuntimeError(
            f"accelerate selected device {accelerator.device!r}, not a GPU. Training on CPU "
            f"is never intended for this pipeline. Check `nvidia-smi`, and that this env's "
            f"torch has CUDA (python -c \"import torch; print(torch.version.cuda)\")."
        )

    logger = get_logger(__name__)

    logger.info("==================================")
    logger.info(cfg)
    logger.info(output_path)

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
    if cfg.get("max_train_steps", None) is None:
        max_train_steps = cfg.epochs * num_update_steps_per_epoch
    else:
        max_train_steps = int(cfg.max_train_steps)

    # lr_warmup_steps used to be read from config and then never passed to the
    # scheduler -- a dead key that looked live. Wired up properly here: with
    # lr_scheduler=constant and warmup 0 this is a no-op, so existing configs
    # behave exactly as before, but a non-zero value now actually does
    # something. x num_processes because the scheduler is stepped once per
    # process per optimizer step.
    lr_scheduler = get_scheduler(
        cfg.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=int(cfg.get("lr_warmup_steps", 0)) * accelerator.num_processes,
        num_training_steps=max_train_steps * accelerator.num_processes,
    )

    logger.info(f"Number params Mapper Network(s) {sum(p.numel() for p in mappers_params):,}")
    logger.info(f"Number params Encoder Network(s) {sum(p.numel() for p in encoder_params):,}")
    logger.info(f"Number params all LoRAs(s) {sum(p.numel() for p in model.params_to_optimize):,}")

    logger.info("init trackers")
    if accelerator.is_main_process:
        accelerator.init_trackers("tensorboard")

    logger.info("prepare network")

    prepared = accelerator.prepare(
        *model.mappers,
        *model.encoders,
        model.unet,
        optimizer,
        train_dataloader,
        val_dataloader,
        lr_scheduler,
    )

    mappers = prepared[: len(model.mappers)]
    encoders = prepared[len(model.mappers) : len(model.mappers) + len(model.encoders)]
    (unet, optimizer, train_dataloader, val_dataloader, lr_scheduler) = prepared[
        len(model.mappers) + len(model.encoders) :
    ]
    model.unet = unet
    model.mappers = mappers
    model.encoders = encoders

    # ── Monitoring setup ─────────────────────────────────────────────────────
    n_grid_images = max(2, min(int(cfg.get("n_grid_images", 10)), len(dm.val_dataset)))
    include_empty = bool(cfg.get("grid_include_empty_prompt", False))
    grid_steps = int(cfg.get("val_num_inference_steps", 25))
    n_fixed = n_grid_images // 2
    n_random = n_grid_images - n_fixed

    # No seed -> OS entropy -> a different fixed set each RUN (logged below for
    # reproducibility), but the SAME set at every checkpoint within a run.
    _fixed_val_idxs = random.Random().sample(range(len(dm.val_dataset)), n_fixed)

    # Main process only: it is the only rank that generates monitoring images,
    # so the metric segmenter's VRAM is paid on ONE device, not all four.
    metric_segmenter = (
        _build_metric_segmenter(cfg, _metric_seg_path, cfg.size, accelerator.device)
        if accelerator.is_main_process else None
    )

    if accelerator.is_main_process:
        tb_dir = output_path / "logs" / "tensorboard"
        logger.info("")
        logger.info("=" * 64)
        logger.info("  PIPELINE   :  SEGMENTATION  (SegFormer precomputed maps -> SDXL)")
        logger.info(f"  size       :  {size_w}x{size_h}  resize_mode={cfg.get('resize_mode', 'aspect')}")
        logger.info(f"  Output     :  {output_path}")
        logger.info(f"  TensorBoard:  tensorboard --logdir \"{tb_dir}\"")
        logger.info(f"  Train      :  {len(dm.train_dataset):,} images  |  Val: {len(dm.val_dataset):,} images")
        logger.info(f"  Schedule   :  {cfg.epochs} epochs x {num_update_steps_per_epoch} steps/epoch, "
                    f"capped at max_train_steps={max_train_steps}")
        logger.info(f"  LR         :  {cfg.learning_rate}  scheduler={cfg.lr_scheduler}  "
                    f"warmup={int(cfg.get('lr_warmup_steps', 0))} steps")
        logger.info(f"  val_steps  :  {cfg.val_steps}   (val/loss + best_model update)")
        logger.info(f"  ckpt_steps :  {cfg.get('ckpt_steps', cfg.val_steps)}   "
                    f"(weights + {n_grid_images} monitoring images)")
        logger.info(f"  grid       :  {n_fixed} fixed {_fixed_val_idxs} + {n_random} fresh, "
                    f"{grid_steps} inference steps, include_empty={include_empty}")
        logger.info(f"  val_miou   :  {bool(cfg.get('val_miou', False))}"
                    f"{'' if metric_segmenter is not None else '  (no segmenter -> mIoU skipped)'}")
        logger.info("=" * 64)
        logger.info("")

    global_step = 0
    epoch = 0  # bound up-front: the post-loop snapshot references it, and with
               # epochs=0 the loop body would never run and leave it undefined.
    progress_bar = tqdm(
        range(global_step, max_train_steps),
        disable=not accelerator.is_main_process,
    )
    progress_bar.set_description("Steps")

    best_loss = float("inf")
    best_epoch = None
    best_step = None

    def save_seg_ckpt_and_grid(stem, is_best=False, info_lines=None):
        """Save the current weights AND this checkpoint's monitoring images
        together, into the same folder. Main process only.

        Image generation is best-effort: any failure is logged but must never
        prevent the weights from being written -- the weights are the thing you
        cannot regenerate later.
        """
        if not accelerator.is_main_process:
            return

        _t_ckpt0 = time.time()
        ckpt_dir = (output_path / "best_model") if is_best else (output_path / stem)
        save_checkpoint(
            model.get_lora_state_dict(accelerator.unwrap_model(unet)),
            [accelerator.unwrap_model(m).state_dict() for m in mappers],
            None,
            ckpt_dir,
        )
        logger.info(f"[seg grid] {stem}: weights saved in {time.time() - _t_ckpt0:.1f}s -- generating monitoring images now")

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
                prompts, images, metrics = _save_checkpoint_segmentation_images(
                    model, dm.val_dataset, idxs, kinds, n_loras, cfg, cfg_mask,
                    accelerator.device, ckpt_dir, include_empty, metric_segmenter,
                    grid_steps,
                )

            if is_best:
                metric_lines = [f"{k}: {v:.4f}" for k, v in metrics.items()]
                (ckpt_dir / "info.txt").write_text(
                    "\n".join((info_lines or []) + metric_lines), encoding="utf-8"
                )

            for tracker in accelerator.trackers:
                if tracker.name == "tensorboard":
                    for n, img in enumerate(images):
                        tracker.writer.add_image(
                            f"val/sample_{n:02d}", img, global_step, dataformats="HWC"
                        )
                    tracker.writer.add_text("val/prompts", " | ".join(prompts), global_step)
                    for k, v in metrics.items():
                        tracker.writer.add_scalar(k, v, global_step)

            if metrics:
                logger.info(f"[seg metric] {stem}: " + "  ".join(f"{k}={v:.4f}" for k, v in metrics.items()))
            logger.info(f"[seg grid] {stem}: {len(prompts)} scene images -> {ckpt_dir}")

        except Exception as e:
            print("!!!!!!!!!!!!!!!!!!!")
            print("ERROR generating checkpoint monitoring images (weights ARE saved)")
            print(e)
            print(traceback.format_exc())
            print("!!!!!!!!!!!!!!!!!!!")
        finally:
            unet.train()
            for m in mappers:
                m.train()
            for enc in encoders:
                # see _segmentation_validation_loss: skip encoders with no
                # trainable params so a self-frozen SegmentationEncoder
                # stays in eval mode
                if any(p.requires_grad for p in enc.parameters()):
                    enc.train()

    def do_segmentation_validation(label, epoch_num, epoch_frac):
        """The CHEAP half, run every val_steps: val/loss, plus a best_model
        update when it improves. Deliberately does NOT write a periodic
        checkpoint -- that is ckpt_steps' job, so the expensive save can run on
        its own (slower) cadence."""
        nonlocal best_loss, best_epoch, best_step
        val_loss = _segmentation_validation_loss(
            model, val_dataloader, n_loras, cfg, cfg_mask,
            accelerator, int(cfg.get("val_batches", 4)),
        )
        accelerator.log({"val/loss": val_loss}, step=global_step)
        if accelerator.is_main_process:
            since = "" if best_epoch is None else (
                f"  | best: epoch{best_epoch} step{best_step} ({best_loss:.6f})"
                f"  | {epoch_num - best_epoch} epoch(s) since improvement"
            )
            logger.info(f"[seg val] {label}: val/loss = {val_loss:.6f}{since}")
        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch_num
            best_step = global_step
            info = [
                f"from:        {label}",
                f"epoch:       {epoch_num}",
                f"epoch_frac:  {epoch_frac:.2f}",
                f"global_step: {global_step}",
                f"val/loss:    {val_loss:.6f}",
                f"timestamp:   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                f"resize_mode: {cfg.get('resize_mode', 'aspect')}",
            ]
            save_seg_ckpt_and_grid("best_model", is_best=True, info_lines=info)
            if accelerator.is_main_process:
                logger.info(f"*** NEW BEST *** epoch {epoch_num} ({epoch_frac:.2f}), "
                            f"step {global_step}, val/loss={val_loss:.6f}")

    ckpt_steps = int(cfg.get("ckpt_steps", cfg.val_steps))
    log_every = int(cfg.get("log_every_steps", 50))

    logger.info("start training")
    for epoch in range(cfg.epochs):
        logger.info(f"Epoch {epoch + 1}/{cfg.epochs} started  (global_step={global_step})")
        unet.train()
        for m in mappers:
            m.train()
        for enc in encoders:
            # see _segmentation_validation_loss: skip encoders with no
            # trainable params so a self-frozen SegmentationEncoder stays
            # in eval mode
            if any(p.requires_grad for p in enc.parameters()):
                enc.train()

        for step, batch in enumerate(train_dataloader):
            _grad_norm = None  # only set on true optimizer steps (sync_gradients)

            with accelerator.accumulate(unet, *mappers, *encoders):
                imgs = batch["jpg"]
                imgs = imgs.to(accelerator.device)
                imgs = imgs.clip(-1.0, 1.0)
                B = imgs.shape[0]

                seg = batch["seg"].to(accelerator.device)
                cs = [seg] * n_loras  # precomputed conditioning map, not the training image itself

                if cfg.get("prompt", None) is not None:
                    prompts = [cfg.prompt] * B
                else:
                    prompts = batch["caption"]

                model_pred, loss, x0, _ = model.forward_easy(
                    imgs,
                    prompts,
                    cs,
                    # NOTE: all-True here means src/model.py's structure dropout
                    # (c_dropout, 5%) is ALWAYS active during training, whatever
                    # lora.struct.cfg says -- that flag only controls the
                    # guidance branch at SAMPLING time. Left as-is deliberately:
                    # conditioning dropout during training is the intended
                    # behaviour, and changing it would silently alter training
                    # dynamics for every existing config.
                    cfg_mask=[True for _ in cfg_mask],
                    skip_encode=True,  # cs is already a finished conditioning map, don't re-encode it
                    batch=batch,
                )

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    all_params = [p for g in optimizer.param_groups for p in g["params"]]
                    # Returns the total norm BEFORE clipping -- that is the
                    # diagnostic value. Healthy: starts near the clip threshold,
                    # settles down. Red flag: repeated spikes, or pinned high.
                    _grad_norm = float(accelerator.clip_grad_norm_(all_params, max_norm=1.0))

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            loss_val = loss.detach().item()
            lr_val = lr_scheduler.get_last_lr()[0]
            epoch_frac = epoch + step / max(len(train_dataloader), 1)
            log_dict = {"train/loss": loss_val, "train/lr": lr_val, "train/epoch": epoch_frac}
            if _grad_norm is not None:
                log_dict["train/grad_norm"] = _grad_norm
            progress_bar.set_postfix(
                loss=loss_val, lr=f"{lr_val:.2e}", gnorm=f"{_grad_norm or 0:.3f}", refresh=False
            )
            accelerator.log(log_dict, step=global_step)

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if global_step % log_every == 0:
                    logger.info(
                        f"[train] step {global_step}/{max_train_steps}  loss={loss_val:.4f}  "
                        f"lr={lr_val:.2e}  "
                        f"grad_norm={_grad_norm if _grad_norm is not None else float('nan'):.3f}  "
                        f"epoch={epoch_frac:.2f}"
                    )

                if global_step >= max_train_steps:
                    # max_train_steps only sized the tqdm bar upstream -- nothing
                    # actually stopped the loop at that count. Reuse stop_training
                    # so the final validation/checkpoint still run before breaking.
                    stop_training = True

                # val_steps (cheap: loss only) and ckpt_steps (heavy: weights +
                # N generated images) are INDEPENDENT cadences on purpose.
                if global_step % cfg.val_steps == 0 or stop_training:
                    do_segmentation_validation(f"step{global_step}", epoch + 1, epoch_frac)
                if global_step % ckpt_steps == 0 or stop_training:
                    save_seg_ckpt_and_grid(f"checkpoint-epoch{epoch + 1}/step{global_step}")

            if stop_training:
                break

        if stop_training:
            break

        # END OF EPOCH: always validate and always snapshot this epoch.
        do_segmentation_validation(f"epoch{epoch + 1}", epoch + 1, float(epoch + 1))
        save_seg_ckpt_and_grid(f"checkpoint-epoch{epoch + 1}/epoch_end")

        patience = int(cfg.get("early_stop_patience", 0))
        if patience > 0 and best_epoch is not None:
            epochs_since_best = (epoch + 1) - best_epoch
            if accelerator.is_main_process:
                logger.info(f"[early-stop] {epochs_since_best} epoch(s) since best "
                            f"(epoch {best_epoch}, val/loss={best_loss:.6f}); patience={patience}")
            if epochs_since_best >= patience:
                if accelerator.is_main_process:
                    logger.info(f"[early-stop] no improvement for {patience} epoch(s) -- stopping. "
                                f"Best: epoch {best_epoch}, step {best_step}, val/loss={best_loss:.6f}")
                break

    # Final snapshot ONLY when the loop was cut short (max_train_steps hit,
    # SIGUSR1, or early stop). On a clean finish the end-of-epoch block above
    # already saved this exact state, and repeating it would regenerate every
    # monitoring image for nothing and overwrite them.
    if stop_training:
        accelerator.wait_for_everyone()
        # is_main_process guard lives inside save_seg_ckpt_and_grid -- the old
        # bare save_checkpoint() here ran on EVERY rank, so a 4-GPU job had four
        # processes writing the same files concurrently.
        save_seg_ckpt_and_grid(f"checkpoint-epoch{epoch + 1}/final_step{global_step}")

    accelerator.end_training()


if __name__ == "__main__":
    main()
