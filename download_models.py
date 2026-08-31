"""
download_models.py
----------------
FIRST-TIME SETUP: download every model this branch's segmentation pipeline
actually needs and save it to checkpoints/local_models/ for fully offline
training and inference.

Run this ONCE on a machine with internet access, then copy
checkpoints/local_models/ to your training machine and set
local_files_only: true in all configs.

SKIP-EXISTING: each model is checked BEFORE downloading -- if its folder
already has the right marker file (model_index.json for a diffusers
pipeline, config.json for a bare transformers/AutoencoderKL model), it is
SKIPPED, not re-downloaded. Safe to re-run this script any time you add a
new model to the list below; only the new one actually downloads. Pass
--force to ignore this and re-download everything anyway.

Models downloaded (verified against what this branch's configs actually
reference -- grep across configs/*.yaml, configs/experiment/*.yaml, and
every .py script's own default model path -- not the original LoRAdapter
repo's full model list):

  1. stable-diffusion-xl-base-1.0 -- SDXL base, the training base for the
                                      segmentation-conditioning path (see
                                      src/model.py's SDXL class -- 3 UNet
                                      stages vs SD1.5's 4, needs H/W divisible
                                      by 32 not 64, fits the real 1280x800
                                      dataset natively). base_model_path in
                                      configs/experiment/train_seg*.yaml and
                                      configs/inference_seg.yaml.
  2. sdxl-vae-fp16-fix             -- SDXL's own baked-in VAE is numerically
                                      unstable in fp16 (overflows to
                                      black/NaN images); this community fix
                                      (madebyollin/sdxl-vae-fp16-fix) is
                                      required, not optional. vae_local_path
                                      in the same configs as above.
  3. segformer-b5-cityscapes       -- LOCKED at b5, not b0 -- measured mIoU
                                      0.76/0.66 train/val on real Cityscapes
                                      ground truth in this repo (see
                                      check_seg_accuracy.py), and b5 is the
                                      documented highest-accuracy SegFormer
                                      variant. Not loaded at TRAINING time
                                      (skip_encode=True bypasses the live
                                      encoder) -- needed by
                                      seg_map_calculations.py (offline
                                      precompute), segformer_inference.py
                                      (mIoU controllability check), and
                                      segformer_training.py's optional
                                      val_miou metric.

Deliberately NOT downloaded here (leftovers from the original LoRAdapter
repo -- stable-diffusion-v1-5, epiCRealism, sd-vae-ft-mse, MiDaS/
dpt-hybrid-midas, TAESD): grepped across every config and script on this
branch and none of them reference any of these. They belong to the
repo's older SD1.5 depth/style experiments (configs/experiment/
train_struct_sd15.yaml, train_style_sd15.yaml, sample_*.yaml), not this
branch's segmentation pipeline. If you need to run one of those older
experiments, download that model yourself the same way this script does
(from_pretrained(...).save_pretrained(...)) -- deliberately not maintained
here so this script stays truthful about what the ACTIVE pipeline needs.

Usage:
  python download_models.py
  python download_models.py --force        # re-download even if already present

  With a HF token (required for gated models, optional here since all
  models below are public):
    Set HF_TOKEN below, or export HF_TOKEN=your_token before running.
"""

import argparse
import os
from diffusers import StableDiffusionXLPipeline, AutoencoderKL
from transformers import SegformerForSemanticSegmentation, BlipForConditionalGeneration, BlipProcessor

# --- Hugging Face authentication -------------------------------------------- #
# Paste your HF READ token here, or leave empty for public models.
#
# SECURITY: never commit this file to a public repo with a token in it.
# Safer: export HF_TOKEN=hf_xxx in your shell, then use token=os.environ.get("HF_TOKEN")
HF_TOKEN = os.environ.get("HF_TOKEN", "")

_parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
_parser.add_argument("--force", action="store_true",
                      help="Re-download every model even if already present locally.")
FORCE = _parser.parse_args().force

LOCAL_MODEL_DIR = "checkpoints/local_models"
os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)


# Helper: print a section banner
def banner(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def already_downloaded(local_path: str, marker: str = "model_index.json") -> bool:
    """
    True if `local_path` already contains `marker` (i.e. this model was
    already fully saved by a previous run of this script) AND --force was
    not passed. This is the skip-existing check every model below runs
    before doing any network I/O.

    marker: "model_index.json" for a full diffusers pipeline (SDXL);
      "config.json" for a bare single model (the VAE, SegFormer) --
      diffusers pipelines don't write a config.json at their own root,
      only inside subfolders, so the two marker files are how this
      distinguishes "a whole pipeline was saved here" from "just one
      component."
    """
    if FORCE:
        return False
    return os.path.isfile(os.path.join(local_path, marker))


# ── 1. Stable Diffusion XL base (the training base for this branch) ──────── #
banner("1/3  Stable Diffusion XL base 1.0")
sdxl_path = os.path.join(LOCAL_MODEL_DIR, "stable-diffusion-xl-base-1.0")
if already_downloaded(sdxl_path):
    print(f"  Already present -> {sdxl_path}  (skipped; pass --force to re-download)")
else:
    sdxl_pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        token=HF_TOKEN or None,
    )
    sdxl_pipe.save_pretrained(sdxl_path)
    print(f"  Saved -> {sdxl_path}")


# ── 2. SDXL VAE fp16-fix (required, not optional, for SDXL) ──────────────── #
banner("2/3  SDXL VAE (fp16-fix)")
sdxl_vae_path = os.path.join(LOCAL_MODEL_DIR, "sdxl-vae-fp16-fix")
if already_downloaded(sdxl_vae_path, marker="config.json"):
    print(f"  Already present -> {sdxl_vae_path}  (skipped; pass --force to re-download)")
else:
    sdxl_vae = AutoencoderKL.from_pretrained(
        "madebyollin/sdxl-vae-fp16-fix",
        token=HF_TOKEN or None,
    )
    sdxl_vae.save_pretrained(sdxl_vae_path)
    print(f"  Saved -> {sdxl_vae_path}")


# ── 3. SegFormer-b5-Cityscapes (offline precompute + inference mIoU check) ─ #
banner("3/3  SegFormer-b5-Cityscapes (segmentation encoder)")
seg_path = os.path.join(LOCAL_MODEL_DIR, "segformer-b5-cityscapes")
if already_downloaded(seg_path, marker="config.json"):
    print(f"  Already present -> {seg_path}  (skipped; pass --force to re-download)")
else:
    seg_model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/segformer-b5-finetuned-cityscapes-1024-1024",
        token=HF_TOKEN or None,
    )
    seg_model.save_pretrained(seg_path)
    print(f"  Saved -> {seg_path}")


# ── 4. BLIP captioning (offline prompt enrichment, enrich_prompts.py) ────── #
banner("4/4  BLIP image captioning (prompt enrichment)")
blip_path = os.path.join(LOCAL_MODEL_DIR, "blip-image-captioning-large")
if already_downloaded(blip_path, marker="config.json"):
    print(f"  Already present -> {blip_path}  (skipped; pass --force to re-download)")
else:
    blip_model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-large",
        token=HF_TOKEN or None,
    )
    blip_model.save_pretrained(blip_path)
    BlipProcessor.from_pretrained(
        "Salesforce/blip-image-captioning-large",
        token=HF_TOKEN or None,
    ).save_pretrained(blip_path)
    print(f"  Saved -> {blip_path}")


# ── Summary ──────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("  All models ready (downloaded now, or already present and skipped).")
print(f"  Location: {os.path.abspath(LOCAL_MODEL_DIR)}/")
print()
print("  Folder structure:")
for name, marker in [
    ("stable-diffusion-xl-base-1.0", "model_index.json"),
    ("sdxl-vae-fp16-fix", "config.json"),
    ("segformer-b5-cityscapes", "config.json"),
    ("blip-image-captioning-large", "config.json"),
]:
    path = os.path.join(LOCAL_MODEL_DIR, name)
    status = "OK" if os.path.isfile(os.path.join(path, marker)) else "MISSING"
    print(f"    [{status}]  {name}")
print()
print("  Next step: set local_files_only: true and point base_model_path /")
print("  vae_local_path / seg_model_path at these folders in your experiment")
print("  config (configs/experiment/train_seg*.yaml, configs/inference_seg.yaml).")
print(f"{'='*60}\n")
