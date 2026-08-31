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
pipeline, config.json for a bare transformers model), it is SKIPPED, not
re-downloaded. Safe to re-run this script any time you add a new model to
the list below; only the new one actually downloads. Pass --force to
ignore this and re-download everything anyway.

Models downloaded (verified against what this branch's configs actually
reference -- grep across configs/*.yaml and configs/experiment/*.yaml --
not a generic list, and not segformer branch's list, which needs a
different segmentation encoder entirely and none of grounded_sam's two):

  1. stable-diffusion-xl-base-1.0 -- SDXL base, the training base for the
                                      segmentation-conditioning path (see
                                      src/model.py's SDXL class -- 3 UNet
                                      stages vs SD1.5's 4, needs H/W divisible
                                      by 32 not 64, fits the real 1280x800
                                      dataset natively). base_model_path in
                                      configs/experiment/train_grounded_sam*.yaml.
  2. sdxl-vae-fp16-fix             -- SDXL's own baked-in VAE is numerically
                                      unstable in fp16 (overflows to
                                      black/NaN images); this community fix
                                      (madebyollin/sdxl-vae-fp16-fix) is
                                      required, not optional. vae_local_path
                                      in the same configs as above.
  3. grounding-dino-tiny           -- open-vocabulary box detector, first
                                      stage of src/encoders/
                                      grounded_sam_encoder.py's
                                      GroundedSamEncoder -- takes an image +
                                      a text list of class names (this
                                      branch's own 28-class CARLA
                                      vocabulary), returns boxes.
  4. sam-vit-base                  -- box -> precise mask, second stage of
                                      the same encoder. The two are
                                      composited into the dense raw-class-ID
                                      map this pipeline saves (see
                                      grounded_sam_map_calculations.py).
  5. qwen2-vl-7b-instruct          -- offline prompt-enrichment captioner
                                      used by enrich_prompts.py. NOT BLIP --
                                      enrich_prompts.py's own docstring
                                      explains why (BLIP is a small,
                                      non-instructable captioner; Qwen2-VL
                                      is a real instructable VLM). This was
                                      BLIP here until enrich_prompts.py was
                                      upgraded to Qwen2-VL and this script
                                      wasn't updated to match -- fixed.

Deliberately NOT downloaded here (segformer branch's own segmentation
encoder, SegFormer-b5-Cityscapes, plus the original LoRAdapter repo's SD1.5
depth/style leftovers -- stable-diffusion-v1-5, epiCRealism, sd-vae-ft-mse,
MiDaS, TAESD, and BLIP -- superseded by Qwen2-VL above): grepped across
every config and script on THIS branch and none of them reference any of
these. This branch's own encoder is Grounding DINO + SAM, not SegFormer --
the two branches use different segmentation sources by design (see the
project decision log). If you need one of the SD1.5-era experiments
(configs/experiment/train_struct_sd15.yaml etc.), download that model
yourself the same way this script does
(from_pretrained(...).save_pretrained(...)) -- deliberately not maintained
here so this script stays truthful about what THIS branch's active
pipeline needs.

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
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection, SamModel, SamProcessor, Qwen2VLForConditionalGeneration

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
      "config.json" for a bare single model (the VAE, DINO, SAM) --
      diffusers pipelines don't write a config.json at their own root,
      only inside subfolders, so the two marker files are how this
      distinguishes "a whole pipeline was saved here" from "just one
      component."
    """
    if FORCE:
        return False
    return os.path.isfile(os.path.join(local_path, marker))


# ── 1. Stable Diffusion XL base (the training base for this branch) ──────── #
banner("1/5  Stable Diffusion XL base 1.0")
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
banner("2/5  SDXL VAE (fp16-fix)")
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


# ── 3. Grounding DINO tiny (open-vocabulary box detector) ────────────────── #
#
# First stage of this branch's encoder (src/encoders/grounded_sam_encoder.py):
# takes an image + a text list of class names, returns boxes. SAM (below)
# turns each box into a precise mask; the two are composited into the same
# dense raw-class-ID map convention the segformer branch uses.
banner("3/5  Grounding DINO tiny (box detector)")
dino_path = os.path.join(LOCAL_MODEL_DIR, "grounding-dino-tiny")
if already_downloaded(dino_path, marker="config.json"):
    print(f"  Already present -> {dino_path}  (skipped; pass --force to re-download)")
else:
    dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(
        "IDEA-Research/grounding-dino-tiny",
        token=HF_TOKEN or None,
    )
    dino_processor = AutoProcessor.from_pretrained(
        "IDEA-Research/grounding-dino-tiny",
        token=HF_TOKEN or None,
    )
    dino_model.save_pretrained(dino_path)
    dino_processor.save_pretrained(dino_path)
    print(f"  Saved -> {dino_path}")


# ── 4. SAM ViT-base (box -> precise mask) ─────────────────────────────────── #
banner("4/5  SAM ViT-base (mask segmenter)")
sam_path = os.path.join(LOCAL_MODEL_DIR, "sam-vit-base")
if already_downloaded(sam_path, marker="config.json"):
    print(f"  Already present -> {sam_path}  (skipped; pass --force to re-download)")
else:
    sam_model = SamModel.from_pretrained(
        "facebook/sam-vit-base",
        token=HF_TOKEN or None,
    )
    sam_processor = SamProcessor.from_pretrained(
        "facebook/sam-vit-base",
        token=HF_TOKEN or None,
    )
    sam_model.save_pretrained(sam_path)
    sam_processor.save_pretrained(sam_path)
    print(f"  Saved -> {sam_path}")


# ── 5. Qwen2-VL-7B-Instruct (offline prompt enrichment, enrich_prompts.py) ── #
# NOT BLIP -- enrich_prompts.py was upgraded to Qwen2-VL (a real instructable
# VLM, unlike BLIP's small non-instructable captioner) and this script must
# download what that script actually loads by default, or a fresh offline
# setup would have the wrong model on disk.
banner("5/5  Qwen2-VL-7B-Instruct (prompt enrichment captioner)")
qwen_path = os.path.join(LOCAL_MODEL_DIR, "qwen2-vl-7b-instruct")
if already_downloaded(qwen_path, marker="config.json"):
    print(f"  Already present -> {qwen_path}  (skipped; pass --force to re-download)")
else:
    qwen_model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-7B-Instruct",
        token=HF_TOKEN or None,
    )
    qwen_model.save_pretrained(qwen_path)
    AutoProcessor.from_pretrained(
        "Qwen/Qwen2-VL-7B-Instruct",
        token=HF_TOKEN or None,
    ).save_pretrained(qwen_path)
    print(f"  Saved -> {qwen_path}")


# ── Summary ──────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("  All models ready (downloaded now, or already present and skipped).")
print(f"  Location: {os.path.abspath(LOCAL_MODEL_DIR)}/")
print()
print("  Folder structure:")
for name, marker in [
    ("stable-diffusion-xl-base-1.0", "model_index.json"),
    ("sdxl-vae-fp16-fix", "config.json"),
    ("grounding-dino-tiny", "config.json"),
    ("sam-vit-base", "config.json"),
    ("qwen2-vl-7b-instruct", "config.json"),
]:
    path = os.path.join(LOCAL_MODEL_DIR, name)
    status = "OK" if os.path.isfile(os.path.join(path, marker)) else "MISSING"
    print(f"    [{status}]  {name}")
print()
print("  Next step: set local_files_only: true and point base_model_path /")
print("  vae_local_path at these folders in your experiment config")
print("  (configs/experiment/train_grounded_sam*.yaml).")
print(f"{'='*60}\n")
