"""
grounded_sam_compute_fid.py
-----------------------------
Compute FID (Frechet Inception Distance) between a set of original images
and a set of this model's generated images, to score overall image realism/
quality -- a different question from mIoU (which scores whether generation
FOLLOWS the segmentation it was given; FID scores whether it LOOKS real).

QUICK COMMANDS (run from repo root with conda loradapter env active):
  # --- Compute FID from a jsonl of original/generated image pairs ---
  python grounded_sam_compute_fid.py --jsonl my_pairs.jsonl

  # --- Custom key names, CPU-only, offline (pre-downloaded weights) ---
  python grounded_sam_compute_fid.py --jsonl my_pairs.jsonl \\
      --orig_key original_image_path --gen_key generated_image_path \\
      --device cpu \\
      --feature_extractor_weights_path checkpoints/local_models/inception-2015-12-05.pth

INPUT: a jsonl with one object per line, each holding two keys pointing at
plain RGB image files (not segmentation maps):
  {"original_image_path": "...", "generated_image_path": "..."}
Default key names are original_image_path/generated_image_path; override
with --orig_key/--gen_key if your manifest uses different names.

IMPORTANT -- FID IS NOT A PAIRWISE METRIC. It compares two whole
DISTRIBUTIONS of images (fits a Gaussian to Inception features of each
set, measures the distance between the two Gaussians), not row-by-row
pairs. The jsonl's per-row pairing is just a convenient way to collect two
lists of image paths -- which original goes with which generated image
(or whether they even correspond to the same scene) does not affect the
result at all. Meaningful FID needs a reasonably large set on each side
(the original FID paper's own guidance: thousands of images; a handful of
images will give a noisy, not-very-meaningful number -- this script will
still compute one, but don't over-read a small-sample result).

OFFLINE MODELS: unlike this repo's other scripts, FID's InceptionV3
feature extractor does NOT go through Hugging Face / local_files_only --
torch_fidelity downloads it once via torch.hub from a fixed GitHub-releases
URL, cached at ~/.cache/torch/hub/checkpoints/ after the first run. On a
genuinely offline machine, either pre-warm that cache from an
internet-connected machine (run this script there once, then copy that
cache directory over), or download the weights file directly and pass
--feature_extractor_weights_path to skip the download entirely:
  https://github.com/toshas/torch-fidelity/releases/download/v0.2.0/weights-inception-2015-12-05-6726825d.pth

Reuses torch_fidelity (already a project dependency, see requirements.txt)
-- not a reimplementation of Inception feature extraction or the Frechet
distance formula.
"""
import argparse
import json
import sys
from pathlib import Path

import torch
import torch_fidelity
from torch_fidelity.datasets import ImagesPathDataset

PROJECT_ROOT = Path(__file__).resolve().parent


def resolve(path: str) -> str:
    p = Path(path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    if not p.is_file():
        raise FileNotFoundError(f"image not found: {p}")
    return str(p)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jsonl", required=True, help="Manifest with one row per image pair.")
    ap.add_argument("--orig_key", default="original_image_path", help="Key for the original image path.")
    ap.add_argument("--gen_key", default="generated_image_path", help="Key for the generated image path.")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                     help="'cuda' or 'cpu'. Auto-detects by default.")
    ap.add_argument("--feature_extractor_weights_path", default=None,
                     help="Local InceptionV3 weights file, to avoid the torch.hub download "
                          "(see module docstring for the URL and offline setup).")
    args = ap.parse_args()

    orig_paths, gen_paths = [], []
    with open(args.jsonl, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if args.orig_key not in row or args.gen_key not in row:
                print(f"[skip] line {lineno}: missing {args.orig_key!r} or {args.gen_key!r}")
                continue
            try:
                orig_paths.append(resolve(row[args.orig_key]))
                gen_paths.append(resolve(row[args.gen_key]))
            except FileNotFoundError as e:
                print(f"[skip] line {lineno}: {e}")

    if not orig_paths or not gen_paths:
        print(f"No valid image pairs found in {args.jsonl} "
              f"(looked for keys {args.orig_key!r}/{args.gen_key!r})")
        sys.exit(1)

    print(f"original set: {len(orig_paths)} images")
    print(f"generated set: {len(gen_paths)} images")
    if len(orig_paths) < 100 or len(gen_paths) < 100:
        print("WARNING: FID is a distributional metric -- a set this small will give a "
              "noisy, not very meaningful number. See module docstring.")

    dataset1 = ImagesPathDataset(orig_paths)
    dataset2 = ImagesPathDataset(gen_paths)

    metrics = torch_fidelity.calculate_metrics(
        input1=dataset1,
        input2=dataset2,
        cuda=(args.device == "cuda"),
        batch_size=args.batch_size,
        fid=True,
        verbose=True,
        feature_extractor_weights_path=args.feature_extractor_weights_path,
    )

    fid = metrics[torch_fidelity.KEY_METRIC_FID]
    print()
    print(f"FID: {fid:.4f}")


if __name__ == "__main__":
    main()
