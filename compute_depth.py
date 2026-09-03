"""
compute_depth.py
-----------------
Offline precompute of MiDaS depth maps for the `depth` branch.

RULE 0 (DEPTH-BRANCH-architecture.md): this branch derives from the original
repo and the paper only. The depth model here is exactly the original repo's
own choice (Intel/dpt-hybrid-midas, src/annotators/midas.py's
DepthEstimator) -- nothing is copied from the segmentation branches'
encoders. This script's *shape* (CLI, output manifest, spec.json, a
--verify contact sheet) mirrors a repo-wide offline-precompute convention
already used elsewhere in this project, not a segmentation-specific design.

RESIZE POLICY -- fixed, no variations (see DEPTH-BRANCH-architecture.md):
  Working resolution: 1280x768 (W,H), always, for both image and depth.
  Source images must already be 1280x800 (W,H). We do NOT resize the RGB --
  we CROP exactly 16px off the top and 16px off the bottom (never pad,
  never letterbox, never resize). An image that isn't exactly 1280x800
  fails loudly rather than being silently force-resized.

  MiDaS input:  1280x768 -> 384x384, bilinear, antialias=True. SQUARE, unlike
    every other resize in this pipeline -- Intel/dpt-hybrid-midas uses
    DPTViTHybridEmbeddings, a ViT-style patch embedding with position
    embeddings fixed to a square 384x384 grid (config.image_size). The HF
    DPTModel.forward() never threads interpolate_pos_encoding down to the
    embeddings layer (verified against the installed transformers source --
    DPTForDepthEstimation.forward() accepts the kwarg but DPTModel.forward()
    drops it before calling self.embeddings(pixel_values)), so a non-square
    model input hard-fails with "Input image size doesn't match model"
    regardless of what's passed at the call site. This is a real, checked
    constraint of this specific hybrid-ViT checkpoint's embeddings, not the
    same "never square" concern as the RGB/depth-map policy below -- only
    the MODEL'S OWN internal input needs to be square; the source image,
    the final depth map, and the crop policy below are completely
    unaffected and still never square.
  MiDaS output: model resolution -> 1280x768, bicubic, align_corners=False.
  Order: interpolate to 1280x768 FIRST, THEN min/max normalize to [0,1].
  This order is load-bearing (bicubic overshoot at depth edges is absorbed
  by the normalization) -- do not swap it.

  Crops for any categorical signal use NEAREST -- that rule is
  segmentation-specific and does not apply here. Depth is continuous;
  bicubic is correct throughout.

Storage: 1-channel float16 .npy (default) or 16-bit PNG (--format png16).
Never 8-bit -- depth is continuous, and 8-bit banding on a continuous
signal shows up in generated images. Expansion to 3 channels happens at
LOAD time (src/data/local_depth.py), not on disk.

src/annotators/util.py's better_resize is deliberately NOT used anywhere in
this file -- it center-crops to a square and would throw away both
roadsides of a 1280x800 frame (defect #6 in the architecture doc).

Usage:
  python compute_depth.py --jsonl data/dataset/data_full/train.jsonl \\
                           --jsonl data/dataset/data_full/val.jsonl \\
                           --output_dir data/dataset/depth_cache \\
                           --verify 12
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import DPTForDepthEstimation

PROJECT_ROOT = Path(os.path.abspath(__file__)).parent

WORKING_SIZE = (1280, 768)  # (W, H) -- the one working resolution, always
SOURCE_SIZE = (1280, 800)  # (W, H) -- required source size, before crop
CROP_TOP = 16
CROP_BOTTOM = 16
MIDAS_INPUT_SIZE = (384, 384)  # (H, W) for F.interpolate -- SQUARE, see module docstring
MIDAS_MODEL_DEFAULT = "Intel/dpt-hybrid-midas"
MIDAS_LOCAL_DEFAULT = "checkpoints/local_models/dpt-hybrid-midas"


class DepthPrecomputer:
    """Loads the MiDaS model once, exposes a single-image depth function that
    implements the resize policy above exactly. Shared by the CLI batch mode
    below and by inference_depth.py's live-source-image mode, so the two
    scripts can never compute depth two different ways.
    """

    def __init__(self, model_name: str, model_path: str | None, local_files_only: bool, device: str, dtype=torch.float32):
        load_from = model_path if (local_files_only and model_path is not None) else model_name
        self.model = DPTForDepthEstimation.from_pretrained(load_from, local_files_only=local_files_only)
        self.model.to(device, dtype)
        self.model.requires_grad_(False)
        self.model.eval()
        self.device = device
        self.dtype = dtype

    @staticmethod
    def crop_source(img: Image.Image) -> Image.Image:
        """1280x800 -> 1280x768 by cropping 16px off top and bottom. Never resize."""
        w, h = img.size
        if (w, h) != SOURCE_SIZE:
            raise ValueError(
                f"Expected source image at exactly {SOURCE_SIZE[0]}x{SOURCE_SIZE[1]} (WxH), "
                f"got {w}x{h}. This script never resizes the RGB -- fix the source dataset "
                f"or point at a manifest of true {SOURCE_SIZE[0]}x{SOURCE_SIZE[1]} images."
            )
        left, top = 0, CROP_TOP
        right, bottom = w, h - CROP_BOTTOM
        cropped = img.crop((left, top, right, bottom))
        assert cropped.size == WORKING_SIZE, f"crop produced {cropped.size}, expected {WORKING_SIZE}"
        return cropped

    @torch.no_grad()
    def depth_from_tensor(self, imgs_01: torch.Tensor) -> torch.Tensor:
        """imgs_01: [B,3,768,1280] float in [0,1]. Returns [B,1,768,1280] float in [0,1]."""
        assert imgs_01.shape[-2:] == (WORKING_SIZE[1], WORKING_SIZE[0]), imgs_01.shape
        assert imgs_01.min() >= 0.0 and imgs_01.max() <= 1.0

        imgs_01 = imgs_01.to(self.device, self.dtype)

        # MiDaS input: 1280x768 -> 384x384, bilinear, antialias=True. SQUARE --
        # see module docstring for why this one resize must be square.
        model_in = F.interpolate(imgs_01, size=MIDAS_INPUT_SIZE, mode="bilinear", align_corners=False, antialias=True)

        depth = self.model(pixel_values=model_in).predicted_depth  # [B, h, w]
        depth = depth.unsqueeze(1)  # [B,1,h,w]

        # MiDaS output -> 1280x768, bicubic, align_corners=False. Interpolate FIRST.
        depth = F.interpolate(
            depth.float(),
            size=(WORKING_SIZE[1], WORKING_SIZE[0]),
            mode="bicubic",
            align_corners=False,
        )
        assert depth.shape[-2:] == (WORKING_SIZE[1], WORKING_SIZE[0])

        # THEN min/max normalize to [0,1]. Order is load-bearing -- do not swap.
        depth_min = torch.amin(depth, dim=[2, 3], keepdim=True)
        depth_max = torch.amax(depth, dim=[2, 3], keepdim=True)
        depth = (depth - depth_min) / (depth_max - depth_min + 1e-6)

        assert depth.shape[-2:] == (WORKING_SIZE[1], WORKING_SIZE[0])
        return depth.clamp(0.0, 1.0)

    def depth_from_pil(self, img: Image.Image) -> torch.Tensor:
        """Full policy: crop -> [0,1] tensor -> depth. Returns [1,768,1280] float in [0,1]."""
        cropped = self.crop_source(img)
        arr = np.asarray(cropped.convert("RGB"), dtype=np.float32) / 255.0
        t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # [1,3,768,1280]
        depth = self.depth_from_tensor(t)
        return depth[0]  # [1,768,1280]


def save_depth(depth_01: torch.Tensor, out_path: Path, fmt: str):
    """depth_01: [1,H,W] or [H,W] float in [0,1]. Never 8-bit."""
    arr = depth_01.squeeze(0).cpu().numpy() if depth_01.dim() == 3 else depth_01.cpu().numpy()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "npy16":
        np.save(out_path.with_suffix(".npy"), arr.astype(np.float16))
    elif fmt == "png16":
        arr16 = (arr.clip(0.0, 1.0) * 65535.0).round().astype(np.uint16)
        Image.fromarray(arr16, mode="I;16").save(out_path.with_suffix(".png"))
    else:
        raise ValueError(f"unknown format {fmt}")


def load_depth(path: Path) -> np.ndarray:
    """Inverse of save_depth. Returns float32 array in [0,1], shape (H,W)."""
    if path.suffix == ".npy":
        return np.load(path).astype(np.float32)
    elif path.suffix == ".png":
        arr16 = np.array(Image.open(path))
        return arr16.astype(np.float32) / 65535.0
    raise ValueError(f"unknown depth file extension {path.suffix}")


def build_contact_sheet(pairs: list[tuple[Image.Image, np.ndarray]], out_path: Path):
    """pairs: list of (cropped source image, depth [H,W] in [0,1]). Side-by-side
    rows so alignment and both-roadsides-survived can be checked by eye."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(pairs)
    fig, axes = plt.subplots(n, 2, figsize=(2 * 6, n * 3.2))
    if n == 1:
        axes = axes[None, :]
    for i, (img, depth) in enumerate(pairs):
        axes[i, 0].imshow(img)
        axes[i, 0].set_title("source (cropped 1280x768)" if i == 0 else "")
        axes[i, 0].axis("off")
        axes[i, 1].imshow(depth, cmap="inferno", vmin=0, vmax=1)
        axes[i, 1].set_title("depth" if i == 0 else "")
        axes[i, 1].axis("off")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def process_split(jsonl_path: Path, output_dir: Path, precomputer: DepthPrecomputer, fmt: str, batch_size: int):
    entries = [json.loads(l) for l in open(jsonl_path, "r", encoding="utf-8") if l.strip()]
    split_name = jsonl_path.stem  # "train" / "val"
    depth_dir = output_dir / split_name

    seen_stems: dict[str, str] = {}
    out_entries = []
    verify_pairs = []

    i = 0
    while i < len(entries):
        batch = entries[i : i + batch_size]
        imgs, cropped_pils = [], []
        for e in batch:
            src_path = PROJECT_ROOT / e["source"]
            img = Image.open(src_path).convert("RGB")
            cropped = precomputer.crop_source(img)
            cropped_pils.append(cropped)
            arr = np.asarray(cropped, dtype=np.float32) / 255.0
            imgs.append(torch.from_numpy(arr).permute(2, 0, 1))

        batch_t = torch.stack(imgs, dim=0)
        depths = precomputer.depth_from_tensor(batch_t)  # [B,1,768,1280]

        for e, cropped, depth in zip(batch, cropped_pils, depths):
            stem = Path(e["source"]).stem
            if stem in seen_stems:
                raise ValueError(
                    f"stem collision: '{stem}' from {e['source']} collides with "
                    f"an earlier entry from {seen_stems[stem]} -- output naming would "
                    f"silently overwrite. Aborting."
                )
            seen_stems[stem] = e["source"]

            depth_path = depth_dir / f"{stem}.{('npy' if fmt == 'npy16' else 'png')}"
            save_depth(depth, depth_path, fmt)

            out_entries.append(
                {
                    "source": e["source"],
                    "prompt": e.get("prompt", ""),
                    "depth_path": str(depth_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                    "crop_top": CROP_TOP,
                    "crop_bottom": CROP_BOTTOM,
                    "orig_size": list(SOURCE_SIZE),
                    "cropped_size": list(WORKING_SIZE),
                }
            )

            if len(verify_pairs) < 1_000_000:
                verify_pairs.append((cropped, depth.squeeze(0).cpu().numpy(), stem))

        i += batch_size
        print(f"[{split_name}] {min(i, len(entries))}/{len(entries)}")

    manifest_path = output_dir / f"{split_name}.jsonl"
    with open(manifest_path, "w", encoding="utf-8") as f:
        for e in out_entries:
            f.write(json.dumps(e) + "\n")

    print(f"[{split_name}] wrote {len(out_entries)} entries -> {manifest_path}")
    return verify_pairs


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--jsonl", action="append", required=True, help="manifest(s) with 'source'/'prompt' keys; repeatable")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--model", default=MIDAS_MODEL_DEFAULT)
    p.add_argument("--model_path", default=MIDAS_LOCAL_DEFAULT)
    p.add_argument("--local_files_only", type=lambda s: s.lower() != "false", default=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--format", choices=["npy16", "png16"], default="npy16")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--verify", type=int, default=0, help="save an N-image contact sheet for visual alignment check")
    args = p.parse_args()

    # Relative --output_dir (the exact form this script's own usage docstring
    # recommends, e.g. "data/dataset/depth_cache") must be anchored to
    # PROJECT_ROOT here, before use -- process_split()'s depth_path.relative_to
    # (PROJECT_ROOT) call below requires depth_path to be absolute, and a
    # relative output_dir would otherwise make every manifest write crash
    # immediately (verified: this is exactly what happened before this fix).
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    precomputer = DepthPrecomputer(args.model, args.model_path, args.local_files_only, args.device)

    all_verify_pairs = []
    for jsonl_arg in args.jsonl:
        pairs = process_split(Path(jsonl_arg), output_dir, precomputer, args.format, args.batch_size)
        all_verify_pairs.extend(pairs)

    spec = {
        "model": args.model,
        "model_path": args.model_path if args.local_files_only else None,
        "local_files_only": args.local_files_only,
        "working_size_wh": list(WORKING_SIZE),
        "source_size_wh": list(SOURCE_SIZE),
        "crop_top": CROP_TOP,
        "crop_bottom": CROP_BOTTOM,
        "crop_left": 0,
        "crop_right": 0,
        "midas_input_size_hw": list(MIDAS_INPUT_SIZE),
        "midas_input_interpolation": "bilinear_antialias",
        "midas_output_interpolation": "bicubic_align_corners_false",
        "normalization": "per_image_minmax_after_interpolate",
        "dtype": "float16" if args.format == "npy16" else "uint16_png",
        "format": args.format,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "num_images": len(all_verify_pairs),
    }
    with open(output_dir / "spec.json", "w", encoding="utf-8") as f:
        json.dump(spec, f, indent=2)
    print(f"wrote {output_dir / 'spec.json'}")

    if args.verify > 0:
        import random

        sample = random.sample(all_verify_pairs, min(args.verify, len(all_verify_pairs)))
        build_contact_sheet(
            [(img, depth) for img, depth, _ in sample],
            output_dir / "verify_contact_sheet.png",
        )
        print(f"verify: wrote contact sheet for {len(sample)} images -> {output_dir / 'verify_contact_sheet.png'}")
        print("verify: check both roadsides are present and depth aligns with source content.")


if __name__ == "__main__":
    main()
