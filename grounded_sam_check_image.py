"""
grounded_sam_check_image.py
----------------------------
Pre-flight check for ONE image, before running it through
grounded_sam_map_calculations.py's expensive Grounding DINO + SAM pipeline.
Answers: "is this image worth computing a segmentation map for?"

TWO TIERS, cheapest first:

  1. File/quality checks (no GPU, near-instant): does it actually open and
     decode, is it a real RGB-convertible image, is it not a blank/corrupt
     frame (flat pixel-variance heuristic), how far is its aspect ratio from
     this branch's 1280x800 target (a big mismatch means heavy distortion
     once seg_map_calculations resizes it).

  2. --check_detections (opt-in, needs a GPU, but meaningfully cheaper than
     the full pipeline): loads ONLY Grounding DINO -- deliberately never
     loads SAM at all, which is the expensive per-box mask-decode step this
     check exists to let you skip paying for on a bad image. Runs detection
     against this branch's own 28-class CARLA vocabulary (imported directly
     from src/encoders/grounded_sam_encoder.py -- same CARLA_CLASSES,
     _match_class, box/text thresholds the real pipeline uses, not a second
     copy that could drift). Zero or very few classes detected means
     grounded_sam_map_calculations.py would produce an empty or near-empty
     (mostly IGNORE_ID) segmentation map for this image -- not worth the
     full DINO+SAM cost.

Usage:
    # Fast, no GPU -- file/quality checks only:
    python grounded_sam_check_image.py --image path/to/frame.png

    # Also runs DINO-only detection (needs a GPU + local checkpoint):
    python grounded_sam_check_image.py --image path/to/frame.png --check_detections
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).parent

TARGET_ASPECT = 1280 / 800  # this branch's real dataset's native ratio
ASPECT_WARN_TOLERANCE = 0.15  # relative difference before flagging distortion risk
MIN_SIDE_PX = 256  # below this, DINO/SAM quality degrades noticeably
BLANK_STD_THRESHOLD = 3.0  # per-channel pixel std below this ~= flat/blank frame


def check_file_and_quality(image_path: Path) -> list[tuple[str, str]]:
    """Cheap, no-GPU checks. Returns [(level, message), ...] -- level is
    'FAIL' (blocking), 'WARN' (usable but flagged), or 'OK'."""
    findings = []

    if not image_path.exists():
        return [("FAIL", f"File does not exist: {image_path}")]
    size_bytes = image_path.stat().st_size
    if size_bytes == 0:
        return [("FAIL", "File is 0 bytes -- broken symlink or failed copy.")]

    try:
        img = Image.open(image_path)
        img.load()  # force full decode now, not lazily later -- catches truncated files
    except Exception as e:
        return [("FAIL", f"Could not open/decode image: {type(e).__name__}: {e}")]

    findings.append(("OK", f"Opens fine: mode={img.mode}, size={img.size[0]}x{img.size[1]}, {size_bytes:,} bytes"))

    try:
        rgb = img.convert("RGB")
    except Exception as e:
        return findings + [("FAIL", f"Could not convert to RGB: {type(e).__name__}: {e}")]

    w, h = rgb.size
    if min(w, h) < MIN_SIDE_PX:
        findings.append(("WARN", f"Small image ({w}x{h}) -- DINO/SAM detection quality "
                                  f"degrades noticeably below ~{MIN_SIDE_PX}px on the short side."))

    aspect = w / h
    rel_diff = abs(aspect - TARGET_ASPECT) / TARGET_ASPECT
    if rel_diff > ASPECT_WARN_TOLERANCE:
        findings.append(("WARN", f"Aspect ratio {aspect:.2f} ({w}x{h}) differs from this branch's "
                                  f"1280x800 target ({TARGET_ASPECT:.2f}) by {rel_diff*100:.0f}% -- "
                                  f"expect visible distortion once seg_map_calculations resizes it."))
    else:
        findings.append(("OK", f"Aspect ratio {aspect:.2f} close to the 1280x800 target ({TARGET_ASPECT:.2f})."))

    arr = np.asarray(rgb).astype("float32")
    per_channel_std = arr.reshape(-1, 3).std(axis=0)
    if per_channel_std.max() < BLANK_STD_THRESHOLD:
        findings.append(("FAIL", f"Looks blank/flat (per-channel pixel std {per_channel_std.tolist()}, "
                                  f"all below {BLANK_STD_THRESHOLD}) -- likely a corrupt render or a "
                                  f"solid-colour frame, no real content for DINO to detect."))
    else:
        findings.append(("OK", f"Has real pixel variance (per-channel std {per_channel_std.round(1).tolist()})."))

    return findings


def check_detections(image_path: Path, local_files_only: bool, dino_model_path: str) -> list[tuple[str, str]]:
    """Opt-in, GPU-based: Grounding DINO ONLY, never loads SAM (the
    expensive part this check exists to let you skip on a bad image)."""
    import os
    import torch
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
    from src.encoders.grounded_sam_encoder import CARLA_CLASSES, _match_class
    from src.utils import resolve_device

    findings = []
    device = resolve_device(None)

    if local_files_only:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        model_id = str(PROJECT_ROOT / dino_model_path)
    else:
        model_id = "IDEA-Research/grounding-dino-tiny"

    processor = AutoProcessor.from_pretrained(model_id, local_files_only=local_files_only)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id, local_files_only=local_files_only)
    model.to(device).eval()

    img = Image.open(image_path).convert("RGB")
    dino_text = ". ".join(CARLA_CLASSES) + "."
    box_threshold = text_threshold = 0.15  # same defaults GroundedSamEncoder uses -- keep in sync

    inputs = processor(images=img, text=dino_text, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**inputs)
    results = processor.post_process_grounded_object_detection(
        out, threshold=box_threshold, text_threshold=text_threshold, target_sizes=[(img.size[1], img.size[0])],
    )[0]

    raw_labels = results.get("text_labels", results.get("labels"))
    n_boxes = len(results["boxes"])
    matched = sorted({c for lbl in raw_labels if (c := _match_class(str(lbl))) is not None})

    if n_boxes == 0:
        findings.append(("FAIL", "Grounding DINO detected NOTHING at all -- this image would produce "
                                  "an all-IGNORE segmentation map, no conditioning signal whatsoever."))
    elif len(matched) <= 2:
        findings.append(("WARN", f"Only {len(matched)} class(es) detected ({n_boxes} box(es) total): "
                                  f"{matched} -- thin, sparse conditioning signal."))
    else:
        findings.append(("OK", f"{len(matched)} classes detected ({n_boxes} box(es) total): {matched}"))

    return findings


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--image", required=True, help="Path to the PNG (or any Pillow-readable) image to check.")
    p.add_argument("--check_detections", action="store_true",
                   help="Also run Grounding-DINO-only detection (needs a GPU + local checkpoint). "
                        "Skips loading SAM entirely -- the expensive part this check exists to avoid "
                        "paying for on a bad image.")
    p.add_argument("--local_files_only", type=lambda s: s.lower() != "false", default=True)
    p.add_argument("--dino_model_path", default="checkpoints/local_models/grounding-dino-tiny")
    args = p.parse_args()

    image_path = Path(args.image).resolve()

    print("=" * 60)
    print(f"  Checking: {image_path}")
    print("=" * 60)

    all_findings = check_file_and_quality(image_path)
    hard_fail = any(level == "FAIL" for level, _ in all_findings)

    if not hard_fail and args.check_detections:
        all_findings += check_detections(image_path, args.local_files_only, args.dino_model_path)
        hard_fail = any(level == "FAIL" for level, _ in all_findings)
    elif not args.check_detections:
        print("\n(--check_detections not passed -- skipping the DINO-only detection check;\n"
              " this verdict only covers file/quality checks, not whether DINO finds anything.)")

    print()
    for level, msg in all_findings:
        print(f"  [{level:<4}] {msg}")

    has_warn = any(level == "WARN" for level, _ in all_findings)
    print()
    if hard_fail:
        print("  VERDICT: NOT SUITABLE -- skip this image, don't spend GPU time computing its segmentation.")
    elif has_warn:
        print("  VERDICT: USABLE, WITH CAVEATS -- see WARN lines above.")
    else:
        print("  VERDICT: SUITABLE -- looks good for the full pipeline.")
    print("=" * 60)

    raise SystemExit(1 if hard_fail else 0)


if __name__ == "__main__":
    main()
