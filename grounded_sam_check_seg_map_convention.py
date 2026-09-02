"""
grounded_sam_check_seg_map_convention.py
-----------------------------------------
Check which class-id convention your REAL precomputed grounded_sam seg-map
PNGs actually use on disk, before trusting them for training.

QUICK COMMANDS (run from repo root with conda loradapter env active):
  # --- Plain directory of seg-map PNGs ---
  python grounded_sam_check_seg_map_convention.py --maps_dir /path/to/seg_maps

  # --- Directory of your manifest files (.jsonl -- this repo's own
  #     train.jsonl/val.jsonl format -- or .yaml/.yml), each row/file
  #     holding a seg_map/seg_path key ---
  python grounded_sam_check_seg_map_convention.py --manifest_dir /path/to/your/manifests

  # --- Same, but seg_map paths inside the manifest are relative to a
  #     different root than the manifest file's own directory ---
  python grounded_sam_check_seg_map_convention.py --manifest_dir /path/to/your/manifests --base_dir /path/to/maps_root

WHAT THIS ANSWERS: does src/data/local_seg.py's "+1" shift at load time
(0-indexed disk value -> 1-indexed training id, see
SegJsonDataset._load_seg_ids) match what's really in your files, or does
it silently double-shift them? One real bug class, not hypothetical: if a
different tool wrote your PNGs already 1-indexed, every class gets
relabeled one position over (road pixels read as sidewalk, etc.) with no
crash anywhere to catch it -- this script settles the question from actual
pixel values instead of guessing.

THIS REPO'S OWN CONVENTION (what the checker treats as "correct", CASE A):
  src/encoders/grounded_sam_encoder.py writes CARLA_CLASSES.index(cls)
  directly, grounded_sam_map_calculations.py saves that unchanged -- so a
  map built by this repo's own pipeline is 0-indexed on disk: road=0,
  sidewalk=1, ..., guard_rail=27, 255=IGNORE (unmatched pixels).
  local_seg.py shifts +1 at load time (0->NULL is never written by the
  encoder, so no collision) to land on the training-time scheme:
  0=NULL/padding, 1..28=real classes, 29=IGNORE.

IF YOUR MAPS CAME FROM SOMEWHERE ELSE (CASE B): a different precompute
tool may already write 1-indexed values (road=1, sidewalk=2, ...). Applying
local_seg.py's default "+1" shift on top of that double-shifts every real
class. Fix without regenerating anything: set
data.seg_map_already_shifted=true on your training launch command (wired
through configs/data/grounded_sam_jsonl.yaml -> SegJsonDataModule ->
SegJsonDataset -- see that flag's docstring in local_seg.py).

MANIFEST SCANNING DETAIL: --manifest_dir recurses into every *.jsonl,
*.yaml, and *.yml under the given directory. .jsonl files are read as this
repo's own manifest convention (see src/data/local_seg.py's SegJsonDataset
docstring) -- one JSON object per line, each row's own seg_map/seg_path
key collected directly, not nested further (a manifest row is flat).
.yaml/.yml files are parsed whole and searched at any nesting depth, since
a config file's structure isn't fixed the way a manifest row's is. Either
way, the key must be literally named seg_map/seg_path/seg_map_path.
Relative paths found inside a file resolve against that file's own
directory by default (the common "path next to the file that names it"
convention); pass --base_dir to resolve against a different root instead.
Any referenced path that doesn't actually exist on disk is reported, not
silently dropped -- see it as a warning before the check runs on whatever
files DID resolve.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import yaml
from PIL import Image

MAP_KEYS = {"seg_map", "seg_path", "seg_map_path"}


def find_map_paths_in_yaml(obj, base_dir: str, found: list) -> None:
    """Recurse into a parsed yaml structure (dict/list/scalar), collecting
    every value under a key in MAP_KEYS. Relative paths are resolved
    against base_dir (the yaml file's own directory, unless overridden)."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in MAP_KEYS and isinstance(v, str):
                p = v if os.path.isabs(v) else os.path.join(base_dir, v)
                found.append(p)
            else:
                find_map_paths_in_yaml(v, base_dir, found)
    elif isinstance(obj, list):
        for item in obj:
            find_map_paths_in_yaml(item, base_dir, found)


def collect_from_jsonl(jsonl_path: str, base_dir: str, found: list) -> None:
    """Each line is one flat JSON object (a manifest row) -- collect any
    MAP_KEYS value directly from it, same relative-path resolution as yaml."""
    with open(jsonl_path, "r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception as e:
                print(f"  [skip] {jsonl_path}:{lineno}: failed to parse ({e})")
                continue
            if not isinstance(row, dict):
                continue
            for k, v in row.items():
                if k in MAP_KEYS and isinstance(v, str):
                    p = v if os.path.isabs(v) else os.path.join(base_dir, v)
                    found.append(p)


def collect_from_manifest_dir(manifest_dir: str, base_dir_override: str | None) -> list:
    jsonl_files = glob.glob(os.path.join(manifest_dir, "**", "*.jsonl"), recursive=True)
    yaml_files = glob.glob(os.path.join(manifest_dir, "**", "*.yaml"), recursive=True)
    yaml_files += glob.glob(os.path.join(manifest_dir, "**", "*.yml"), recursive=True)
    if not jsonl_files and not yaml_files:
        print(f"No .jsonl/.yaml/.yml files found under {manifest_dir}")
        sys.exit(1)

    found: list = []
    for jf in jsonl_files:
        base_dir = base_dir_override or os.path.dirname(jf)
        collect_from_jsonl(jf, base_dir, found)
    for yf in yaml_files:
        try:
            with open(yf, "r", encoding="utf-8") as fh:
                data = yaml.safe_load(fh)
        except Exception as e:
            print(f"  [skip] {yf}: failed to parse ({e})")
            continue
        base_dir = base_dir_override or os.path.dirname(yf)
        find_map_paths_in_yaml(data, base_dir, found)

    print(f"Scanned {len(jsonl_files)} jsonl + {len(yaml_files)} yaml file(s) "
          f"under {manifest_dir}, found {len(found)} seg_map reference(s).")
    if not found:
        print(f"No key in {sorted(MAP_KEYS)} was found under {manifest_dir} -- "
              f"if your key is named something else, pass --maps_dir directly instead.")
        sys.exit(1)
    return found


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--maps_dir", help="Plain directory of seg-map PNGs (recursed).")
    src.add_argument("--manifest_dir", help="Directory of .jsonl/.yaml/.yml files, each with a seg_map key.")
    ap.add_argument("--base_dir", default=None,
                     help="Resolve relative seg_map paths against this dir instead of "
                          "each manifest file's own directory.")
    ap.add_argument("--limit", type=int, default=30, help="Max number of PNGs to sample.")
    args = ap.parse_args()

    if args.maps_dir:
        files = glob.glob(os.path.join(args.maps_dir, "**", "*.png"), recursive=True)
        if not files:
            print(f"No PNGs found under {args.maps_dir}")
            sys.exit(1)
    else:
        files = collect_from_manifest_dir(args.manifest_dir, args.base_dir)
        missing = [f for f in files if not os.path.isfile(f)]
        files = [f for f in files if os.path.isfile(f)]
        if missing:
            print(f"WARNING: {len(missing)} referenced seg_map path(s) do not exist on disk, e.g.:")
            for m in missing[:5]:
                print(f"    {m}")
            if args.base_dir is None:
                print("  If these are relative paths meant to resolve against a different root, "
                      "pass --base_dir explicitly.")
        if not files:
            print("None of the referenced seg_map paths exist on disk -- nothing to check.")
            sys.exit(1)

    files = files[: args.limit]
    all_vals = set()
    for f in files:
        arr = np.asarray(Image.open(f))
        all_vals.update(np.unique(arr).tolist())

    real_vals = sorted(v for v in all_vals if v != 255)
    print(f"Sampled {len(files)} file(s).")
    print(f"All unique pixel values seen: {sorted(all_vals)}")
    print(f"Real-class values (excluding 255): min={min(real_vals)}, max={max(real_vals)}")
    print()

    if min(real_vals) == 0 and max(real_vals) <= 27:
        print("=> CASE A: 0-indexed on disk (road=0 .. guard_rail=27), 255=ignore.")
        print("   Matches src/encoders/grounded_sam_encoder.py's own output exactly.")
        print("   local_seg.py's existing '+1' shift is CORRECT. No fix needed.")
        print("   Keep data.seg_map_already_shifted=false (the default).")
    elif min(real_vals) == 1 and max(real_vals) <= 28:
        print("=> CASE B: already 1-indexed on disk (road=1 .. guard_rail=28).")
        print("   local_seg.py's '+1' shift would DOUBLE-SHIFT this -- every real")
        print("   class gets silently relabeled one position over (road pixels")
        print("   read as sidewalk, etc.).")
        print("   Set data.seg_map_already_shifted=true to fix this without")
        print("   regenerating any of your existing maps.")
    else:
        print("=> UNEXPECTED range -- neither a clean 0-27 nor a clean 1-28 spread.")
        print("   Don't apply either fix blindly -- something else is going on;")
        print("   inspect a few individual files by hand before proceeding.")


if __name__ == "__main__":
    main()
