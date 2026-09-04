"""
filter_jsonl_by_existing_images.py
------------------------------------
Remove jsonl entries that are no longer TRAINING-READY -- keeps a row only
if its raw image, its segmentation map, AND its prompt all check out
together. A row surviving because its raw image was found, while its seg
map was silently missing (never checked in the same pass), would poison
training with a mismatched pair -- this script checks everything a row
needs in ONE pass, so nothing incomplete slips through.

QUICK COMMANDS (run from repo root with conda loradapter env active):
  # --- Check both raw image AND seg map together (the normal case) ---
  python filter_jsonl_by_existing_images.py --jsonl train.jsonl \\
      --raw_images_dir /path/to/extracted/raw_images \\
      --seg_images_dir /path/to/extracted/seg_maps

  # --- Only checking one side (e.g. you haven't extracted seg maps yet) ---
  python filter_jsonl_by_existing_images.py --jsonl train.jsonl \\
      --raw_images_dir /path/to/extracted/raw_images

  # --- Against your ORIGINAL NESTED dataset instead of an extracted folder ---
  python filter_jsonl_by_existing_images.py --jsonl train.jsonl \\
      --raw_images_dir /path/to/nested/dataset/root --raw_mode nested \\
      --seg_images_dir /path/to/nested/dataset/root --seg_mode nested

Writes <jsonl_stem>_filtered.jsonl next to the input -- never overwrites
your original jsonl.

A row is KEPT only if EVERY check you've enabled passes:
  - raw image check (if --raw_images_dir given)
  - seg map check (if --seg_images_dir given)
  - prompt is a real, non-empty string (unless --no_require_prompt)
Any single failure drops the row, even if its other fields are fine --
that's the whole point: a training row is only as good as its weakest field.

TWO MATCHING STRATEGIES per side (--raw_mode/--seg_mode, each independently
"nested" or "flat", default "flat"; --raw_match_by/--seg_match_by, each
"hash" or "path", default "hash"):

  mode nested: your ORIGINAL nested dataset (folders inside folders).
    Checks the path (used as-is if absolute, or resolved against the
    images_dir if relative) exists as a real file. Exact, no fuzzy
    matching -- nothing about a nested path should have changed.

  mode flat, match_by hash (the default -- use this if every sample has
    the SAME filename, e.g. "raw_image.jpg"/"class_map.png" everywhere
    and only the folder path made each one unique; filenames then carry
    zero identifying information, so this is the only reliable option).
    Hashes every file in the images_dir once, then for each row hashes
    the ORIGINAL file (which must still exist -- this only ever prunes
    the jsonl, never your untouched nested dataset) and checks whether a
    byte-identical copy still exists anywhere in the extracted folder,
    regardless of what it got renamed to.

  mode flat, match_by path (only if your extracted filenames really are
    unique, e.g. "A__city1__scene2__image003.png"-style __-joined names).
    Reconstructs the expected flat name first; falls back to matching by
    the original filename alone (whatever came after the last "__") if
    that exact reconstruction misses. If more than one file shares that
    filename, there is no safe way to guess which is right, so the row
    is dropped and reported under AMBIGUOUS instead of a silent guess.

--raw_key/--seg_key default to "raw_image_path"/"seg_path" (this repo's
own manifest convention) -- override if your jsonl uses different keys.
"""
import argparse
import hashlib
import json
from pathlib import Path


# ── match_by hash ───────────────────────────────────────────────────────── #

def file_hash(path: Path, chunk_size: int = 1 << 20) -> str:
    """SHA-256 of the file's actual bytes, read in chunks so this doesn't
    load a large image fully into memory at once."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def build_hash_index(images_dir: Path) -> dict[str, list[Path]]:
    """content-hash -> every file under images_dir with that exact content."""
    index: dict[str, list[Path]] = {}
    for f in images_dir.rglob("*"):
        if f.is_file():
            index.setdefault(file_hash(f), []).append(f)
    return index


# ── match_by path ───────────────────────────────────────────────────────── #

def flat_name_for(raw_path: str) -> str:
    """Reconstruct the expected __-joined flat filename from a nested path.
    'D:/data/A/city1/scene2/image003.png' -> 'A__city1__scene2__image003.png'
    (drive letters and bare path separators are dropped, not joined in)."""
    parts = [p for p in Path(raw_path).parts if p not in ("/", "\\") and not p.endswith(":")]
    return "__".join(parts)


def build_basename_index(images_dir: Path) -> dict[str, list[Path]]:
    """original-filename (e.g. 'image003.png') -> every file under images_dir
    that could be it. "__" isn't a real path separator in a flat folder --
    a file there is genuinely named "A__city1__scene2__image003.png" as one
    whole filename, so matching on f.name directly would never find
    anything. Split each flat filename on "__" and take the last segment
    as its reconstructed original basename instead."""
    index: dict[str, list[Path]] = {}
    for f in images_dir.rglob("*"):
        if f.is_file():
            original_basename = f.name.split("__")[-1]
            index.setdefault(original_basename, []).append(f)
    return index


class Check:
    """One field's verification setup: which jsonl key it reads, which
    folder/strategy to verify it against, and the (lazily built, built-once)
    lookup index that strategy needs."""

    def __init__(self, label: str, key: str, images_dir: Path, mode: str, match_by: str):
        self.label = label      # "raw image" / "seg map" -- for readable output only
        self.key = key
        self.images_dir = images_dir
        self.mode = mode
        self.match_by = match_by
        self.index: dict[str, list[Path]] | None = None

        if mode == "flat":
            if match_by == "hash":
                print(f"Hashing every file in {images_dir} for {label} content matching...")
                self.index = build_hash_index(images_dir)
            else:
                print(f"Indexing {images_dir} by filename for {label} basename fallback matching...")
                self.index = build_basename_index(images_dir)
            total_files = sum(len(v) for v in self.index.values())
            print(f"  indexed {total_files} file(s), {len(self.index)} unique key(s)")

    def verify(self, row: dict) -> tuple[bool, str]:
        """Returns (passed, reason). reason is always filled in, even on
        success, so a verbose caller could log it -- kept short here."""
        raw_path = row.get(self.key)
        if not raw_path:
            return False, f"{self.label}: row has no {self.key!r} key"

        if self.mode == "nested":
            candidate = Path(raw_path)
            if not candidate.is_absolute():
                candidate = self.images_dir / candidate
            if candidate.is_file():
                return True, f"{self.label}: found at {candidate}"
            return False, f"{self.label}: not found: {candidate}"

        # mode == "flat"
        if self.match_by == "hash":
            original = Path(raw_path)
            if not original.is_file():
                return False, f"{self.label}: UNVERIFIABLE, original source missing: {original}"
            h = file_hash(original)
            matches = (self.index or {}).get(h, [])
            if matches:
                return True, f"{self.label}: content match found ({len(matches)} identical copy/copies)"
            return False, f"{self.label}: no file with matching content found in {self.images_dir}"

        # mode == "flat", match_by == "path"
        expected_flat = flat_name_for(raw_path)
        exact_path = self.images_dir / expected_flat
        if exact_path.is_file():
            return True, f"{self.label}: exact match {expected_flat}"
        basename = Path(raw_path).name
        matches = (self.index or {}).get(basename, [])
        if len(matches) == 1:
            return True, f"{self.label}: basename match {matches[0]}"
        if len(matches) == 0:
            return False, f"{self.label}: not found (tried exact {expected_flat!r} and basename {basename!r})"
        return False, f"{self.label}: AMBIGUOUS, {len(matches)} files named {basename!r} exist: {matches}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jsonl", action="append", required=True, help="jsonl file(s) to filter; repeatable")

    ap.add_argument("--raw_images_dir", default=None, help="folder to verify raw images against (omit to skip this check)")
    ap.add_argument("--raw_key", default="raw_image_path")
    ap.add_argument("--raw_mode", choices=["nested", "flat"], default="flat")
    ap.add_argument("--raw_match_by", choices=["hash", "path"], default="hash")

    ap.add_argument("--seg_images_dir", default=None, help="folder to verify seg maps against (omit to skip this check)")
    ap.add_argument("--seg_key", default="seg_path")
    ap.add_argument("--seg_mode", choices=["nested", "flat"], default="flat")
    ap.add_argument("--seg_match_by", choices=["hash", "path"], default="hash")

    ap.add_argument("--no_require_prompt", action="store_true",
                     help="by default a row with an empty/missing prompt is dropped too -- pass this to allow it through")
    ap.add_argument("--prompt_key", default="prompt")

    ap.add_argument("--output_suffix", default="_filtered",
                     help="output written as <jsonl_stem><suffix>.jsonl next to the input -- never overwrites the original")
    args = ap.parse_args()

    if not args.raw_images_dir and not args.seg_images_dir:
        raise SystemExit("Give at least one of --raw_images_dir / --seg_images_dir -- nothing to check otherwise.")

    checks: list[Check] = []
    if args.raw_images_dir:
        d = Path(args.raw_images_dir)
        if not d.is_dir():
            raise SystemExit(f"--raw_images_dir not found or not a directory: {d}")
        checks.append(Check("raw image", args.raw_key, d, args.raw_mode, args.raw_match_by))
    if args.seg_images_dir:
        d = Path(args.seg_images_dir)
        if not d.is_dir():
            raise SystemExit(f"--seg_images_dir not found or not a directory: {d}")
        checks.append(Check("seg map", args.seg_key, d, args.seg_mode, args.seg_match_by))

    for jsonl_path_str in args.jsonl:
        jsonl_path = Path(jsonl_path_str)
        with open(jsonl_path, "r", encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]

        kept: list[dict] = []
        drop_reasons: list[str] = []

        for row in rows:
            failures: list[str] = []

            if not args.no_require_prompt:
                prompt = row.get(args.prompt_key)
                if not prompt or not str(prompt).strip():
                    failures.append(f"prompt: missing/empty ({args.prompt_key!r})")

            for check in checks:
                passed, reason = check.verify(row)
                if not passed:
                    failures.append(reason)

            if not failures:
                kept.append(row)
            else:
                identifying = row.get(args.raw_key) or row.get(args.seg_key) or "<row with no identifying path>"
                drop_reasons.append(f"{identifying} -- " + " | ".join(failures))

        out_path = jsonl_path.with_name(f"{jsonl_path.stem}{args.output_suffix}{jsonl_path.suffix}")
        with open(out_path, "w", encoding="utf-8") as f:
            for row in kept:
                f.write(json.dumps(row) + "\n")

        checks_desc = " + ".join(c.label for c in checks) + (" + prompt" if not args.no_require_prompt else "")
        print(f"\n{jsonl_path} (checking: {checks_desc}):")
        print(f"  original entries : {len(rows)}")
        print(f"  kept (complete)  : {len(kept)}")
        print(f"  dropped          : {len(drop_reasons)}")
        if drop_reasons:
            print(f"  dropped rows (first 20):")
            for r in drop_reasons[:20]:
                print(f"    {r}")
            if len(drop_reasons) > 20:
                print(f"    ... and {len(drop_reasons) - 20} more")
        print(f"  wrote -> {out_path}")


if __name__ == "__main__":
    main()
