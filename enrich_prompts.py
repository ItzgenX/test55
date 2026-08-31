"""
enrich_prompts.py
------------------
Walk a folder for every *.jsonl manifest (the {"raw_image_path"/"source",
"prompt", ...} convention this repo's precompute scripts already produce),
caption each image with a local vision-language model, and REPLACE the
"prompt" field with a real, per-image caption -- instead of the one fixed
template every image in a Cityscapes city currently shares (e.g. "a photo
of a city street in aachen", identical across ~150 images).
configs/experiment/train_seg.yaml already flags this as a known gap: "a
vague prompt lets the base model's own prior dominate the structure
signal" -- this script is that fix.

Why this doesn't fight the segmentation/depth-conditioning goal: the
caption describes scene CONTENT ("a two-lane asphalt road under overcast
sky, a silver sedan parked beside a row of brick townhouses"), not precise
spatial layout -- it gives the text encoder something real to work with,
while the structure map stays the only signal carrying WHERE things are.
The two are complementary, not competing.

MODEL: Qwen2-VL-Instruct (default 7B), not BLIP. BLIP is a small
(~470M param), non-instructable captioning model -- it produces short,
generic, often repetitive captions and gives you no control over style or
content emphasis. Qwen2-VL is a real instructable vision-language model:
you can tell it exactly what to describe (road type, weather, key
objects), and it produces meaningfully more accurate, detailed captions.
This is a one-time OFFLINE preprocessing pass over the dataset -- model
size has no bearing on training-time compute, so there's no reason to
stay small once real GPUs (e.g. a 4x95GB cluster) are available. A 7B
model in bf16 is ~15GB, comfortable on a single 95GB GPU with room to
spare; it will NOT fit on a 12GB local dev GPU -- use --model with the 2B
variant (Qwen/Qwen2-VL-2B-Instruct) for local smoke-testing only.

Reversible: every touched file gets a one-time <name>.jsonl.bak copy before
the first write (never overwritten on reruns -- your true original always
survives), and each entry keeps "prompt_original" alongside the new
"prompt", so nothing is silently lost even without the backup.

Resumable: an entry that already has "prompt_original" (already enriched by
a prior run) is skipped unless --force is passed -- safe to interrupt and
rerun, or to run again after adding new entries to a manifest.

Usage:
  python enrich_prompts.py --jsonl_dir data/seg_training_aspect
  python enrich_prompts.py --jsonl_dir data/dataset/data_full --dry_run_n 5
  python enrich_prompts.py --jsonl_dir data/seg_training_aspect --force   # re-caption already-enriched entries too

  # local smoke test on a 12GB dev GPU (2B model, tiny sample):
  python enrich_prompts.py --jsonl_dir data/dataset/data_full --model Qwen/Qwen2-VL-2B-Instruct \\
      --local_files_only false --dry_run_n 3

  # real cluster run (7B, offline once weights are cached locally):
  python enrich_prompts.py --jsonl_dir data/seg_training_aspect \\
      --model checkpoints/local_models/qwen2-vl-7b-instruct

First run per model needs internet once to download the weights (~15GB for
the 7B model); every run after that is fully offline via --local_files_only
(default True, once you've saved the model to the --model local folder).
"""

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

from src.utils import resolve_device

DEFAULT_CAPTION_MODEL = "Qwen/Qwen2-VL-7B-Instruct"

DEFAULT_INSTRUCTION = (
    "Describe this driving scene in one or two dense, specific sentences, written as an "
    "image-generation caption. Mention the road type, weather and lighting, and the key "
    "vehicles, pedestrians, and buildings actually visible. Be accurate -- do not invent "
    "objects that aren't there. Do not mention cameras, segmentation, watermarks, image "
    "quality, or that this is a photo -- describe only the scene content itself."
)


def _get_image_path(entry: dict, image_path: str) -> str:
    """Same key-fallback convention as seg_map_calculations.py's
    _get_image_path: try the configured key first, then the two other keys
    this repo's manifests use in different places, so one script works
    against data_full's own "source" jsonls AND the *_seg_map jsonls'
    "raw_image_path" jsonls without a --image_path flag most of the time."""
    for key in (image_path, "raw_image_path", "source", "target"):
        if key in entry:
            return entry[key]
    raise KeyError(f"Entry has none of raw_image_path/source/target. Keys: {list(entry.keys())}")


def build_chat_text(processor: AutoProcessor, instruction: str) -> str:
    """Qwen2-VL's chat template is IDENTICAL for every image in a batch (only
    the actual image tensor differs), so render it once and reuse the string
    -- avoids re-running apply_chat_template per image."""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": instruction},
            ],
        }
    ]
    return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def caption_batch(
    imgs: list[Image.Image],
    processor: AutoProcessor,
    model: Qwen2VLForConditionalGeneration,
    device: str,
    chat_text: str,
    max_new_tokens: int,
) -> list[str]:
    inputs = processor(
        text=[chat_text] * len(imgs),
        images=imgs,
        padding=True,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        # greedy decoding: accuracy over variety here -- the goal is a
        # faithful description of what's actually in the frame, not a
        # creative one. Sampling would trade some of that faithfulness for
        # variety we don't need (each image is already visually distinct).
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.1,
        )

    # generate() returns prompt+completion; slice off the input tokens per-sample
    trimmed = [out[len(inp):] for inp, out in zip(inputs["input_ids"], generated)]
    captions = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    return [c.strip() for c in captions]


def enrich_jsonl(
    jsonl_path: Path,
    processor: AutoProcessor,
    model: Qwen2VLForConditionalGeneration,
    device: str,
    chat_text: str,
    image_path_key: str = "raw_image_path",
    batch_size: int = 8,
    max_new_tokens: int = 120,
    force: bool = False,
    dry_run_n: int | None = None,
    root: Path | None = None,
) -> tuple[int, int, int]:
    """Caption every image referenced in jsonl_path, rewrite "prompt" in
    place. Returns (n_enriched, n_skipped_already_done, n_failed)."""
    with open(jsonl_path, "r", encoding="utf-8") as f:
        entries = [json.loads(line) for line in f if line.strip()]
    if dry_run_n:
        entries = entries[:dry_run_n]

    todo_idx, todo_paths = [], []
    n_skipped = 0
    for i, entry in enumerate(entries):
        if not force and "prompt_original" in entry:
            n_skipped += 1
            continue
        try:
            p = Path(_get_image_path(entry, image_path_key))
        except KeyError as e:
            print(f"  [WARN] entry {i}: {e}")
            continue
        if not p.is_absolute() and root is not None:
            p = root / p
        todo_idx.append(i)
        todo_paths.append(p)

    n_failed = 0
    for b in tqdm(range(0, len(todo_idx), batch_size), desc=jsonl_path.name):
        batch_idx = todo_idx[b: b + batch_size]
        batch_paths = todo_paths[b: b + batch_size]

        imgs, valid_idx = [], []
        for idx, p in zip(batch_idx, batch_paths):
            try:
                imgs.append(Image.open(p).convert("RGB"))
                valid_idx.append(idx)
            except Exception as e:
                print(f"  [WARN] entry {idx}: image load failed ({p}): {e}")
                n_failed += 1

        if not imgs:
            continue

        captions = caption_batch(imgs, processor, model, device, chat_text, max_new_tokens)

        for idx, caption in zip(valid_idx, captions):
            if not caption:
                n_failed += 1
                continue
            entries[idx]["prompt_original"] = entries[idx].get("prompt", "")
            entries[idx]["prompt"] = caption

    n_enriched = len(todo_idx) - n_failed

    # ---- backup once, never overwritten on reruns -------------------------- #
    backup_path = jsonl_path.with_suffix(jsonl_path.suffix + ".bak")
    if not backup_path.exists():
        with open(jsonl_path, "r", encoding="utf-8") as src, open(backup_path, "w", encoding="utf-8") as dst:
            dst.write(src.read())

    with open(jsonl_path, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    return n_enriched, n_skipped, n_failed


def main():
    parser = argparse.ArgumentParser(
        description="Caption every image in every *.jsonl under a folder, enriching the 'prompt' field.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--jsonl_dir", required=True, help="Folder to search recursively for *.jsonl manifests.")
    parser.add_argument("--image_path", default="raw_image_path", help="Preferred key holding each entry's image path.")
    parser.add_argument("--model", default="checkpoints/local_models/qwen2-vl-7b-instruct",
                         help=f"Local folder (default) or a hub id like {DEFAULT_CAPTION_MODEL!r} with "
                              f"--local_files_only false for first download. Use Qwen/Qwen2-VL-2B-Instruct "
                              f"for local smoke-testing on a 12GB GPU -- the 7B default needs ~15GB.")
    parser.add_argument("--instruction", default=DEFAULT_INSTRUCTION, help="Instruction given to the model per image.")
    parser.add_argument("--local_files_only", type=lambda s: s.lower() != "false", default=True)
    parser.add_argument("--device", default=None, help="e.g. 'cpu' to force CPU (auto-detects GPU if omitted).")
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=120)
    parser.add_argument("--force", action="store_true", help="Re-caption entries that already have prompt_original.")
    parser.add_argument("--dry_run_n", type=int, default=None, help="Only the first N entries per file (verify before committing to the full folder).")
    args = parser.parse_args()

    jsonl_dir = Path(args.jsonl_dir).resolve()
    if not jsonl_dir.exists():
        raise FileNotFoundError(f"--jsonl_dir not found: {jsonl_dir}")

    jsonl_files = sorted(jsonl_dir.rglob("*.jsonl"))
    jsonl_files = [p for p in jsonl_files if p.suffix == ".jsonl"]
    if not jsonl_files:
        print(f"[ERROR] No .jsonl files found under {jsonl_dir}")
        return

    print(f"Found {len(jsonl_files)} jsonl file(s) under {jsonl_dir}:")
    for p in jsonl_files:
        print(f"  {p}")

    device = resolve_device(args.device)
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]

    print(f"\nLoading caption model: {args.model}  (local_files_only={args.local_files_only}, dtype={args.dtype})")
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=args.local_files_only)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model, local_files_only=args.local_files_only, torch_dtype=dtype
    ).to(device).eval()
    print("Caption model ready.\n")
    print(f"Instruction: {args.instruction!r}\n")

    chat_text = build_chat_text(processor, args.instruction)

    total_enriched = total_skipped = total_failed = 0
    for jsonl_path in jsonl_files:
        n_enriched, n_skipped, n_failed = enrich_jsonl(
            jsonl_path, processor, model, device, chat_text,
            image_path_key=args.image_path,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            force=args.force,
            dry_run_n=args.dry_run_n,
            root=jsonl_dir,
        )
        total_enriched += n_enriched
        total_skipped += n_skipped
        total_failed += n_failed
        print(f"  [{jsonl_path.name}] enriched={n_enriched}  already-done-skipped={n_skipped}  failed={n_failed}")

    print(f"\n{'='*52}")
    print(f"  Total enriched : {total_enriched}")
    print(f"  Already done   : {total_skipped}")
    print(f"  Failed         : {total_failed}")
    print(f"{'='*52}")
    if total_enriched:
        print("\nSample enriched entries:")
        with open(jsonl_files[0], "r", encoding="utf-8") as f:
            for line in list(f)[:3]:
                e = json.loads(line)
                if "prompt_original" in e:
                    print(f"  before: {e['prompt_original']!r}")
                    print(f"  after : {e['prompt']!r}\n")


if __name__ == "__main__":
    main()
