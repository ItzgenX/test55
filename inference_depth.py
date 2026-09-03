"""
inference_depth.py
-------------------
Run inference with a trained depth-conditioned LoRAdapter (DEPTH-BRANCH-
architecture.md RULE 0: derives from the original repo + paper only).

Accepts EITHER:
  - a raw source image ("images" / "image_dir" in config) -- runs the exact
    same DepthPrecomputer used by compute_depth.py (imported from it, so
    there is only one implementation of the resize policy in this repo), or
  - a precomputed depth map ("depth_maps") -- loaded via compute_depth.py's
    own load_depth(), and checked against spec.json in the same cache
    directory so a checkpoint trained against one resize policy can't
    silently be fed maps computed under a different one.

Domain LoRA (§7): if domain_lora_path is set, it is loaded and FUSED into
the base UNet BEFORE add_lora_to_unet wires the depth adapter in --
`add_lora_to_unet` copies each weight out of unet.state_dict() into the
frozen `lora.W`, so fusing after would mean the module surgery bypasses the
domain adaptation entirely.

Saves a 3-panel grid (ORIGINAL | DEPTH | PREDICTED) per image, and lora_scale
is a CLI-overridable float (`inference.lora_scale=0.7`), for picking the
best checkpoint / conditioning strength.

Usage:
  python inference_depth.py ckpt_path=outputs/train/depth/runs/.../checkpoint-2000 \\
      inference.image_dir=some/folder inference.lora_scale=1.0

  python inference_depth.py ckpt_path=... \\
      inference.json_file=data/dataset/depth_cache/val.jsonl
"""

import json
import os
from pathlib import Path

import hydra
import numpy as np
import torch
from PIL import Image

from compute_depth import DepthPrecomputer, WORKING_SIZE, load_depth
from src.mapper_network import DepthStructureMapperXL
from src.model import SDXL
from src.utils import DataProvider

PROJECT_ROOT = Path(os.path.abspath(__file__)).parent


def resolve_device(cfg_device):
    if cfg_device is not None:
        return cfg_device
    return "cuda" if torch.cuda.is_available() else "cpu"


def build_model(cfg, device):
    model_name = cfg.base_model_path if cfg.local_files_only else cfg.base_model_name
    model = SDXL(
        pipeline_type="diffusers.StableDiffusionXLPipeline",
        model_name=model_name,
        local_files_only=cfg.local_files_only,
        guidance_scale=cfg.inference.guidance_scale,
    )
    model = model.to(device)
    model.pipe.to(device)
    return model


def fuse_domain_lora(model: SDXL, domain_lora_path: str | None):
    if not domain_lora_path:
        return
    print(f"fusing domain LoRA from {domain_lora_path} (must happen before add_lora_to_unet)")
    model.pipe.load_lora_weights(domain_lora_path)
    model.pipe.fuse_lora()


def wire_depth_lora(model: SDXL, cfg, ckpt_path: Path, device):
    mapper = DepthStructureMapperXL(c_dim=cfg.lora.struct.config.c_dim).to(device)
    dp = DataProvider()

    class_config = dict(cfg.lora.struct.config)
    lora_cls = class_config.pop("lora_cls")
    adaption_mode = class_config.pop("adaption_mode")

    class Cfg:
        pass

    lora_cfg = Cfg()
    lora_cfg.lora_cls = lora_cls
    lora_cfg.adaption_mode = adaption_mode
    for k, v in class_config.items():
        setattr(lora_cfg, k, v)

    model.add_lora_to_unet(
        lora_cfg,
        name="struct",
        data_provider=dp,
        encoder=torch.nn.Identity(),
        mapper=mapper,
        optimize=False,
        transforms=[],
    )

    lora_sd = torch.load(ckpt_path / "struct" / "lora-checkpoint.pt", map_location=device)
    mapper_sd = torch.load(ckpt_path / "struct" / "mapper-checkpoint.pt", map_location=device)
    model.unet.load_state_dict(lora_sd, strict=False)
    mapper.load_state_dict(mapper_sd)
    mapper.eval()
    return mapper, dp


def set_lora_scale(model: SDXL, scale: float):
    for layer in model.lora_layers["struct"]:
        layer.lora_scale = scale


def check_spec(depth_dir: Path, precomputer_cls=DepthPrecomputer):
    spec_path = depth_dir / "spec.json"
    if not spec_path.exists():
        print(f"WARNING: no spec.json found next to {depth_dir} -- cannot verify resize policy, proceeding anyway.")
        return
    with open(spec_path) as f:
        spec = json.load(f)
    if tuple(spec["working_size_wh"]) != WORKING_SIZE:
        raise ValueError(
            f"{spec_path} was computed at working size {spec['working_size_wh']}, "
            f"but this script's policy is {WORKING_SIZE}. Refusing to feed a checkpoint "
            f"maps it wasn't trained to expect."
        )


def make_panel(original: Image.Image | None, depth_01: np.ndarray, predicted: Image.Image) -> Image.Image:
    w, h = predicted.size
    depth_img = Image.fromarray((np.clip(depth_01, 0, 1) * 255).astype(np.uint8)).convert("RGB").resize((w, h))
    panels = []
    if original is not None:
        panels.append(original.resize((w, h)))
    panels.append(depth_img)
    panels.append(predicted)
    grid = Image.new("RGB", (w * len(panels), h))
    for i, p in enumerate(panels):
        grid.paste(p, (i * w, 0))
    return grid


def collect_items(cfg):
    """Returns list of dicts: {"prompt", "source"(optional), "depth_path"(optional)}."""
    items = []
    if cfg.inference.json_file:
        with open(PROJECT_ROOT / cfg.inference.json_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                items.append(json.loads(line))
    if cfg.inference.get("image_dir", None):
        d = Path(cfg.inference.image_dir)
        for p in sorted(d.glob("*")):
            if p.suffix.lower() in (".jpg", ".jpeg", ".png"):
                items.append({"source": str(p), "prompt": cfg.inference.get("default_prompt", "")})
    for i, p in enumerate(cfg.inference.get("depth_maps", [])):
        items.append({"depth_path": p, "prompt": (cfg.inference.prompts[i] if i < len(cfg.inference.prompts) else "")})
    for i, p in enumerate(cfg.inference.get("images", [])):
        items.append({"source": p, "prompt": (cfg.inference.prompts[i] if i < len(cfg.inference.prompts) else "")})
    return items


@hydra.main(config_path="configs", config_name="inference_depth")
def main(cfg):
    device = resolve_device(cfg.device)
    torch.manual_seed(cfg.seed)

    ckpt_path = Path(cfg.ckpt_path)
    output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir) / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    model = build_model(cfg, device)
    fuse_domain_lora(model, cfg.get("domain_lora_path", None))
    mapper, dp = wire_depth_lora(model, cfg, ckpt_path, device)
    set_lora_scale(model, cfg.inference.lora_scale)

    precomputer = None  # lazily built only if a raw source image is actually provided

    items = collect_items(cfg)
    if len(items) == 0:
        raise ValueError("no inputs: set inference.json_file, inference.image_dir, inference.images, or inference.depth_maps")

    print(f"running inference on {len(items)} item(s), lora_scale={cfg.inference.lora_scale}")

    for i, item in enumerate(items):
        prompt = item.get("prompt", "")
        original_img = None

        if "depth_path" in item:
            depth_dir = (PROJECT_ROOT / item["depth_path"]).parent.parent  # <cache_dir>/<split>/<stem> -> <cache_dir>
            check_spec(depth_dir)
            depth_arr = load_depth(PROJECT_ROOT / item["depth_path"])  # (H,W) in [0,1]
            if "source" in item:
                original_img = Image.open(PROJECT_ROOT / item["source"]).convert("RGB")
        elif "source" in item:
            if precomputer is None:
                precomputer = DepthPrecomputer(cfg.midas_model_name, cfg.midas_model_path, cfg.local_files_only, device)
            original_img = Image.open(PROJECT_ROOT / item["source"]).convert("RGB")
            depth_t = precomputer.depth_from_pil(original_img)  # [1,768,1280] in [0,1]
            depth_arr = depth_t.squeeze(0).cpu().numpy()
            original_img = precomputer.crop_source(original_img)
        else:
            raise ValueError(f"item {i} has neither 'source' nor 'depth_path': {item}")

        depth_3ch = torch.from_numpy(depth_arr).float().unsqueeze(0).repeat(3, 1, 1).unsqueeze(0).to(device)

        generator = torch.Generator(device=device).manual_seed(cfg.seed)
        preds = model.sample(
            prompt=[prompt],
            num_images_per_prompt=cfg.inference.n_samples,
            cs=[depth_3ch],
            generator=generator,
            cfg_mask=[True],
            skip_encode=True,
            num_inference_steps=cfg.inference.num_inference_steps,
        )

        for j, pred in enumerate(preds):
            panel = make_panel(original_img, depth_arr, pred)
            out_name = f"{i:04d}_{j}.jpg"
            panel.save(output_dir / out_name, quality=92)
            if cfg.inference.save_generated_only:
                pred.save(output_dir / f"{i:04d}_{j}_generated.jpg", quality=92)

        print(f"[{i+1}/{len(items)}] '{prompt[:60]}' -> {output_dir}/{i:04d}_*.jpg")

    print(f"done. results in {output_dir}")


if __name__ == "__main__":
    main()
