from typing import Any, Literal
import torch
from accelerate import Accelerator
from datetime import datetime
from pathlib import Path
from torch.nn.utils import clip_grad_norm_
from functools import reduce
import os

# from src.model import ModelBase

MODE = Literal[
    "train",
    "val",
    "always",
]


class DataProvider:
    def __init__(self):
        self.batch = None

    def set_batch(self, batch):
        if self.batch is not None:
            if isinstance(self.batch, torch.Tensor):
                assert self.batch.shape[1:] == batch.shape[1:], "Check: shapes probably should not change during training"

        self.batch = batch

    def get_batch(self):
        assert self.batch is not None, "Error: need to set a batch first"

        return self.batch

    def reset(self):
        self.batch = None


def getattr_recursive(obj: Any, path: str) -> Any:
    parts = path.split(".")
    for part in parts:
        if part.isnumeric():
            obj = obj[int(part)]
        else:
            obj = getattr(obj, part)
    return obj


def add_lora_from_config(model, cfg: Any, device: torch.device, dtype: torch.dtype = torch.float32) -> list[bool]:
    total_dict_keys: list[str] = []
    cfg_mask: list[bool] = []

    global_ckpt_path = cfg.get("ckpt_path", None)
    project_root = Path(os.path.abspath(__file__)).parent.parent

    for name, l in cfg.lora.items():
        if l.get("enable", "always") == "never":
            continue

        optimize = l.get("optimize", False)
        lora_cfg = l.config
        print(f"Adding {name} lora! Optimize: {optimize}")

        dp = DataProvider()
        mapper_network = l.mapper_network.to(device, dtype)
        encoder = l.encoder.to(device, dtype)
        local_ckpt_path = l.get("ckpt_path", None)

        model.add_lora_to_unet(
            lora_cfg,
            name=name,
            data_provider=dp,
            mapper=mapper_network,
            encoder=encoder,
            optimize=optimize,
            transforms=l.get("transforms", []),
        )

        cfg_mask.append(l.get("cfg", True))

        p = None
        if global_ckpt_path is not None:
            p = Path(project_root, global_ckpt_path) / name

        # local checkpoints path always override global ones
        if local_ckpt_path is not None:
            p = Path(project_root, local_ckpt_path) / name

        if p is not None:
            print("loaded checkpoint for lora", name)
            mapper_sd = torch.load(p / "mapper-checkpoint.pt", map_location=device)
            lora_sd = torch.load(p / "lora-checkpoint.pt", map_location=device)

            if os.path.isfile(p / "encoder-checkpoint.pt"):
                encoder_sd = torch.load(p / "encoder-checkpoint.pt", map_location=device)
                encoder.load_state_dict(encoder_sd)

            mapper_network.load_state_dict(mapper_sd)

            if not optimize:
                mapper_network.requires_grad_(False)
                mapper_network.eval()

            model.unet.load_state_dict(lora_sd, strict=False)
            model.unet.to(device, dtype)
            total_dict_keys += list(lora_sd.keys())

    if len(total_dict_keys) > 0 and not cfg.get("ignore_check", False):
        assert set([v for vs in model.lora_state_dict_keys.values() for v in vs]) == set(
            total_dict_keys
        ), "Probably missing or incorrect checkpoint file path. Otherwise set ignore_check=true in config."

    return cfg_mask


def toggle_loras(model, cfg: Any, mode: MODE):
    for name, l in cfg.lora.items():
        if l.get("enable", "always") in [mode, "always"]:
            for layer in model.lora_layers[name]:
                layer.lora_scale = l.config.get("lora_scale", 1.0)
        else:
            try:
                for layer in model.lora_layers[name]:
                    layer.lora_scale = 0.0
            except:
                print(f"LoRA {name} is disabled. Ignoring...")


def global_gradient_norm(model):
    mappers_params = list(filter(lambda p: p.requires_grad, reduce(lambda x, y: x + list(y.parameters()), model.mappers, [])))
    encoder_params = list(filter(lambda p: p.requires_grad, reduce(lambda x, y: x + list(y.parameters()), model.encoders, [])))

    total_norm = clip_grad_norm_(model.params_to_optimize + mappers_params + encoder_params, 1e9)
    return total_norm.item()


def save_checkpoint(unet_sds: dict[str, dict[str, torch.Tensor]], mapper_network_sd: list[dict[str, torch.Tensor]], encoder_sd: list[dict[str, torch.Tensor]] | None, path: Path):
    for i, (name, sd) in enumerate(unet_sds.items()):
        p = path / name
        p.mkdir(parents=True, exist_ok=True)

        torch.save(sd, p / "lora-checkpoint.pt")
        torch.save(mapper_network_sd[i], p / f"mapper-checkpoint.pt")
        if encoder_sd is not None and len(encoder_sd[i]) > 0:
            torch.save(encoder_sd[i], p / f"encoder-checkpoint.pt")


def roll_list(l, n):
    # consistent with torch.roll
    return l[-n:] + l[:-n]


def resolve_device(device: str | None = None) -> str:
    """Resolve a device string, refusing to silently fall back to CPU.

    device=None (or "auto"): auto-detect the first visible CUDA GPU; raises
    with a fix checklist if none is visible. Pass device="cpu" to explicitly
    opt into CPU, or "cuda:N" to pin a specific GPU on a multi-GPU box.
    """
    fix_checklist = (
        "Fix checklist:\n"
        "  - Is a GPU actually attached to this machine/container?\n"
        "  - Does `nvidia-smi` show it?\n"
        "  - Was torch installed with CUDA support (check torch.version.cuda)?\n"
        "Pass device='cpu' to explicitly opt into CPU if that's really what you want."
    )

    if device is not None and device != "auto":
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(f"Requested device={device!r} but torch.cuda.is_available() is False.\n{fix_checklist}")
        return device

    if torch.cuda.is_available():
        return "cuda:0"

    raise RuntimeError(f"No CUDA GPU visible and no device was explicitly requested.\n{fix_checklist}")


def auto_batch_size(default: int = 4, device: str = "cuda:0", baseline_gb: float = 12.0) -> int:
    """Scale a baseline batch size linearly with the target GPU's VRAM.

    `default` is calibrated for a `baseline_gb` GPU (12GB by default); larger
    GPUs get a proportionally larger batch, smaller ones a proportionally
    smaller one (floor of 1). Pass --batch_size explicitly to skip this.
    """
    if not device.startswith("cuda") or not torch.cuda.is_available():
        return default

    idx = int(device.split(":")[1]) if ":" in device else 0
    total_gb = torch.cuda.get_device_properties(idx).total_memory / (1024**3)
    return max(1, int(default * (total_gb / baseline_gb)))


def compute_miou(pred_ids: torch.Tensor, target_ids: torch.Tensor, num_classes: int) -> float:
    """Mean IoU between two [H, W] (or any-shape, will be flattened) class-id
    maps, averaged only over classes present in either map -- a class absent
    from both isn't evidence of anything and shouldn't dilute the score."""
    pred_ids = pred_ids.flatten()
    target_ids = target_ids.flatten()

    ious = []
    for c in range(num_classes):
        pred_c = pred_ids == c
        target_c = target_ids == c
        if not pred_c.any() and not target_c.any():
            continue
        intersection = (pred_c & target_c).sum().item()
        union = (pred_c | target_c).sum().item()
        ious.append(intersection / union if union > 0 else 0.0)

    return sum(ious) / len(ious) if ious else 0.0


def print_gpu_diagnostics() -> None:
    """Print what torch actually sees, so a misconfigured environment (CPU-only
    torch build, wrong conda env, driver mismatch) is obvious at a glance in the
    log instead of silently training ~100x too slowly on CPU. Call once, on the
    main process only -- with `accelerate launch --num_processes=4` every rank
    would otherwise print its own interleaved copy."""
    print("=" * 60)
    print("  GPU DIAGNOSTICS")
    print(f"  torch            : {torch.__version__}")
    print(f"  torch.version.cuda: {torch.version.cuda}")
    print(f"  cuda available   : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  device count     : {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            print(f"    [{i}] {p.name}  |  {p.total_memory / 1024 ** 3:.1f} GB")
    else:
        print("  !! No CUDA device visible -- training will refuse to start.")
    print("=" * 60)


def write_training_params_txt(cfg: Any, output_path: Path, device: str, original_cwd: str = "") -> None:
    """Snapshot the full resolved config next to this run's checkpoints, so a
    result folder is self-documenting: months later you can look at a
    checkpoint and know exactly what produced it, without reconstructing the
    command line. Written BEFORE training starts, so even a crashed run leaves
    its recipe behind.

    Deliberately dumps the WHOLE resolved config rather than a hand-picked
    subset -- a hand-picked list silently goes stale the moment a new key is
    added, which is exactly when you would most want it recorded. This
    naturally includes cfg.prompt, data.train_jsonl/val_jsonl, and everything
    else -- nothing grounded_sam-specific needed here, it's dataset/branch-agnostic.
    """
    from omegaconf import OmegaConf

    output_path.mkdir(parents=True, exist_ok=True)
    lines = [
        "training run parameters",
        f"timestamp    : {datetime.now().isoformat(timespec='seconds')}",
        f"device       : {device}",
        f"original_cwd : {original_cwd}",
        f"output_path  : {output_path}",
        "",
        "----- resolved config -----",
        OmegaConf.to_yaml(cfg, resolve=True) if OmegaConf.is_config(cfg) else str(cfg),
    ]
    (output_path / "training_params.txt").write_text("\n".join(lines), encoding="utf-8")
