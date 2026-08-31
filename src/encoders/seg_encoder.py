import torch
from torch import nn
import torch.nn.functional as F
from jaxtyping import Float
from transformers import SegformerForSemanticSegmentation

from src.data.transforms import normalize_size

# cityscapesScripts' standard 19-class train-id color palette, in class-id order.
# This is the single source of truth for id<->colour everywhere in the
# segmentation pipeline: seg_map_calculations.py (offline computation),
# local_seg.py (training's dataset loader), segformer_inference.py (display
# + mIoU). Every one of them imports it from here rather than keeping its
# own copy, so a palette change can never desync two parts of the pipeline.
SEG_CITYSCAPES_PALETTE = [
    (128, 64, 128),
    (244, 35, 232),
    (70, 70, 70),
    (102, 102, 156),
    (190, 153, 153),
    (153, 153, 153),
    (250, 170, 30),
    (220, 220, 0),
    (107, 142, 35),
    (152, 251, 152),
    (70, 130, 180),
    (220, 20, 60),
    (255, 0, 0),
    (0, 0, 142),
    (0, 0, 70),
    (0, 60, 100),
    (0, 80, 100),
    (0, 0, 230),
    (119, 11, 32),
]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def seg_palette_tensor() -> torch.Tensor:
    """[num_classes, 3] float in [0, 1] -- the id<->colour lookup table."""
    return torch.tensor(SEG_CITYSCAPES_PALETTE, dtype=torch.float32) / 255.0


def seg_colorize_ids(ids: torch.Tensor, palette: torch.Tensor) -> torch.Tensor:
    """Class ids -> colour map. ids: [H, W] or [B, H, W] long -> [B, 3, H, W] float in [0, 1]."""
    if ids.dim() == 2:
        ids = ids.unsqueeze(0)
    color = palette[ids]  # [B, H, W, 3]
    return color.permute(0, 3, 1, 2).contiguous()


def seg_ids_from_colormap(color: torch.Tensor, palette: torch.Tensor) -> torch.Tensor:
    """Inverse of seg_colorize_ids: nearest-palette-color lookup.
    color: [3, H, W] or [B, 3, H, W] float in [0, 1] -> class ids, same
    leading batch shape as the input (squeezed back out if it was added)."""
    squeeze = color.dim() == 3
    if squeeze:
        color = color.unsqueeze(0)

    flat = color.permute(0, 2, 3, 1).unsqueeze(-2)  # [B, H, W, 1, 3]
    dist = (flat - palette.view(1, 1, 1, -1, 3)).pow(2).sum(-1)  # [B, H, W, num_classes]
    ids = dist.argmin(-1)  # [B, H, W]

    return ids[0] if squeeze else ids


class SegmentationEncoder(nn.Module):
    """Frozen SegFormer semantic segmentation encoder.

    Same [-1, 1]-in contract as every other module in src/annotators (see
    midas.py). Unlike depth, the primary output is class ids (label_ids()),
    not a colour map directly: seg_map_calculations.py saves raw ids as an
    8-bit PNG, and callers colourise with seg_colorize_ids + seg_palette_tensor
    so the exact same palette lookup runs everywhere (calc time, dataset load
    time, inference display time) -- never three separate copies of "what
    colour is class 7" that could drift apart.

    Does not force a square crop before inference: SegFormer's Mix-FFN has no
    fixed positional embedding, so it handles arbitrary (H, W) natively.

    live_available=True: this encoder has a real live inference path
    (label_ids() can run on any image, including a freshly generated one),
    which is what lets segformer_inference.py's mIoU controllability metric
    run. An encoder with no live path (e.g. a box-prompted, vocabulary-driven
    source) would set this False so that scoring step is skipped instead of
    crashing.
    """

    def __init__(
        self,
        size,
        model: str = "nvidia/segformer-b5-finetuned-cityscapes-1024-1024",
        local_files_only: bool = True,
    ) -> None:
        super().__init__()
        self.model_name = model
        self.size = normalize_size(size)  # (width, height)
        self.live_available = True

        self.segformer = SegformerForSemanticSegmentation.from_pretrained(model, local_files_only=local_files_only)
        self.segformer.requires_grad_(False)
        self.segformer.eval()
        self.num_classes = self.segformer.config.num_labels

        self.register_buffer("palette", seg_palette_tensor(), persistent=False)
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False)

    @torch.no_grad()
    def label_ids(self, imgs: Float[torch.Tensor, "B C H W"]) -> torch.Tensor:
        """[-1, 1] images -> [B, H, W] long class ids in [0, num_classes-1]."""
        assert imgs.min() >= -1.0
        assert imgs.max() <= 1.0
        assert len(imgs.shape) == 4

        imgs01 = (imgs + 1.0) / 2.0
        normed = (imgs01 - self.mean) / self.std

        logits = self.segformer(pixel_values=normed).logits  # [B, num_classes, h', w'], h'/w' < H, W
        w, h = self.size
        logits = F.interpolate(logits, size=(h, w), mode="bilinear", align_corners=False)
        return logits.argmax(dim=1)

    @torch.no_grad()
    def forward(self, imgs: Float[torch.Tensor, "B C H W"]) -> Float[torch.Tensor, "B C H W"]:
        """[-1, 1] images -> [B, 3, H, W] colourised class map in [0, 1]."""
        return seg_colorize_ids(self.label_ids(imgs), self.palette)
