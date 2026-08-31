import torch

# On this branch (grounded_sam), this file exists ONLY for the two generic,
# palette-parameterized id<->colour helpers below -- every caller here passes
# grounded_sam_encoder.py's own carla_palette_tensor(), never a Cityscapes
# palette. The SegFormer-specific live encoder (SegmentationEncoder) and its
# 19-class SEG_CITYSCAPES_PALETTE that used to live in this file belong to
# the segformer branch's own copy, not this one -- removed here since nothing
# on this branch ever instantiated them (no config references
# SegmentationEncoder; the only encoder config on this branch is
# configs/lora/encoder/grounded_sam.yaml).


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
