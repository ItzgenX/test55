import torch

# Generic, palette-parameterized class-id <-> colour helpers, used for
# DISPLAY ONLY (checkpoint monitoring grids, inference comparison grids,
# dataset-loading preview fallback) -- the model's actual conditioning input
# never goes through these, it stays raw class ids the whole way (see
# SegIDStructureMapperXL). Every caller on this branch passes
# grounded_sam_encoder.py's own carla_palette_tensor(); the palette is a
# parameter here so the same two functions work for any class scheme.


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
