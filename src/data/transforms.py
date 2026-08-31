from torchvision import transforms


def normalize_size(size) -> tuple[int, int]:
    """int -> (size, size); a (width, height) pair -> itself as ints.
    Always returns (width, height) -- the convention used across the
    segmentation pipeline (seg_map_calculations.py, segformer_inference.py,
    local_seg.py)."""
    if isinstance(size, int):
        return (size, size)
    w, h = size
    return (int(w), int(h))


def build_seg_preprocess(size, resize_mode: str = "aspect"):
    """RGB PIL image -> [3, H, W] tensor in [-1, 1].

    resize_mode="aspect" is the only supported mode: a direct resize to an
    explicit (width, height) target, no pad, no crop -- so nothing spurious
    (a pad band) gets baked into what the model learns, and no real scene
    content (a crop) gets thrown away. This is the single source of truth
    for RGB geometry, shared by seg_map_calculations.py (offline map
    computation) and local_seg.py (training's dataset loader), so a seg
    map's geometry always matches the RGB image it's paired with.
    """
    assert resize_mode == "aspect", f"unsupported resize_mode: {resize_mode!r}"
    w, h = normalize_size(size)
    return transforms.Compose(
        [
            transforms.Resize((h, w)),  # torchvision Resize takes (H, W)
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )


def build_seg_display_preprocess(size, resize_mode: str = "aspect"):
    """RGB PIL image -> resized RGB PIL image (no tensor, no normalize).

    Used only for inference's "ORIGINAL" display panel, which is never fed
    to any model -- just needs to visually line up with the SEG MAP panel.
    """
    assert resize_mode == "aspect", f"unsupported resize_mode: {resize_mode!r}"
    w, h = normalize_size(size)

    def _resize(img):
        return img.resize((w, h))

    return _resize
