from torchvision.transforms.v2 import Transform
import torchvision.transforms.v2.functional as F
from torchvision import transforms


class SquarePad(Transform):
    # use standard pad transform of v2
    # but always pads it to be a square

    def __init__(self):
        super().__init__()

        # self.fill = fill
        # self.padding_mode = padding_mode

    def _transform(self, inpt, params):
        h, w = inpt.shape[-2], inpt.shape[-1]

        if h > w:
            padding = [h - w // 2, 0, h - w // 2, 0]
        else:
            padding = [0, w - h // 2, 0, w - h // 2]

        return F.pad(inpt, padding, fill=255)


class TopCrop(Transform):
    # use standard crop transform of v2
    # but always crops from the top

    def __init__(self, size):
        super().__init__()
        self.size = size

    def _transform(self, inpt, params):
        return F.crop(inpt, 0, 0, self.size, self.size)


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
