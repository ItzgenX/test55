import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from jaxtyping import Float
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection, SamModel, SamProcessor

from src.data.transforms import normalize_size
from src.encoders.seg_encoder import seg_colorize_ids

# Own 28-class vocabulary, NOT SegFormer's 19-class Cityscapes scheme (that
# scheme and its own live encoder live only in the segformer branch's copy of
# this codebase, not this one) -- segformer has its own fixed pipeline matched
# to what SegFormer-B5 was trained on, unrelated to this. This branch's photos
# still come from real Cityscapes images (data/dataset/extracted/leftImg8bit);
# only the VOCABULARY Grounding DINO searches for changed, taken from CARLA's
# own semantic segmentation camera class list (carla.readthedocs.io/en/latest/
# ref_sensors/#semantic-segmentation-camera) -- 29 classes there, 28 here
# because CARLA's "Unlabeled" (id 0) isn't a real detectable object, it's
# CARLA's own equivalent of "nothing here", which this pipeline already
# represents via IGNORE_ID below -- would be meaningless as a DINO query.
#
# Ids here are THIS pipeline's own sequential numbering (0-27, palette row
# order) -- NOT CARLA's own numbering (which starts real classes at 1). Only
# the NAMES and RGB COLOURS are borrowed from CARLA's spec; the id scheme is
# local to keep this self-contained, same pattern segformer's own 19-class
# scheme uses for its palette.
#
# Two entries use CARLA's own natural-language rewording rather than its
# exact class name, so the phrase actually reads as a describable visual
# thing for an open-vocab detector to search for: "RoadLine" -> "lane
# marking" (closer to how a person would actually describe it), "Pedestrian"
# -> "person" (matches the 19-class scheme's own existing choice).
#
# HONEST CAVEAT, not hidden: "static object", "dynamic object", and "other"
# (CARLA's Static/Dynamic/Other) are not concrete visual concepts the way
# "car" or "sky" are -- Grounding DINO has nothing specific to latch onto for
# these three, so expect meaningfully weaker/noisier detection on them than
# on the other 25 classes. Kept in because the full 29-class list was the
# explicit ask, not because they're expected to detect well.
CARLA_CLASSES = [
    "road",            # 0  Roads          (128, 64, 128)
    "sidewalk",        # 1  SideWalks      (244, 35, 232)
    "building",        # 2  Building       (70, 70, 70)
    "wall",            # 3  Wall           (102, 102, 156)
    "fence",           # 4  Fence          (190, 153, 153)
    "pole",            # 5  Pole           (153, 153, 153)
    "traffic light",   # 6  TrafficLight   (250, 170, 30)
    "traffic sign",    # 7  TrafficSign    (220, 220, 0)
    "vegetation",      # 8  Vegetation     (107, 142, 35)
    "terrain",         # 9  Terrain        (152, 251, 152)
    "sky",             # 10 Sky            (70, 130, 180)
    "person",          # 11 Pedestrian     (220, 20, 60)
    "rider",           # 12 Rider          (255, 0, 0)
    "car",             # 13 Car            (0, 0, 142)
    "truck",           # 14 Truck          (0, 0, 70)
    "bus",             # 15 Bus            (0, 60, 100)
    "train",           # 16 Train          (0, 80, 100)
    "motorcycle",      # 17 Motorcycle     (0, 0, 230)
    "bicycle",         # 18 Bicycle        (119, 11, 32)
    "static object",   # 19 Static         (110, 190, 160)  -- vague, see caveat above
    "dynamic object",  # 20 Dynamic        (170, 120, 50)   -- vague, see caveat above
    "other",           # 21 Other          (55, 90, 80)     -- vague, see caveat above
    "water",           # 22 Water          (45, 60, 150)
    "lane marking",    # 23 RoadLine       (157, 234, 50)
    "ground",          # 24 Ground         (81, 0, 81)
    "bridge",          # 25 Bridge         (150, 100, 100)
    "rail track",      # 26 RailTrack      (230, 150, 140)
    "guard rail",      # 27 GuardRail      (180, 165, 180)
]

# Same RGB values CARLA's own docs publish for each class, in CARLA_CLASSES
# order -- used as-is rather than inventing new colours, so a rendered map
# here looks like CARLA's own visualization convention if anyone compares.
CARLA_PALETTE = [
    (128, 64, 128), (244, 35, 232), (70, 70, 70), (102, 102, 156),
    (190, 153, 153), (153, 153, 153), (250, 170, 30), (220, 220, 0),
    (107, 142, 35), (152, 251, 152), (70, 130, 180), (220, 20, 60),
    (255, 0, 0), (0, 0, 142), (0, 0, 70), (0, 60, 100), (0, 80, 100),
    (0, 0, 230), (119, 11, 32), (110, 190, 160), (170, 120, 50),
    (55, 90, 80), (45, 60, 150), (157, 234, 50), (81, 0, 81),
    (150, 100, 100), (230, 150, 140), (180, 165, 180),
]
assert len(CARLA_CLASSES) == len(CARLA_PALETTE)


def carla_palette_tensor() -> torch.Tensor:
    """[28, 3] float in [0, 1] -- the id<->colour lookup table for THIS
    encoder's own 28-class scheme. Deliberately separate from segformer
    branch's own 19-class Cityscapes palette -- the two branches no longer
    share a class taxonomy, only the generic colourise/decolourise math
    (seg_colorize_ids, imported below) is shared."""
    return torch.tensor(CARLA_PALETTE, dtype=torch.float32) / 255.0


# Countability-based split, same principle Cityscapes' own official
# stuff/thing distinction uses (not motion-based) -- matches this encoder's
# previous 19-class version exactly for the 19 overlapping classes, extended
# the same way for the 9 new ones: Static/Dynamic are discrete countable
# objects (fire hydrants, trash bins) so THING; Water/RoadLine/Ground/Bridge/
# RailTrack/GuardRail/Other are amorphous surfaces/structures/catch-all so
# STUFF, same role their closest existing neighbours (road, fence) already play.
STUFF_CLASSES = {
    "road", "sidewalk", "building", "wall", "fence", "pole",
    "traffic light", "traffic sign", "vegetation", "terrain", "sky",
    "other", "water", "lane marking", "ground", "bridge", "rail track", "guard rail",
}
THING_CLASSES = set(CARLA_CLASSES) - STUFF_CLASSES

IGNORE_ID = 255  # unmatched pixels -- same convention Cityscapes' own gtFine uses,
# already handled correctly everywhere downstream (compute_miou, check_seg_accuracy.py).

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def _match_class(label_text: str) -> str | None:
    """Grounding DINO's phrase-grounding decoder doesn't always echo the exact
    input phrase back (e.g. might return "cars" for "car", or a slightly
    reworded span) -- match case-insensitively, longest class name first so
    "traffic light" doesn't get swallowed by a looser match on "light"."""
    t = label_text.lower().strip()
    for cls in sorted(CARLA_CLASSES, key=len, reverse=True):
        if cls in t:
            return cls
    return None


class GroundedSamEncoder(nn.Module):
    """Grounding DINO (open-vocabulary box detection) + SAM (box -> precise
    mask) composited into a dense raw-class-ID map, own 28-class vocabulary
    (see CARLA_CLASSES above -- borrowed from CARLA's own semantic
    segmentation camera class list, NOT SegFormer's 19-class scheme; the two
    branches use different taxonomies now, only the mechanism is shared).

    Structurally very different from SegFormer under the hood: SegFormer is
    one dense classifier that hands back a full per-pixel prediction in a
    single forward pass. This is two models (detector + mask head) whose
    outputs have to be composited by hand -- STUFF classes (road, sky,
    building, ...) painted first as a background layer, THING classes (car,
    person, ...) painted on top so they visibly occlude the background the
    way they actually do in a real scene, each tier sorted by mask area
    (larger first) so small objects/regions aren't erased by bigger ones
    painted after them. Pixels no detection ever touches stay IGNORE_ID
    (255) -- same "not one of the scored classes" convention Cityscapes'
    own ground truth uses, already handled correctly by every downstream
    consumer of this pipeline's raw-ID PNGs.

    live_available=True: has a genuine live inference path (label_ids() runs
    on any image, including a freshly generated one), same contract as
    segformer branch's own live encoder -- lets segformer_inference.py-style
    mIoU controllability scoring work unmodified against this encoder too.
    """

    def __init__(
        self,
        size,
        dino_model: str = "IDEA-Research/grounding-dino-tiny",
        sam_model: str = "facebook/sam-vit-base",
        # 0.3/0.25 (grounding-dino-tiny's own README defaults) visibly left
        # large swaths of "stuff" regions (sky, building, road) undetected
        # on real test images. 0.15/0.15 measured 92.6%/89.7% pixel coverage
        # on two different images with no obvious quality loss (see field
        # guide Lesson 21) -- validated across 3 separate successful runs.
        # DO NOT go lower without real investigation first: 0.1/0.1 caused
        # a genuine segfault (SIGSEGV, exit 139) partway through processing,
        # not just noisier detections -- reproduced once, not yet root-caused
        # (likely SAM's mask decoder choking on an unusually large box count
        # from the more permissive threshold, but unconfirmed). These
        # thresholds were tuned against the 19-class vocabulary -- with 28
        # classes now queried per image, re-check coverage/stability before
        # trusting them unchanged at scale.
        box_threshold: float = 0.15,
        text_threshold: float = 0.15,
        local_files_only: bool = True,
    ) -> None:
        super().__init__()
        self.size = normalize_size(size)  # (width, height)
        self.live_available = True
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.num_classes = len(CARLA_CLASSES)

        # Grounding DINO wants ". "-joined lowercase phrases, one query per class.
        self._dino_text = ". ".join(CARLA_CLASSES) + "."

        self.dino_processor = AutoProcessor.from_pretrained(dino_model, local_files_only=local_files_only)
        self.dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(
            dino_model, local_files_only=local_files_only
        )
        self.dino_model.requires_grad_(False)
        self.dino_model.eval()

        self.sam_processor = SamProcessor.from_pretrained(sam_model, local_files_only=local_files_only)
        self.sam_model = SamModel.from_pretrained(sam_model, local_files_only=local_files_only)
        self.sam_model.requires_grad_(False)
        self.sam_model.eval()

        self.register_buffer("palette", carla_palette_tensor(), persistent=False)
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False)

    def _label_ids_single(self, img_pil: Image.Image, device) -> torch.Tensor:
        """One image -> [H, W] long class ids (IGNORE_ID where nothing was detected).
        H, W = self.size (height, width), matching the rest of the pipeline's
        NEAREST-resize-to-target convention -- detection/masking runs at the
        image's own native resolution, then the composited id map is resized
        (NEAREST, to not fabricate classes at boundaries) to the target size."""
        w0, h0 = img_pil.size

        dino_inputs = self.dino_processor(images=img_pil, text=self._dino_text, return_tensors="pt").to(device)
        with torch.no_grad():
            dino_out = self.dino_model(**dino_inputs)

        results = self.dino_processor.post_process_grounded_object_detection(
            dino_out,
            threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            target_sizes=[(h0, w0)],
        )[0]

        boxes = results["boxes"]  # [N, 4] xyxy, pixel coords at native (w0,h0)
        raw_labels = results.get("text_labels", results.get("labels"))

        canvas = torch.full((h0, w0), IGNORE_ID, dtype=torch.long)

        if boxes.shape[0] == 0:
            return F.interpolate(
                canvas[None, None].float(), size=(self.size[1], self.size[0]), mode="nearest"
            ).long()[0, 0]

        sam_inputs = self.sam_processor(img_pil, input_boxes=[boxes.cpu().tolist()], return_tensors="pt").to(device)
        with torch.no_grad():
            sam_out = self.sam_model(**sam_inputs, multimask_output=False)
        masks = self.sam_processor.image_processor.post_process_masks(
            sam_out.pred_masks.cpu(),
            sam_inputs["original_sizes"].cpu(),
            sam_inputs["reshaped_input_sizes"].cpu(),
        )[0]  # [N, 1, h0, w0] bool/float

        detections = []  # (class_name, mask[h0,w0] bool, area)
        for i in range(boxes.shape[0]):
            cls = _match_class(str(raw_labels[i]))
            if cls is None:
                continue
            m = masks[i, 0].bool()
            area = int(m.sum().item())
            if area == 0:
                continue
            detections.append((cls, m, area))

        stuff = sorted([d for d in detections if d[0] in STUFF_CLASSES], key=lambda d: -d[2])
        things = sorted([d for d in detections if d[0] in THING_CLASSES], key=lambda d: -d[2])

        # Stuff first (background layer), things last (occlude on top) -- within
        # each tier, biggest first so smaller regions/objects painted after
        # aren't erased by a bigger one painted later in the same tier.
        for cls, mask, _area in stuff + things:
            canvas[mask] = CARLA_CLASSES.index(cls)

        canvas = F.interpolate(
            canvas[None, None].float(), size=(self.size[1], self.size[0]), mode="nearest"
        ).long()[0, 0]
        return canvas

    @torch.no_grad()
    def label_ids(self, imgs: Float[torch.Tensor, "B C H W"]) -> torch.Tensor:
        """[-1, 1] images -> [B, H, W] long class ids in [0, num_classes-1] or IGNORE_ID."""
        assert imgs.min() >= -1.0
        assert imgs.max() <= 1.0
        assert len(imgs.shape) == 4

        device = imgs.device
        imgs01 = ((imgs + 1.0) / 2.0).clamp(0, 1)

        out = []
        for b in range(imgs.shape[0]):
            arr = (imgs01[b].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            img_pil = Image.fromarray(arr)
            out.append(self._label_ids_single(img_pil, device))
        return torch.stack(out).to(device)

    @torch.no_grad()
    def forward(self, imgs: Float[torch.Tensor, "B C H W"]) -> Float[torch.Tensor, "B C H W"]:
        """[-1, 1] images -> [B, 3, H, W] colourised class map in [0, 1].
        IGNORE_ID has no palette entry (palette only covers the 28 real
        classes) and reusing an existing class's colour for it would be
        misleading -- an "unlabeled" pixel would render identically to
        "bicycle". Paint those pixels solid black instead, the standard
        segmentation-visualization convention for void/ignore regions."""
        ids = self.label_ids(imgs)
        ignore_mask = ids == IGNORE_ID
        ids_safe = torch.where(ignore_mask, torch.zeros_like(ids), ids)
        colour = seg_colorize_ids(ids_safe, self.palette.to(ids.device))
        colour = torch.where(ignore_mask.unsqueeze(1), torch.zeros_like(colour), colour)
        return colour
