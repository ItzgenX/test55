from torch import nn
import torch
import torch.nn.functional as F
from functools import reduce
from einops import rearrange


class SimpleMapper(nn.Module):
    def __init__(self, d_model, c_dim):
        super().__init__()

        self.ls = nn.Sequential(nn.Linear(d_model, c_dim), nn.LayerNorm(c_dim))  # just [b, d] (no n as it's a single vector)

    def forward(self, x):
        return self.ls(x)


class FixedStructureMapper15(nn.Module):
    def __init__(self, c_dim: int):
        super().__init__()
        self.c_dim = c_dim

        self.down = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(16, 32, 3, padding=1, stride=2),  # 256
            nn.SiLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 64, 3, padding=1, stride=2),  # 128
            nn.SiLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(64, 128, 3, padding=1, stride=2),  # 64
            nn.SiLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.SiLU(),
            # nn.Conv2d(128, 128, 3, padding=1),
        )

        self.block0 = nn.Identity()

        self.block1 = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1, stride=2),  # 32
            nn.SiLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.SiLU(),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1, stride=2),  # 16
            nn.SiLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.SiLU(),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1, stride=2),  # 8
            nn.SiLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.SiLU(),
        )

        self.out0 = nn.Sequential(nn.Conv2d(128, c_dim, 1))
        self.out1 = nn.Sequential(nn.Conv2d(128, c_dim, 1))
        self.out2 = nn.Sequential(nn.Conv2d(128, c_dim, 1))
        self.out3 = nn.Sequential(nn.Conv2d(128, c_dim, 1))

    def forward(self, x, *args, **kwargs):
        base = self.down(x)

        b0 = self.block0(base)
        b1 = self.block1(b0)
        b2 = self.block2(b1)
        b3 = self.block3(b2)

        out0 = self.out0(b0)
        out1 = self.out1(b1)
        out2 = self.out2(b2)
        out3 = self.out3(b3)

        return out0, out1, out2, out3


class FixedStructureMapperXL(nn.Module):
    def __init__(self, c_dim: int):
        super().__init__()
        self.c_dim = c_dim

        self.down = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 64, 3, padding=1, stride=2),  # /2
            nn.SiLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(64, 128, 3, padding=1, stride=2),  # /4
            nn.SiLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.SiLU(),
            # /8 total -- matches the VAE's own downsample, so block0 (below)
            # lands on the SAME resolution as the UNet's actual stage-0 latent
            # feature map, not 4x too large. See field guide for the bug this fixes.
            nn.Conv2d(128, 128, 3, padding=1, stride=2),
            nn.SiLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.SiLU(),
        )

        self.block0 = nn.Identity()

        self.block1 = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1, stride=2),  # /16
            nn.SiLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.SiLU(),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1, stride=2),  # /32
            nn.SiLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.SiLU(),
        )

        self.out0 = nn.Sequential(nn.Conv2d(128, c_dim, 1))
        self.out1 = nn.Sequential(nn.Conv2d(128, c_dim, 1))
        self.out2 = nn.Sequential(nn.Conv2d(128, c_dim, 1))

    def forward(self, x, *args, **kwargs):
        base = self.down(x)

        b0 = self.block0(base)
        b1 = self.block1(b0)
        b2 = self.block2(b1)

        out0 = self.out0(b0)
        out1 = self.out1(b1)
        out2 = self.out2(b2)

        return out0, out1, out2


class SegIDStructureMapperXL(nn.Module):
    """Class-ID conditioning for grounded_sam -- replaces FixedStructureMapperXL's
    RGB-colorize-then-conv path, same reasoning and same embedding-stem
    pattern as the segformer branch's own SegIDStructureMapperXL (see that
    branch's decision log and field guide Lesson 29): segmentation
    classes are categorical, an embedding table gives each class its own
    learned vector directly instead of forcing the network to re-derive
    class identity from a palette color.

    Index scheme (num_classes=30 default), DIFFERENT from segformer's because
    GroundedSamEncoder can legitimately produce IGNORE_ID (255) for pixels no
    detection touched -- segformer's dense classifier never does:
      0      = NULL / padding_idx -- reserved for src/model.py's CFG dropout
               (c[dropout_mask] = 0), so a fully-dropped-out sample decodes
               to a true zero vector, not "class 0" (road).
      1..28  = the 28 real CARLA-vocabulary classes, shifted +1 from the raw
               0..27 ids GroundedSamEncoder/grounded_sam_map_calculations.py save.
      29     = IGNORE/void -- a DEDICATED row, distinct from NULL (0). This
               matters: IGNORE_ID means "this specific pixel had no
               detection," a real per-pixel signal that should stay visible
               to the network, not the same thing as "the whole conditioning
               map was CFG-dropped." Collapsing both onto row 0 would erase
               that distinction.

    Fully convolutional, same three-stage /8-/16-/32 output as
    FixedStructureMapperXL (SDXL's three UNet stages).
    """

    def __init__(self, c_dim: int, num_classes: int = 30, embed_dim: int = 32):
        super().__init__()
        self.c_dim = c_dim
        self.cls_emb = nn.Embedding(num_classes, embed_dim, padding_idx=0)

        ch = embed_dim + 1  # +1 derived boundary channel
        self.down = nn.Sequential(
            nn.Conv2d(ch, 64, 3, padding=1), nn.SiLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.SiLU(),
            nn.Conv2d(64, 128, 3, padding=1, stride=2), nn.SiLU(),   # /2
            nn.Conv2d(128, 128, 3, padding=1), nn.SiLU(),
            nn.Conv2d(128, 128, 3, padding=1, stride=2), nn.SiLU(),  # /4
            nn.Conv2d(128, 128, 3, padding=1), nn.SiLU(),
            nn.Conv2d(128, 128, 3, padding=1, stride=2), nn.SiLU(),  # /8
            nn.Conv2d(128, 128, 3, padding=1), nn.SiLU(),
        )
        self.block0 = nn.Identity()
        self.block1 = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1, stride=2), nn.SiLU(),  # /16
            nn.Conv2d(128, 128, 3, padding=1), nn.SiLU(),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1, stride=2), nn.SiLU(),  # /32
            nn.Conv2d(128, 128, 3, padding=1), nn.SiLU(),
        )
        self.out0 = nn.Conv2d(128, c_dim, 1)
        self.out1 = nn.Conv2d(128, c_dim, 1)
        self.out2 = nn.Conv2d(128, c_dim, 1)

    @staticmethod
    def _boundary(idx: torch.Tensor) -> torch.Tensor:
        """[B,H,W] long -> [B,1,H,W] float in {0,1}: 1 where a 3x3
        neighbourhood spans more than one class id (a class edge)."""
        f = idx.unsqueeze(1).float()
        mx = F.max_pool2d(f, 3, 1, 1)
        mn = -F.max_pool2d(-f, 3, 1, 1)
        return (mx != mn).float()

    def forward(self, c: torch.Tensor, *args, **kwargs):
        """c: Long [B,H,W] or [B,1,H,W] class ids (1..28 real, 29=IGNORE,
        0=NULL). Matches the skip_encode=True contract: cond = c straight
        from the dataset's precomputed map, no colorization step in between."""
        c = c.long()
        if c.dim() == 4:
            c = c[:, 0]
        cls = self.cls_emb(c).permute(0, 3, 1, 2)  # [B,embed_dim,H,W]
        bnd = self._boundary(c)
        x = torch.cat([cls, bnd], dim=1).contiguous()

        b0 = self.block0(self.down(x))
        b1 = self.block1(b0)
        b2 = self.block2(b1)

        return self.out0(b0), self.out1(b1), self.out2(b2)


# we don't have attention in the deepest blocks
# so we only have three outputs here for SD15
class AttentionStructureMapper15(nn.Module):
    def __init__(self, c_dim: int):
        super().__init__()
        self.c_dim = c_dim

        self.down = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(16, 32, 3, padding=1, stride=2),  # 256
            nn.SiLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 64, 3, padding=1, stride=2),  # 128
            nn.SiLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(64, 128, 3, padding=1, stride=2),  # 64
            nn.SiLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.SiLU(),
            # nn.Conv2d(128, 128, 3, padding=1),
        )

        # the output channels correspond to the token dim
        self.block0 = nn.Identity()

        self.block1 = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1, stride=2),  # 32
            nn.SiLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.SiLU(),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1, stride=2),  # 16
            nn.SiLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.SiLU(),
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1, stride=2),  # 8
            nn.SiLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.SiLU(),
        )

        # here we project them down again
        self.out0 = nn.Sequential(nn.Conv2d(128, c_dim, 1))
        self.out1 = nn.Sequential(nn.Conv2d(128, c_dim, 1))
        self.out2 = nn.Sequential(nn.Conv2d(128, c_dim, 1))
        self.out3 = nn.Sequential(nn.Conv2d(128, c_dim, 1))

    def forward(self, x, *args, **kwargs):
        base = self.down(x)

        b0 = self.block0(base)
        b1 = self.block1(b0)
        b2 = self.block2(b1)
        b3 = self.block3(b2)

        out0 = self.out0(b0)
        out1 = self.out1(b1)
        out2 = self.out2(b2)
        out3 = self.out3(b3)

        # convert to tokens
        ot0 = rearrange(out0, "B C H W -> B (H W) C")
        ot1 = rearrange(out1, "B C H W -> B (H W) C")
        ot2 = rearrange(out2, "B C H W -> B (H W) C")
        ot3 = rearrange(out3, "B C H W -> B (H W) C")

        return ot0, ot1, ot2, ot3
