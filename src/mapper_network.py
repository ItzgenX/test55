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
            nn.Conv2d(32, 128, 3, padding=1, stride=2),  # 256
            nn.SiLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.SiLU(),
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


class DepthStructureMapperXL(nn.Module):
    """Input Float [B,3,H,W] in [0,1] -> feature maps at H/8, H/16, H/32.

    Depth-branch analogue of FixedStructureMapperXL, per
    DEPTH-BRANCH-architecture.md §4: FixedStructureMapperXL.down has only one
    stride-2 conv (outputs at H/2,H/4,H/8), which silently expects the
    condition at 1/4 image resolution -- wrong for a full-res depth map fed
    at H/8,H/16,H/32 to match SDXL's three UNet stages. This trunk has three
    stride-2 stages instead. Conv stem (not an embedding stem): depth is
    continuous 3-channel float, not categorical.
    """

    def __init__(self, c_dim: int = 128):
        super().__init__()
        self.c_dim = c_dim
        self.down = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(64, 128, 3, padding=1, stride=2),  # /2
            nn.SiLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(128, 128, 3, padding=1, stride=2),  # /4
            nn.SiLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(128, 128, 3, padding=1, stride=2),  # /8
            nn.SiLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.SiLU(),
        )
        self.block0 = nn.Identity()
        self.block1 = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1, stride=2),
            nn.SiLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.SiLU(),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1, stride=2),
            nn.SiLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.SiLU(),
        )
        self.out0 = nn.Conv2d(128, c_dim, 1)
        self.out1 = nn.Conv2d(128, c_dim, 1)
        self.out2 = nn.Conv2d(128, c_dim, 1)

    def forward(self, x, *args, **kwargs):
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
