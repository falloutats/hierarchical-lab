"""A small CNN backbone + three heads, one per stage of the pathway.

We deliberately swap the repo's ResNet50+CBAM for a compact CNN so an epoch is
seconds on MPS and you can iterate on the *ideas*. The head shapes are what
carry the lessons:

  FlatModel        backbone -> 100                      (no hierarchy at all)
  TwoHeadModel     backbone -> {20, 100}                (deep supervision / lloss)
  CoarseToFineModel backbone -> 20, then cat(20,100)->100  (the repo's conditioning)

Every model exposes `.features(x)` returning the pooled penultimate vector, so
Stage 5 can watch the representation organize by superclass.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from hierarchy import NUM_COARSE, NUM_FINE

FEATURE_DIM = 256


def _block(cin: int, cout: int) -> nn.Sequential:
    """conv-BN-ReLU twice, then halve the spatial size.

    bias=False on the convolutions is not a typo: BatchNorm immediately
    recentres its input to zero mean and then applies its own learnable shift
    beta. Any bias the conv added would be subtracted right back out, so it is a
    parameter that cannot affect the output -- pure dead weight in the gradient.

    Two 3x3 convolutions before pooling rather than one 5x5: both see a 5x5
    receptive field, but stacking costs 2*(3*3)=18 weights per channel pair
    instead of 25 and inserts a nonlinearity in the middle, so the composition is
    strictly more expressive than the single linear 5x5 filter it replaces.
    """
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class Backbone(nn.Module):
    """32x32x3 image -> a single FEATURE_DIM vector. Shared by all three heads.

    Spatial size halves at each block (32 -> 16 -> 8 -> 4) while channel depth
    grows (3 -> 64 -> 128 -> 256): the usual trade of *where* something is for
    *what* it is.

    The final AdaptiveAvgPool2d(1) averages each channel's 4x4 map down to one
    number, turning a (C, H, W) tensor into a C-vector. Read as linear algebra
    that is an inner product of each channel plane with the uniform vector
    (1/HW)*ones -- a projection onto the constant mode that throws away all
    spatial arrangement. That is what buys translation invariance: shift the
    object in the image and the mean over the plane is unchanged. We use the
    adaptive form rather than AvgPool2d(4) so the same module still works if the
    input resolution changes; it solves for the kernel size from the output size.
    """

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            _block(3, 64),
            _block(64, 128),
            _block(128, FEATURE_DIM),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x)
        # flatten(1) collapses the trailing (1, 1) spatial dims but preserves the
        # batch axis -- reshape(-1) would silently fuse the batch together.
        return self.pool(x).flatten(1)


class FlatModel(nn.Module):
    """Stage 1: plain 100-way classifier. No idea the hierarchy exists."""

    def __init__(self) -> None:
        super().__init__()
        self.backbone = Backbone()
        self.fine = nn.Linear(FEATURE_DIM, NUM_FINE)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fine(self.backbone(x))


class TwoHeadModel(nn.Module):
    """Stage 2: coarse + fine heads on a shared trunk (deep supervision)."""

    def __init__(self) -> None:
        super().__init__()
        self.backbone = Backbone()
        self.coarse = nn.Linear(FEATURE_DIM, NUM_COARSE)
        self.fine = nn.Linear(FEATURE_DIM, NUM_FINE)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def forward(self, x: torch.Tensor):
        z = self.backbone(x)
        return self.coarse(z), self.fine(z)


class CoarseToFineModel(nn.Module):
    """Stage 3+: the coarse logits are fed into the fine head. The repo's key move.

    Faithful to their model/resnet50.py:
        level_1 = softmax_reg1(linear_lvl1(z))                # feat -> 20 -> 20
        level_2 = softmax_reg2(cat(level_1, linear_lvl2(z)))  # cat(20,100) -> 100

    Why concatenating-then-one-Linear is genuinely "conditioning", in one line of
    block-matrix algebra. Write the fine head's weight W (100 x 120) as two blocks
    side by side, W = [A | B] with A of shape 100x20 and B of shape 100x100. Then

        W @ [level_1 ; linear_lvl2(z)] + b  =  A @ level_1  +  B @ linear_lvl2(z) + b

    Concatenating the inputs and applying ONE linear map is identically equal to
    applying two separate linear maps and adding. So the fine logits are the usual
    feature-driven logits (B term) plus a learned correction driven purely by the
    coarse opinion (A term). Column j of A is exactly "how believing in superclass
    j shifts every fine logit" -- the model can learn to add evidence to the five
    children of the predicted parent and subtract it elsewhere.

    Note this is an additive, linear coupling: it biases the fine logits but
    cannot *forbid* an inconsistent answer, which is why the architecture alone
    does not drive the violation rate to zero and the loss still has work to do.

    It also adds no expressive power. level_1 is a linear function of z (two
    Linear layers with no nonlinearity between them), so A @ level_1 is too, and
    the fine logits are still one affine function of z -- exactly the function
    class of TwoHeadModel's fine head. What changes is only the parameterization:
    the coarse loss now shapes weights the fine head also uses. That is a
    plausible reason Stage 3 does not beat Stage 2 in our runs. (Feeding in
    softmax(level_1) instead would make the coupling genuinely nonlinear.)

    We return raw logits and let the loss apply softmax -- matching their
    pre-softmax concatenation, and keeping the numerically stable fused
    log-softmax inside cross_entropy rather than doing softmax twice.
    """

    def __init__(self) -> None:
        super().__init__()
        self.backbone = Backbone()
        self.linear_lvl1 = nn.Linear(FEATURE_DIM, NUM_COARSE)
        self.softmax_reg1 = nn.Linear(NUM_COARSE, NUM_COARSE)
        self.linear_lvl2 = nn.Linear(FEATURE_DIM, NUM_FINE)
        self.softmax_reg2 = nn.Linear(NUM_COARSE + NUM_FINE, NUM_FINE)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def forward(self, x: torch.Tensor):
        z = self.backbone(x)
        level_1 = self.softmax_reg1(self.linear_lvl1(z))
        level_2 = self.softmax_reg2(torch.cat((level_1, self.linear_lvl2(z)), dim=1))
        return level_1, level_2
