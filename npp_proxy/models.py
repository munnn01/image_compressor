"""Author NPP modules with explicit RGB padding and QP conditioning."""

import math

import torch
from torch import nn
from torch.nn import functional as F

from .vendor.npp_fixed.Preprocessor import Preprocessor as FixedPreprocessor
from .vendor.npp_adaptive.Preprocessor import Preprocessor as AdaptivePreprocessor


def pad_rgb(x, multiple):
    h, w = x.shape[-2:]
    return F.pad(x, (0, max(multiple, math.ceil(w / multiple) * multiple) - w,
                     0, max(multiple, math.ceil(h / multiple) * multiple) - h), mode='replicate')


class NPP(nn.Module):
    def __init__(self, adaptive=False):
        super().__init__()
        self.adaptive = adaptive
        self.network = AdaptivePreprocessor() if adaptive else FixedPreprocessor()

    def load_fixed(self, state):
        if not self.adaptive:
            raise ValueError('Fixed initialization requires adaptive NPP')
        raw = {k.removeprefix('network.'): v for k, v in state.items()}
        expected = {k for k in self.network.state_dict() if 'Scaling_Net' not in k}
        if set(raw) != expected:
            raise ValueError('A complete author fixed NPP checkpoint is required')
        combined = self.network.state_dict()
        combined.update(raw)
        self.network.load_state_dict(combined, strict=True)

    def forward(self, x, qp):
        h, w = x.shape[-2:]
        x = pad_rgb(x, 16)
        if self.adaptive:
            condition = torch.as_tensor(qp, device=x.device, dtype=x.dtype).reshape(-1) / 51
            if condition.numel() != 1:
                raise ValueError('Author QA module accepts one QP per batch')
            x = self.network(x, condition)
        else:
            x = self.network(x)
        return x[..., :h, :w].clamp(0, 1)
