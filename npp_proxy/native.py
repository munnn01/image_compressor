"""Adapter for the author's ImageCompressor; no CompressAI dependency or fallback."""

from pathlib import Path

import torch
from torch import nn

from .models import pad_rgb
from .vendor.imagecompression.model import ImageCompressor


UPSTREAM_COMMIT = '6a77c4633cd5bb6750c77ea9d213d38ffc88ee7a'
PROXY_KIND = 'author_imagecompressor'


class ImageCompressorProxy(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = ImageCompressor(192, 192)

    def native_forward(self, rgb):
        return self.network(pad_rgb(rgb, 64))

    def forward(self, rgb):
        h, w = rgb.shape[-2:]
        padded = pad_rgb(rgb, 64)
        decoded, _, _, _, bpp = self.network(padded)
        return decoded[..., :h, :w], bpp * (padded.shape[-2] * padded.shape[-1]) / (h * w)


def native_checkpoint(path, qp=None, codec_level=None):
    try:
        saved = torch.load(Path(path), map_location='cpu', weights_only=True, mmap=True)
    except RuntimeError as error:
        if 'mmap can only be used with files saved with' not in str(error):
            raise
        # Native author weights may predate the zip-based torch.save format.
        saved = torch.load(Path(path), map_location='cpu', weights_only=True)
    metadata = saved.get('metadata', {})
    if metadata.get('proxy_kind', PROXY_KIND) != PROXY_KIND:
        raise ValueError('A native ImageCompressor checkpoint is required')
    if metadata.get('target') == 'bpg':
        if qp is not None and metadata.get('qp') != qp:
            raise ValueError(f'Native BPG checkpoint QP {metadata.get("qp")} differs from {qp}')
        if codec_level is not None and metadata.get('codec_level') != codec_level:
            raise ValueError('Native proxy BPG level mismatch')
    weights = saved.get('model', saved.get('state_dict', saved))
    if not isinstance(weights, dict):
        raise ValueError('A native ImageCompressor state dictionary is required')
    raw = {key.removeprefix('module.').removeprefix('network.'): value for key, value in weights.items()}
    model = ImageCompressorProxy()
    try:
        model.network.load_state_dict(raw, strict=True)
    except (RuntimeError, TypeError) as error:
        raise ValueError('Checkpoint is not fully compatible with author ImageCompressor(192,192)') from error
    return model, metadata
