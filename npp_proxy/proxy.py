"""Native-only checkpoint API used by training and evaluation."""

from .native import ImageCompressorProxy, native_checkpoint

checkpoint_proxy = native_checkpoint
