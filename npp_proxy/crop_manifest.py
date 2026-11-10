"""A fixed virtual set of 800k crops, with source-image-disjoint holdout."""

import hashlib
import json
from pathlib import Path
import random

import torch
from torch.utils.data import Dataset
from torch.nn import functional as F

from .codec import read_rgb


def make_manifest(data, output, count=800000, crop_size=256, seed=2024, holdout=64):
    data, output = Path(data).resolve(), Path(output)
    paths = sorted(p for p in data.rglob('*') if p.is_file() and p.suffix.lower() in {'.jpg', '.jpeg', '.png'})
    if count < 1 or crop_size < 1 or not 0 < holdout < len(paths):
        raise ValueError('Positive crop count/size and a nonempty, separate holdout are required')
    chosen = set(random.Random(seed).sample(paths, holdout))
    manifest = {'schema': 1, 'kind': 'fixed-virtual-crops', 'root': str(data), 'count': count,
                'crop_size': crop_size, 'seed': seed,
                'train': [p.relative_to(data).as_posix() for p in paths if p not in chosen],
                'holdout': [p.relative_to(data).as_posix() for p in paths if p in chosen]}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return manifest


class ManifestCrops(Dataset):
    def __init__(self, path, split='train', root_override=None):
        self.path = Path(path).resolve()
        self.manifest = json.loads(self.path.read_text(encoding='utf-8'))
        if self.manifest.get('kind') != 'fixed-virtual-crops' or split not in ('train', 'holdout'):
            raise ValueError('A fixed crop manifest and train/holdout split are required')
        if set(self.manifest['train']) & set(self.manifest['holdout']):
            raise ValueError('Training and holdout source images overlap')
        root = Path(root_override or self.manifest['root']).resolve()
        relatives = self.manifest[split]
        self.paths = [(root / relative).resolve() for relative in relatives]
        if any(not p.is_relative_to(root) or not p.is_file() for p in self.paths):
            raise ValueError('Manifest source images missing or outside the image root')
        if not self.paths:
            raise ValueError('Empty manifest split')
        self.split = split
        self.size = self.manifest['crop_size']
        self.count = self.manifest['count'] if split == 'train' else len(self.paths)
        self.sha256 = hashlib.sha256(self.path.read_bytes()).hexdigest()

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        if not 0 <= index < self.count:
            raise IndexError(index)
        rng = random.Random(f'{self.manifest["seed"]}:{self.split}:{index}')
        image = read_rgb(self.paths[rng.randrange(len(self.paths))] if self.split == 'train' else self.paths[index])
        h, w = image.shape[-2:]
        if h < self.size or w < self.size:
            image = F.pad(image, (0, max(0, self.size - w), 0, max(0, self.size - h)), mode='replicate')
        h, w = image.shape[-2:]
        top, left = rng.randrange(h - self.size + 1), rng.randrange(w - self.size + 1)
        return image[:, top:top + self.size, left:left + self.size]


def proxy_dataset(data, crop_size, split='train', root_override=None):
    from .tasks import FlickrPatches
    if Path(data).is_file():
        result = ManifestCrops(data, split, root_override)
        if result.size != crop_size:
            raise ValueError('Manifest crop size differs from the requested crop size')
        return result
    return FlickrPatches(data, crop_size)
