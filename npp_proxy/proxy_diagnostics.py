"""Calibrate and audit native ImageCompressor checkpoints on a separate holdout."""

import hashlib
import json
import math
from pathlib import Path
from statistics import mean

import torch
from torch.nn import functional as F

from .crop_manifest import ManifestCrops, proxy_dataset
from .native import native_checkpoint, PROXY_KIND, UPSTREAM_COMMIT


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def mean_psnr(errors):
    return mean(-10 * math.log10(max(value, 1e-12)) for value in errors)


@torch.inference_mode()
def proxy_diagnostics(args, codec, calibrating):
    data = proxy_dataset(args.data, args.crop_size, split='holdout',
                         root_override=getattr(args, 'image_root', None))
    if not 0 < args.samples <= len(data):
        raise ValueError('Not enough separate holdout samples')
    indices = torch.randperm(len(data), generator=torch.Generator().manual_seed(args.seed))[:args.samples].tolist()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(args.seed)
        patches = [data[index] for index in indices]
    report = {'schema': 1, 'kind': 'native-calibration' if calibrating else 'native-audit',
              'proxy_kind': PROXY_KIND, 'upstream_commit': UPSTREAM_COMMIT,
              'data': str(args.data.resolve()), 'split': 'holdout', 'seed': args.seed,
              'crop_size': args.crop_size, 'codec_level': args.codec_level,
              'manifest_sha256': data.sha256 if isinstance(data, ManifestCrops) else None,
              'holdout_indices': indices, 'points': {}}
    candidates = {}
    if calibrating:
        root = args.candidate_dir.resolve()
        paths = sorted(p for p in root.rglob('*') if p.is_file() and p.suffix in {'.pt', '.pth', '.tar'})
        if not paths:
            raise ValueError('Provide native RD-pretrained ImageCompressor candidates')
        for path in paths:
            model, metadata = native_checkpoint(path)
            if metadata.get('target') == 'bpg':
                raise ValueError('Calibration candidates must be RD-pretrained weights, not BPG-finetuned outputs')
            model = model.to(args.device).eval()
            errors, rates = [], []
            for patch in patches:
                decoded, bpp = model(patch.unsqueeze(0).to(args.device))
                errors.append(F.mse_loss(decoded[0], patch.to(args.device)).item())
                rates.append(bpp.item())
            candidates[path.relative_to(root).as_posix()] = {
                'mean_psnr': mean_psnr(errors), 'mean_estimated_bpp': mean(rates),
                'sha256': file_sha256(path), 'metadata': metadata}
            del model
        report.update(candidates_root=str(root), candidates=candidates)
    for qp in args.qps:
        bpg_errors, bpg_rates, proxy_errors, proxy_rates, alignment = [], [], [], [], []
        if not calibrating:
            model, metadata = native_checkpoint(args.proxy_dir / f'qp{qp}.pt', qp, args.codec_level)
            model = model.to(args.device).eval()
        for patch in patches:
            decoded, bits = codec.roundtrip(patch, qp)
            bpg_errors.append(F.mse_loss(decoded, patch).item())
            bpg_rates.append(bits / (patch.shape[-2] * patch.shape[-1]))
            if not calibrating:
                reconstructed, bpp = model(patch.unsqueeze(0).to(args.device))
                proxy_errors.append(F.mse_loss(reconstructed[0], patch.to(args.device)).item())
                proxy_rates.append(bpp.item())
                alignment.append(F.mse_loss(reconstructed[0], decoded.to(args.device)).item())
        point = {'bpg_mean_psnr': mean_psnr(bpg_errors), 'bpg_mean_bpp': mean(bpg_rates)}
        if calibrating:
            chosen = min(candidates, key=lambda key: (abs(candidates[key]['mean_psnr'] - point['bpg_mean_psnr']), key))
            point.update(checkpoint=chosen, checkpoint_sha256=candidates[chosen]['sha256'])
        else:
            point.update(proxy_mean_psnr=mean_psnr(proxy_errors), proxy_mean_bpp=mean(proxy_rates),
                         proxy_to_bpg_rate_ratio=mean(proxy_rates) / mean(bpg_rates),
                         proxy_to_bpg_rgb_mse=mean(alignment))
            del model
        report['points'][str(qp)] = point
        print(f'QP {qp}: {point}', flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')


def calibrated_initialization(path, args):
    report = json.loads(Path(path).read_text(encoding='utf-8'))
    if (report.get('kind') != 'native-calibration' or report.get('proxy_kind') != PROXY_KIND
            or report.get('crop_size') != args.crop_size or report.get('codec_level') != args.codec_level):
        raise ValueError('Native calibration requires matching ImageCompressor, crop size and BPG configuration')
    train, holdout = args.data.resolve(), Path(report['data']).resolve()
    if train.is_file() and train == holdout:
        if file_sha256(train) != report.get('manifest_sha256') or report.get('split') != 'holdout':
            raise ValueError('Crop manifest or holdout split changed')
    elif train.is_relative_to(holdout) or holdout.is_relative_to(train):
        raise ValueError('Calibration holdout must be separate from proxy training images')
    point = report['points'].get(str(args.qp))
    if point is None:
        raise ValueError(f'No calibrated native checkpoint for QP {args.qp}')
    root = (args.calibration_root or Path(report['candidates_root'])).resolve()
    selected = (root / point['checkpoint']).resolve()
    if not selected.is_relative_to(root) or file_sha256(selected) != point['checkpoint_sha256']:
        raise ValueError('Native calibration checkpoint content/path changed')
    return selected
