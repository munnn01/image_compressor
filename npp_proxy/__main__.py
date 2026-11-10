"""Command-line entry points for training, evaluation and single-image coding."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import torch

from .codec import BPG, read_rgb, save_rgb
from .metrics import bd_rate
from .models import NPP
from .native import ImageCompressorProxy, native_checkpoint as checkpoint_proxy, PROXY_KIND, UPSTREAM_COMMIT
from .proxy_diagnostics import calibrated_initialization, proxy_diagnostics, file_sha256
from .crop_manifest import make_manifest, proxy_dataset, ManifestCrops
from .recipe import require_completed_native
from .tasks import CLASSIFIERS, DETECTORS, FlickrPatches, Task
from .training import PROXY_LAMBDA, TASK_LAMBDA, algorithm1, fit_proxy_loss, fixed_sample_offset, load_state, optimize


def codec_for(args):
    return BPG(args.bpgenc, args.bpgdec, args.codec_level)


def load_npp(path, device):
    state = load_state(path)
    model = NPP(adaptive=any('Scaling_Net' in name for name in state)).to(device)
    if not all(name.startswith('network.') for name in state):
        state = {'network.' + name: value for name, value in state.items()}
    model.load_state_dict(state)
    return model.eval()


def train_proxy(args):
    if args.resume and (args.init or args.calibration):
        raise ValueError('Use --resume alone to continue the optimizer, or --init/--calibration for a new phase')
    initialization = args.init
    if args.calibration_root and not args.calibration:
        raise ValueError('--calibration-root requires --calibration')
    if args.calibration:
        if args.target != 'bpg':
            raise ValueError('Calibration selects native RD-pretrained weights for BPG fine-tuning')
        initialization = calibrated_initialization(args.calibration, args)
    if args.target == 'bpg' and not (initialization or args.resume):
        raise ValueError('Native BPG fine-tuning requires an ImageCompressor RD-pretrained --init or native --calibration, or --resume')
    if args.target == 'source' and args.steps is None:
        raise ValueError('Native RD pretraining requires an explicit --steps budget')
    if args.resume or initialization:
        proxy, saved_metadata = checkpoint_proxy(args.resume or initialization, args.qp)
        if initialization and args.target == 'bpg' and saved_metadata.get('target') == 'bpg':
            raise ValueError('Use --resume to continue BPG fine-tuning; --init must be native RD-pretrained weights')
    else:
        proxy, saved_metadata = ImageCompressorProxy(), {}
    codec = codec_for(args) if args.target == 'bpg' else None
    data = proxy_dataset(args.data, args.crop_size, root_override=args.image_root)
    if not len(data):
        raise ValueError(f'No proxy training images found in {args.data}')
    proxy = proxy.to(args.device).train()

    def update(batch, _rng):
        rgb = torch.stack(batch).to(args.device)
        if args.target == 'source':
            _, distortion, _, _, bpp = proxy.native_forward(rgb)
            return bpp + PROXY_LAMBDA[args.qp] * distortion, {'proxy_bpp': bpp.item(), 'source_mse': distortion.item()}
        return fit_proxy_loss(rgb, args.qp, proxy, codec)

    metadata = {'qp': args.qp, 'target': args.target, 'crop_size': args.crop_size, 'codec_level': args.codec_level}
    metadata.update(proxy_kind=PROXY_KIND, upstream_commit=UPSTREAM_COMMIT, channels=[192, 192])
    if isinstance(data, ManifestCrops):
        metadata['crop_manifest_sha256'] = data.sha256
    if args.resume:
        metadata.update({key: saved_metadata[key] for key in ('initial_checkpoint_sha256', 'initial_checkpoint_provenance') if key in saved_metadata})
    elif initialization:
        metadata.update(initial_checkpoint_sha256=file_sha256(initialization),
                        initial_checkpoint_provenance='native RD checkpoint metadata' if saved_metadata else 'raw native state; pretraining provenance supplied externally')
    optimize(args, proxy, 'rd_pretrain' if args.target == 'source' else 'proxy', data, update, metadata)


def train_npp(args):
    if args.codec_level != 8:
        raise ValueError('This reference pipeline requires BPG level 8')
    adaptive = args.stage == 'adaptive'
    if not adaptive and args.qp != 34:
        raise ValueError('The fixed reference stage requires QP34')
    for qp in TASK_LAMBDA:
        require_completed_native(args.proxy_dir / f'qp{qp}.pt', 'proxy', qp)
    checkpoint = None
    if args.fixed:
        checkpoint = require_completed_native(args.fixed, 'fixed')
        backbone = 'resnet50' if args.task == 'classification' else 'fcos'
        registry = CLASSIFIERS if args.task == 'classification' else DETECTORS
        weights = registry[backbone][1]
        weight_id = (str(args.task_weights.resolve()) if args.task_weights else
                     f'{weights.__class__.__name__}.{weights.name}')
        expected = {'task': args.task, 'backbone': backbone, 'task_weights': weight_id}
        if any(checkpoint['metadata'].get(key) != value for key, value in expected.items()):
            raise ValueError('Fixed checkpoint must match the adaptive task, backbone and task weights')
    if adaptive and not (args.fixed or args.resume):
        raise ValueError('Adaptive training requires --fixed or --resume')
    if not adaptive and args.fixed:
        raise ValueError('--fixed initializes the adaptive stage')
    if args.identity_init and adaptive:
        raise ValueError('--identity-init applies to the fixed stage')
    npp = NPP(adaptive).to(args.device).train()
    if args.identity_init:
        npp.initialize_identity()
    sample_offset = 0
    if args.fixed:
        npp.load_fixed(checkpoint.get('model', checkpoint.get('state_dict', checkpoint)))
        sample_offset = fixed_sample_offset(checkpoint, args.data, args.seed, args.task, args.annotations)
    codec = None if args.proxy_only else codec_for(args)
    task = Task(args.task, None, args.device, args.task_weights)
    data = task.dataset(args.data, args.annotations, train=True)
    if not len(data):
        raise ValueError(f'No task training images found in {args.data}')
    qps = tuple(TASK_LAMBDA) if adaptive else (args.qp,)
    proxies = {}
    for qp in qps:
        proxy_path = args.proxy_dir / f'qp{qp}.pt'
        model, _ = checkpoint_proxy(proxy_path, qp, codec.level if codec else None)
        # Training mode adds quantization noise and preserves input gradients. Weights stay frozen.
        proxies[qp] = model.to(args.device).train().requires_grad_(False)

    def update(batch, rng):
        qp = rng.choice(qps)
        if args.task == 'classification':
            images, labels = zip(*batch, strict=True)
            return algorithm1(torch.stack(images).to(args.device), qp, npp, proxies[qp], codec,
                              lambda decoded: task.loss(decoded, labels))
        samples = [algorithm1(image.unsqueeze(0).to(args.device), qp, npp, proxies[qp], codec,
                              lambda decoded: task.loss(decoded, target)) for image, target, _ in batch]
        metrics = {key: sum(values[key] for _, values in samples) / len(samples) for key in samples[0][1]}
        return torch.stack([loss for loss, _ in samples]).mean(), metrics

    metadata = {'task': args.task, 'backbone': task.backbone, 'task_weights': task.weight_id,
              'proxy_dir': str(args.proxy_dir.resolve()), 'qps': list(qps), 'proxy_only': args.proxy_only,
              'codec_level': args.codec_level,
              'annotations': str(args.annotations.resolve()) if args.annotations else None}
    metadata.update(npp_kind='author_preprocessor', qa_conditioning='QP/51')
    if args.identity_init:
        metadata['initialization'] = 'rgb_identity'
    optimize(args, npp, args.stage, data, update, metadata, sample_offset)


@torch.no_grad()
def evaluate(args):
    if len(set(args.qps)) != len(args.qps):
        raise ValueError('--qps requires distinct values')
    codec = codec_for(args)
    npp = load_npp(args.npp, args.device)
    task = Task(args.task, args.backbone, args.device, args.task_weights)
    data = task.dataset(args.data, args.annotations)
    count = min(len(data), args.limit) if args.limit is not None else len(data)
    if not count:
        raise ValueError('Evaluation requires at least one image')
    rows = []
    for qp in args.qps:
        rates, bit_totals, results = defaultdict(list), defaultdict(int), defaultdict(list)
        total_pixels = 0
        total_pre_mse = 0.0
        ids = []
        for index in range(count):
            example = data[index]
            source = example[0].to(args.device)
            filtered = npp(source.unsqueeze(0), qp)[0]
            total_pre_mse += (filtered - source).square().mean().item()
            pixel_count = source.shape[-2] * source.shape[-1]
            total_pixels += pixel_count
            if args.task == 'detection':
                ids.append(example[2])
            for variant, rgb in (('bpg', source), ('npp+bpg', filtered)):
                decoded, bits = codec.roundtrip(rgb, qp)
                rates[variant].append(bits / pixel_count)
                bit_totals[variant] += bits
                prediction = task.predict(decoded, example)
                if args.task == 'classification':
                    results[variant].append(prediction)
                else:
                    results[variant].extend(prediction)
        for variant, values in rates.items():
            if args.task == 'detection':
                metric = data.evaluate(results[variant], ids)
            else:
                metric = sum(results[variant]) / count * 100
            row = {'variant': variant, 'qp': qp, 'images': count, 'mean_bpp': sum(values) / count,
                   'pooled_bpp': bit_totals[variant] / total_pixels, 'metric_percent': metric,
                   'mean_pre_mse': total_pre_mse / count if variant == 'npp+bpg' else 0.0}
            rows.append(row)
            print(row, flush=True)
    curves = {name: [(r['mean_bpp'], r['metric_percent']) for r in rows if r['variant'] == name]
              for name in ('bpg', 'npp+bpg')}
    report = {'task': args.task, 'backbone': task.backbone, 'task_weights': task.weight_id,
              'metric': 'COCO AP@[.50:.95]' if args.task == 'detection' else 'ImageNet top-1',
              'checkpoint': str(args.npp.resolve()), 'bpg_encoder': codec.encoder, 'bpg_decoder': codec.decoder,
              'codec_level': args.codec_level, 'bd_rate': bd_rate(curves['bpg'], curves['npp+bpg']), 'rows': rows}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    with (args.output / 'rate_task.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(report['bd_rate']), flush=True)


@torch.no_grad()
def encode_image(args):
    codec = codec_for(args)
    rgb = read_rgb(args.input).to(args.device)
    if args.npp:
        rgb = load_npp(args.npp, args.device)(rgb.unsqueeze(0), args.qp)[0]
    payload = codec.encode(rgb, args.qp)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(payload)
    if args.decoded:
        args.decoded.parent.mkdir(parents=True, exist_ok=True)
        save_rgb(codec.decode(payload), args.decoded)
    print(json.dumps({'qp': args.qp, 'bytes': len(payload), 'bpp': 8 * len(payload) / (rgb.shape[-2] * rgb.shape[-1])}))


def positive(text):
    number = int(text)
    if number < 1:
        raise argparse.ArgumentTypeError('must be positive')
    return number


def parser():
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest='command', required=True)

    def command(name, action):
        sub = commands.add_parser(name)
        sub.set_defaults(action=action)
        sub.add_argument('--device', default='auto')
        sub.add_argument('--threads', type=positive, default=4)
        sub.add_argument('--bpgenc', default='bpgenc')
        sub.add_argument('--bpgdec', default='bpgdec')
        sub.add_argument('--codec-level', type=int, choices=range(1, 10), default=8)
        sub.add_argument('--output', required=True, type=Path)
        return sub

    def training(sub):
        sub.add_argument('--data', required=True, type=Path)
        sub.add_argument('--steps', type=positive)
        sub.add_argument('--run-updates', type=positive, help='Updates in this session, keeping --steps as the full target')
        sub.add_argument('--decay', type=int)
        sub.add_argument('--batch-size', type=positive, default=1)
        sub.add_argument('--seed', type=int, default=2024)
        sub.add_argument('--resume', type=Path)
        sub.set_defaults(lr_schedule='step', warmup_steps=None, cosine_start=None, allow_schedule_change=False)
        sub.add_argument('--log-every', type=positive, default=100)
        sub.add_argument('--save-every', type=positive, default=1000)

    def task_flags(sub):
        sub.add_argument('--task', required=True, choices=['classification', 'detection'])
        sub.add_argument('--annotations', type=Path)
        sub.add_argument('--task-weights', type=Path)

    for name, target in (('train-proxy', 'bpg'), ('pretrain-proxy', 'source')):
        proxy = command(name, train_proxy)
        training(proxy)
        proxy.set_defaults(target=target, batch_size=4)
        proxy.add_argument('--qp', required=True, type=int, choices=PROXY_LAMBDA)
        proxy.add_argument('--crop-size', type=positive, default=256)
        proxy.add_argument('--image-root', type=Path, help='Relocate a crop manifest image root')
        initial = proxy.add_mutually_exclusive_group()
        initial.add_argument('--init', type=Path)
        initial.add_argument('--calibration', type=Path)
        proxy.add_argument('--calibration-root', type=Path, help='Relocated native pretrained candidate directory')

    for name, calibrating in (('calibrate-proxy', True), ('audit-proxy', False)):
        diagnostics = command(name, lambda args, calibrating=calibrating: proxy_diagnostics(
            args, codec_for(args), calibrating))
        diagnostics.add_argument('--data', type=Path, required=True)
        diagnostics.add_argument('--seed', type=int, default=2024)
        diagnostics.add_argument('--samples', type=positive, default=64)
        diagnostics.add_argument('--crop-size', type=positive, default=256)
        diagnostics.add_argument('--qps', nargs='+', type=int, choices=PROXY_LAMBDA, default=list(TASK_LAMBDA))
        diagnostics.add_argument('--image-root', type=Path)
        if calibrating:
            diagnostics.add_argument('--candidate-dir', type=Path, required=True)
        else:
            diagnostics.add_argument('--proxy-dir', type=Path, required=True)

    npp = command('train-npp', train_npp)
    training(npp)
    task_flags(npp)
    npp.add_argument('--stage', required=True, choices=['fixed', 'adaptive'])
    npp.add_argument('--qp', type=int, choices=TASK_LAMBDA, default=34)
    npp.add_argument('--proxy-dir', required=True, type=Path)
    npp.add_argument('--fixed', type=Path)
    npp.set_defaults(proxy_only=False, identity_init=False, batch_size=2)

    crops = command('make-crops', lambda args: make_manifest(args.data, args.output, args.count, args.crop_size, args.seed, args.holdout))
    crops.add_argument('--data', type=Path, required=True)
    crops.add_argument('--count', type=positive, default=800000)
    crops.add_argument('--crop-size', type=positive, default=256)
    crops.add_argument('--seed', type=int, default=2024)
    crops.add_argument('--holdout', type=positive, default=64)

    evaluation = command('evaluate', evaluate)
    task_flags(evaluation)
    evaluation.add_argument('--data', required=True, type=Path)
    evaluation.add_argument('--npp', required=True, type=Path)
    evaluation.add_argument('--backbone', choices=[*CLASSIFIERS, *DETECTORS])
    evaluation.add_argument('--qps', nargs='+', type=int, choices=range(52), default=list(TASK_LAMBDA))
    evaluation.add_argument('--limit', type=positive)

    single = command('encode', encode_image)
    single.add_argument('--input', required=True, type=Path)
    single.add_argument('--npp', type=Path)
    single.add_argument('--qp', required=True, type=int, choices=range(52))
    single.add_argument('--decoded', type=Path)
    return root


def main():
    root = parser()
    args = root.parse_args()
    torch.set_num_threads(args.threads)
    if hasattr(args, 'seed'):
        torch.manual_seed(args.seed)
    if args.device == 'auto':
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    try:
        args.action(args)
    except (ValueError, FileNotFoundError) as error:
        root.error(str(error))


if __name__ == '__main__':
    main()
