"""Real native-model/BPG integration smoke; synthetic data, never benchmark results."""

import argparse
import gc
import json
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image
import torch

from npp_proxy.__main__ import parser
from npp_proxy.codec import BPG
from npp_proxy.models import NPP
from npp_proxy.native import native_checkpoint
from npp_proxy.training import algorithm1


def main():
    options = argparse.ArgumentParser(description=__doc__)
    options.add_argument('--bpgenc', required=True)
    options.add_argument('--bpgdec', required=True)
    options.add_argument('--output', type=Path, required=True)
    args = options.parse_args()
    torch.set_num_threads(2)
    codec = BPG(args.bpgenc, args.bpgdec, 8)

    def cli(*argv):
        config = parser().parse_args([str(x) for x in argv])
        config.device = 'cpu'
        torch.manual_seed(config.seed)
        config.action(config)
        gc.collect()

    with tempfile.TemporaryDirectory(prefix='native-smoke-') as temporary:
        root = Path(temporary)
        images = root / 'images'
        images.mkdir()
        rng = np.random.default_rng(71)
        for index in range(3):
            Image.fromarray(rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)).save(images / f'{index}.png')
        manifest = root / 'crops.json'
        cli('make-crops', '--data', images, '--count', 4, '--crop-size', 64, '--holdout', 1, '--output', manifest)
        common = ['--data', manifest, '--crop-size', 64, '--qp', 34, '--batch-size', 1,
                  '--steps', 3, '--decay', 2, '--log-every', 1, '--save-every', 1]
        rd = root / 'candidates' / 'rd.pt'
        cli('pretrain-proxy', *common, '--output', rd, '--run-updates', 2)
        first = torch.load(rd, weights_only=True, mmap=True)
        assert first['step'] == 2 and first['optimizer']['param_groups'][0]['lr'] == 1e-4
        before = next(iter(first['model'].values())).clone()
        del first
        cli('pretrain-proxy', *common, '--output', rd, '--resume', rd)
        resumed = torch.load(rd, weights_only=True, mmap=True)
        assert resumed['step'] == 3 and resumed['optimizer']['param_groups'][0]['lr'] == 1e-5
        assert all(int(s['step']) == 3 for s in resumed['optimizer']['state'].values())
        assert not torch.equal(before, next(iter(resumed['model'].values())))
        del resumed
        calibration = root / 'calibration.json'
        codec_args = ['--bpgenc', codec.encoder, '--bpgdec', codec.decoder]
        cli('calibrate-proxy', '--data', manifest, '--crop-size', 64, '--samples', 1,
            '--qps', 34, '--candidate-dir', rd.parent, '--output', calibration, *codec_args)
        proxy_path = root / 'bank' / 'qp34.pt'
        cli('train-proxy', *common, '--calibration', calibration, '--output', proxy_path,
            '--run-updates', 1, *codec_args)
        proxy, metadata = native_checkpoint(proxy_path, 34, 8)
        assert metadata['initial_checkpoint_sha256']
        proxy.train().requires_grad_(False)
        torch.manual_seed(71)
        fixed, adaptive = NPP(), NPP(adaptive=True)
        gates = {k: v.clone() for k, v in adaptive.state_dict().items() if 'Scaling_Net' in k}
        adaptive.load_fixed(fixed.state_dict())
        for key, value in fixed.state_dict().items():
            torch.testing.assert_close(adaptive.state_dict()[key], value, atol=0, rtol=0)
        for key, value in gates.items():
            torch.testing.assert_close(adaptive.state_dict()[key], value, atol=0, rtol=0)
        task = torch.nn.Conv2d(3, 2, 3, padding=1).requires_grad_(False)
        rgb = torch.rand(1, 3, 64, 64)
        stage_metrics = {}
        for stage, model in [('fixed', fixed), ('adaptive', adaptive)]:
            observed = []

            def task_loss(decoded):
                observed.append(decoded.detach().clone())
                return task(decoded).square().mean()

            filtered = model(rgb, 34)
            real, bits = codec.roundtrip(filtered[0], 34)
            loss, metrics = algorithm1(rgb, 34, model, proxy, codec, task_loss)
            torch.testing.assert_close(observed[0][0], real, atol=0, rtol=0)
            assert metrics['bpp'] == bits / (64 * 64)
            loss.backward()
            assert any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0 for p in model.parameters())
            assert all(p.grad is None for p in proxy.parameters())
            assert all(p.grad is None for p in task.parameters())
            torch.optim.Adam(model.parameters(), lr=1e-4).step()
            stage_metrics[stage] = metrics
        result = {'passed': True, 'synthetic_only': True, 'device': 'cpu', 'rd_updates': 3,
                  'task_model': 'synthetic frozen Conv2d, not pretrained ResNet/FCOS',
                  'torch': str(torch.__version__),
                  'bpg_proxy_updates': 1, 'npp_updates': {'fixed': 1, 'adaptive': 1},
                  'resume_optimizer_and_decay': True, 'real_bpg_forward_frozen_proxy_gradient': True,
                  'stage_metrics': stage_metrics}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
        print('Native ImageCompressor / BPG / fixed / adaptive smoke passed.')


if __name__ == '__main__':
    main()
