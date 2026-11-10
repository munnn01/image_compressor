import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import torch
from PIL import Image
import numpy as np

from npp_proxy.crop_manifest import make_manifest, ManifestCrops
from npp_proxy.recipe import require_completed_native


class RecipeTests(unittest.TestCase):
    def test_finetune_requires_native_pretraining_before_codec_access(self):
        from npp_proxy.__main__ import parser
        args = parser().parse_args(['train-proxy', '--data', 'missing-data', '--qp', '34', '--output', 'model.pt'])
        with self.assertRaisesRegex(ValueError, 'native|pretrain'):
            args.action(args)

    def test_manifest_crops_are_fixed_and_holdout_is_disjoint(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            images = root / 'images'
            images.mkdir()
            for index in range(4):
                pixels = np.arange(80 * 91 * 3, dtype=np.uint8).reshape(80, 91, 3)
                Image.fromarray(np.roll(pixels, index * 11, axis=1)).save(images / f'{index}.png')
            make_manifest(images, root / 'crops.json', count=8, crop_size=64, seed=71, holdout=1)
            train = ManifestCrops(root / 'crops.json')
            heldout = ManifestCrops(root / 'crops.json', split='holdout')
            self.assertEqual(len(train), 8)
            self.assertEqual(len(heldout), 1)
            self.assertTrue(set(train.paths).isdisjoint(heldout.paths))
            a = train[3]
            torch.manual_seed(999)
            torch.testing.assert_close(a, train[3], atol=0, rtol=0)
            self.assertEqual(a.shape, (3, 64, 64))

    def test_npp_gate_rejects_short_proxy_and_rd_only_checkpoint(self):
        with TemporaryDirectory() as folder:
            path = Path(folder) / 'qp34.pt'
            state = {'model': {'dummy': torch.zeros(1)}, 'step': 99999,
                     'metadata': {'stage': 'proxy', 'target': 'bpg', 'qp': 34, 'codec_level': 8,
                                  'proxy_kind': 'author_imagecompressor'},
                     'plan': {'target_steps': 100000, 'decay_step': 80000}}
            torch.save(state, path)
            with self.assertRaises(ValueError):
                require_completed_native(path, 'proxy', qp=34)
            state['step'] = 100000
            state['metadata']['target'] = 'source'
            torch.save(state, path)
            with self.assertRaises(ValueError):
                require_completed_native(path, 'proxy', qp=34)
            state['metadata']['target'] = 'bpg'
            torch.save(state, path)
            require_completed_native(path, 'proxy', qp=34)

    def test_phase_gate_rejects_cosine_and_non_author_npp(self):
        with TemporaryDirectory() as folder:
            path = Path(folder) / 'fixed.pt'
            state = {'step': 30000, 'metadata': {'stage': 'fixed', 'npp_kind': 'author_preprocessor'},
                     'plan': {'target_steps': 30000, 'decay_step': 20000, 'lr_schedule': {'kind': 'cosine'}}}
            torch.save(state, path)
            with self.assertRaises(ValueError):
                require_completed_native(path, 'fixed')
            state['plan']['lr_schedule'] = {'kind': 'step'}
            state['metadata']['npp_kind'] = 'other'
            torch.save(state, path)
            with self.assertRaises(ValueError):
                require_completed_native(path, 'fixed')

    def test_fixed_gate_rejects_wrong_qp_codec_and_proxy_only(self):
        with TemporaryDirectory() as folder:
            path = Path(folder) / 'fixed.pt'
            valid = {'stage': 'fixed', 'npp_kind': 'author_preprocessor', 'qps': [34],
                     'codec_level': 8, 'proxy_only': False, 'task': 'classification'}
            state = {'step': 30000, 'metadata': valid,
                     'plan': {'target_steps': 30000, 'decay_step': 20000, 'lr_schedule': {'kind': 'step'}}}
            for key, value in [('qps', [28]), ('codec_level', 1), ('proxy_only', True)]:
                state['metadata'] = {**valid, key: value}
                torch.save(state, path)
                with self.subTest(key=key), self.assertRaises(ValueError):
                    require_completed_native(path, 'fixed')

    def test_rejects_non_qp34_fixed_training_before_loading_proxies(self):
        from npp_proxy.__main__ import parser
        args = parser().parse_args(['train-npp', '--stage', 'fixed', '--qp', '28', '--task', 'classification',
                                   '--data', 'missing', '--proxy-dir', 'missing', '--output', 'model.pt'])
        with self.assertRaisesRegex(ValueError, 'QP34'):
            args.action(args)

    def test_adaptive_rejects_fixed_checkpoint_for_other_task(self):
        from unittest.mock import patch
        from npp_proxy.__main__ import parser
        args = parser().parse_args(['train-npp', '--stage', 'adaptive', '--task', 'detection', '--fixed', 'fixed.pt',
                                   '--data', 'missing', '--proxy-dir', 'missing', '--output', 'model.pt'])
        valid = {'task': 'detection', 'backbone': 'fcos', 'task_weights': 'FCOS_ResNet50_FPN_Weights.COCO_V1'}
        for key, value in [('task', 'classification'), ('backbone', 'resnet50'), ('task_weights', 'other')]:
            checkpoint = {'metadata': {**valid, key: value}}
            with self.subTest(key=key), patch('npp_proxy.__main__.require_completed_native', return_value=checkpoint):
                with self.assertRaisesRegex(ValueError, 'task'):
                    args.action(args)



if __name__ == '__main__':
    unittest.main()
