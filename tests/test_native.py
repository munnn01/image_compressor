import contextlib
import io
import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory
from types import ModuleType
import sys
import unittest
from unittest.mock import patch

import torch

from npp_proxy.native import ImageCompressorProxy, native_checkpoint
from npp_proxy.vendor.imagecompression import models as native_models


ROOT = Path(__file__).resolve().parents[1]


class NativeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def reference(self):
        path = ROOT / 'upstream/NPP_Codec/ImageCompression/model.py'
        namespace = {'__name__': 'upstream_reference'}
        models_path = path.parent / 'models'
        spec = importlib.util.spec_from_file_location('author_reference_models', models_path / '__init__.py',
                                                     submodule_search_locations=[str(models_path)])
        original_models = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {'author_reference_models': original_models, 'models': original_models}):
            spec.loader.exec_module(original_models)
            exec(compile(path.read_text(encoding='utf-8'), str(path), 'exec'), namespace)
        with contextlib.redirect_stdout(io.StringIO()):
            return namespace['ImageCompressor'](192, 192)

    def test_matches_author_forward_and_input_gradient(self):
        for training in (False, True):
            with self.subTest(training=training), contextlib.redirect_stdout(io.StringIO()):
                torch.manual_seed(73)
                reference = self.reference().train(training).requires_grad_(False)
                actual = ImageCompressorProxy().train(training).requires_grad_(False)
                actual.network.load_state_dict(reference.state_dict(), strict=True)
                a = torch.rand(1, 3, 64, 64, requires_grad=True)
                b = a.detach().clone().requires_grad_(True)
                torch.manual_seed(56)
                expected = reference(a)
                torch.manual_seed(56)
                decoded, bpp = actual(b)
                torch.testing.assert_close(decoded, expected[0], atol=0, rtol=0)
                torch.testing.assert_close(bpp, expected[4], atol=0, rtol=0)
                grad_a, = torch.autograd.grad(expected[0].mean() + expected[4], a)
                grad_b, = torch.autograd.grad(decoded.mean() + bpp, b)
                torch.testing.assert_close(grad_b, grad_a, atol=0, rtol=0)

    def test_strict_raw_native_checkpoint_and_compressai_rejection(self):
        with TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            root = Path(folder)
            model = ImageCompressorProxy()
            torch.save(model.network.state_dict(), root / 'author.pt')
            restored, metadata = native_checkpoint(root / 'author.pt', 34)
            self.assertEqual(set(restored.network.state_dict()), set(model.network.state_dict()))
            broken = dict(model.network.state_dict())
            broken.pop(next(iter(broken)))
            torch.save(broken, root / 'partial.pt')
            with self.assertRaisesRegex(ValueError, 'native|ImageCompressor|compatible'):
                native_checkpoint(root / 'partial.pt', 34)
            torch.save({'network.g_a.0.weight': torch.rand(1)}, root / 'mbt.pt')
            with self.assertRaisesRegex(ValueError, 'native|ImageCompressor|compatible'):
                native_checkpoint(root / 'mbt.pt', 34)

    def test_odd_size_is_cropped_and_rate_counts_padded_bits(self):
        with contextlib.redirect_stdout(io.StringIO()):
            model = ImageCompressorProxy().eval()
            x = torch.rand(1, 3, 65, 67)
            decoded, bpp = model(x)
            self.assertEqual(decoded.shape, x.shape)
            _, _, _, _, padded_bpp = model.network(torch.nn.functional.pad(x, (0, 61, 0, 63), mode='replicate'))
            torch.testing.assert_close(bpp, padded_bpp * (128 * 128) / (65 * 67))

    def test_accepts_legacy_serialized_native_weights(self):
        with TemporaryDirectory() as folder:
            path = Path(folder) / 'native-legacy.pth'
            model = ImageCompressorProxy()
            torch.save(model.network.state_dict(), path, _use_new_zipfile_serialization=False)
            restored, _ = native_checkpoint(path)
            for name, value in model.network.state_dict().items():
                torch.testing.assert_close(restored.network.state_dict()[name], value, atol=0, rtol=0)


if __name__ == '__main__':
    unittest.main()
