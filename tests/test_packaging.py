import base64
import io
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

spec = importlib.util.spec_from_file_location('native_kaggle_prepare', Path(__file__).resolve().parents[1] / 'kaggle/prepare.py')
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)
build = prepare.build


class PackagingTests(unittest.TestCase):
    def test_notebook_is_complete_native_only_and_compiles(self):
        with tempfile.TemporaryDirectory() as folder:
            destination = Path(folder)
            build(destination, 'owner/native-test', [['make-crops', '--data', '/kaggle/input/images', '--output', '{WORK}/crops.json']], [], [])
            notebook = json.loads((destination / 'notebook.ipynb').read_text())
            code = ''.join(notebook['cells'][0]['source'])
            compile(code, 'notebook', 'exec')
            namespace = {}
            # Embedded values are assignments before the bootstrap's imports.
            exec(code.split('import shutil', 1)[0], namespace)
            compile(namespace['EXERCISE'], 'exercise', 'exec')
            archive_text = namespace['EXERCISE'].splitlines()[0]
            embedded = {}
            exec(archive_text, embedded)
            with zipfile.ZipFile(io.BytesIO(base64.b64decode(embedded['SOURCE']))) as source:
                names = source.namelist()
                self.assertIn('npp_proxy/vendor/imagecompression/model.py', names)
                self.assertIn('npp_proxy/vendor/npp_adaptive/Scaling_net.py', names)
                self.assertIn('upstream/manifest.json', names)
                self.assertIn('docs/PROVENANCE.md', names)
                self.assertFalse(any('pool.json' in name or '.pt' == Path(name).suffix for name in names))
                self.assertNotIn('compressai', source.read('requirements.txt').decode().lower())
            meta = json.loads((destination / 'kernel-metadata.json').read_text())
            self.assertEqual(meta['id'], 'owner/native-test')
            self.assertFalse(meta['is_private'])
            self.assertTrue(meta['enable_gpu'])

    def test_rejects_empty_commands_and_non_native_pipeline(self):
        with tempfile.TemporaryDirectory() as folder:
            for commands in ([], [[]], [['python', '-c', 'print(1)']]):
                with self.assertRaises(ValueError):
                    build(Path(folder), 'owner/native-test', commands, [], [])
