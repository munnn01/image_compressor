import ast
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


PATH = Path(__file__).resolve().parents[1] / 'kaggle/prepare_rd.py'
spec = importlib.util.spec_from_file_location('prepare_native_rd', PATH)
rd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rd)


class RDDeploymentTests(unittest.TestCase):
    def test_resume_uses_full_budget_and_skips_completed_qp(self):
        starts = {28: 100000, 31: 4096, 34: 0, 37: 8192, 41: 0}
        commands = rd.stage_commands(starts, 4096)
        self.assertEqual(commands[0][0], 'make-crops')
        runs = {int(row[row.index('--qp') + 1]): row for row in commands[1:]}
        self.assertNotIn(28, runs)
        for qp, row in runs.items():
            self.assertEqual(row[0], 'pretrain-proxy')
            self.assertEqual(row[row.index('--steps') + 1], '100000')
            self.assertEqual(row[row.index('--decay') + 1], '80000')
            self.assertEqual(row[row.index('--run-updates') + 1], '4096')
            self.assertEqual('--resume' in row, starts[qp] > 0)
            self.assertNotIn('--init', row)

    def test_pinned_native_notebook_is_public_and_compiles(self):
        with tempfile.TemporaryDirectory() as folder:
            rd.build_rd(Path(folder), 'nguoingoaihanhlang', 4096, previous='nguoingoaihanhlang/native-imagecompressor-rd-bank/1')
            notebook = json.loads((Path(folder) / 'notebook.ipynb').read_text())
            code = ''.join(notebook['cells'][0]['source'])
            compile(code, 'rd-notebook', 'exec')
            exercise = ast.literal_eval(ast.parse(code).body[0].value)
            compile(exercise, 'rd-runtime', 'exec')
            self.assertIn('ImageCompressorProxy', exercise)
            self.assertIn('gpu_smoke.json', exercise)
            self.assertIn('progress.json', exercise)
            self.assertNotIn("run('make'", exercise)
            metadata = json.loads((Path(folder) / 'kernel-metadata.json').read_text())
            self.assertFalse(metadata['is_private'])
            self.assertTrue(metadata['enable_gpu'])
            self.assertEqual(metadata['id'].split('/')[1], metadata['title'].lower().replace(' ', '-'))
            self.assertEqual(metadata['kernel_sources'], ['nguoingoaihanhlang/native-imagecompressor-rd-bank/1'])
            self.assertEqual(metadata['dataset_sources'], ['eeshawn/flickr30k/1'])

    def test_rejects_bad_previous_source_and_excess_completed_steps(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(ValueError):
                rd.build_rd(Path(folder), 'owner', 4096, previous='owner/unversioned')
        with self.assertRaises(ValueError):
            rd.stage_commands({28: 100001, 31: 0, 34: 0, 37: 0, 41: 0}, 4096)

    def test_single_qp_shard_keeps_its_own_checkpoint_and_source(self):
        commands = rd.stage_commands({31: 16384}, 16384)
        self.assertEqual(len(commands), 2)
        row = commands[1]
        self.assertEqual(row[row.index('--qp') + 1], '31')
        self.assertEqual(row[row.index('--resume') + 1], '{WORK}/rd/qp31.pt')
        with tempfile.TemporaryDirectory() as folder:
            rd.build_rd(Path(folder), 'thuha1205', 16384, qps=(31,))
            meta = json.loads((Path(folder) / 'kernel-metadata.json').read_text())
            self.assertEqual(meta['id'], 'thuha1205/imagecompressor-native-rd-qp31')
            self.assertEqual(meta['title'], 'ImageCompressor Native RD QP31')
