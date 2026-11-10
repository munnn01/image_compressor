import ast
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('native_proxy_builder', Path(__file__).resolve().parents[1] / 'kaggle/prepare_proxy.py')
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


class ProxyDeploymentTests(unittest.TestCase):
    def test_initial_calibration_precedes_native_bpg_training(self):
        commands = builder.proxy_commands(34, 8192, False)
        self.assertEqual([r[0] for r in commands], ['make-crops', 'calibrate-proxy', 'train-proxy', 'audit-proxy'])
        train = commands[2]
        self.assertIn('--calibration', train)
        self.assertNotIn('--resume', train)
        self.assertEqual(train[train.index('--steps') + 1], '100000')
        self.assertEqual(train[train.index('--decay') + 1], '80000')

    def test_resume_preserves_optimizer_and_does_not_reinitialize(self):
        commands = builder.proxy_commands(28, 8192, True)
        self.assertEqual([r[0] for r in commands], ['make-crops', 'train-proxy', 'audit-proxy'])
        self.assertIn('--resume', commands[1])
        self.assertNotIn('--calibration', commands[1])
        self.assertNotIn('--init', commands[1])

    def test_packages_all_pinned_native_rd_inputs_and_compiles(self):
        sources = {str(qp): f'owner/native-rd-qp{qp}/7' for qp in (28,31,34,37,41)}
        with tempfile.TemporaryDirectory() as folder:
            builder.build_proxy(Path(folder), 'owner', 34, sources)
            meta = json.loads((Path(folder) / 'kernel-metadata.json').read_text())
            self.assertEqual(set(meta['kernel_sources']), set(sources.values()))
            self.assertEqual(meta['id'].split('/')[1], meta['title'].lower().replace(' ', '-'))
            code = ''.join(json.loads((Path(folder) / 'notebook.ipynb').read_text())['cells'][0]['source'])
            compile(code, 'notebook', 'exec')
            runtime = ast.literal_eval(ast.parse(code).body[0].value)
            compile(runtime, 'runtime', 'exec')
            self.assertIn('USE_JCTVC=y', runtime)
            self.assertIn("metadata['stage'] == 'rd_pretrain'", runtime)
            self.assertIn('calibration.json', runtime)
