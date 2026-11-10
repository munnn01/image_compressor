"""Build a public, self-contained native ImageCompressor Kaggle notebook. No submission."""

import argparse
import base64
import hashlib
import io
import json
from pathlib import Path
import re
import zipfile


PROJECT = Path(__file__).resolve().parents[1]
COMMANDS = {'make-crops', 'pretrain-proxy', 'calibrate-proxy', 'train-proxy', 'audit-proxy', 'train-npp', 'evaluate', 'encode'}

EXERCISE = '''
import base64
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import urllib.request
import zipfile

def run(*argv, cwd=None):
    subprocess.run([str(arg) for arg in argv], cwd=cwd, check=True)

work = Path('/kaggle/working')
root = work / 'imagecompressor_code'
root.mkdir(exist_ok=True)
payload = base64.b64decode(SOURCE)
assert hashlib.sha256(payload).hexdigest() == SOURCE_SHA256
with zipfile.ZipFile(io.BytesIO(payload)) as source:
    source.extractall(root)
with urllib.request.urlopen('https://bellard.org/bpg/libbpg-0.9.8.tar.gz', timeout=60) as response:
    archive = response.read(5_000_001)
assert hashlib.sha256(archive).hexdigest() == 'c0788e23bdf1a7d36cb4424ccb2fae4c7789ac94949563c4ad0e2569d3bf0095'
with tarfile.open(fileobj=io.BytesIO(archive), mode='r:gz') as package:
    package.extractall(root, filter='data')
bpg = root / 'libbpg-0.9.8'
run('make', '-C', bpg, '-j2', 'USE_X265=', 'USE_JCTVC=y', 'bpgenc', 'bpgdec')
replacements = {'{WORK}': str(work), '{BPGENC}': str(bpg / 'bpgenc'), '{BPGDEC}': str(bpg / 'bpgdec')}
(work / 'native_run.json').write_text(json.dumps({'source_sha256': SOURCE_SHA256, 'commands': COMMAND_LIST}, indent=2))
for command in COMMAND_LIST:
    argv = []
    for value in command:
        for key, replacement in replacements.items():
            value = value.replace(key, replacement)
        argv.append(value)
    run(sys.executable, '-u', '-m', 'npp_proxy', *argv,
        '--bpgenc', bpg / 'bpgenc', '--bpgdec', bpg / 'bpgdec', cwd=root)
print('All requested native ImageCompressor commands completed.')
'''

BOOTSTRAP = '''
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

def run(*args):
    subprocess.run([str(arg) for arg in args], check=True)

uv = shutil.which('uv')
if uv is None:
    run(sys.executable, '-m', 'pip', 'install', 'uv')
    uv = shutil.which('uv')
assert uv, 'uv executable missing'
with tempfile.TemporaryDirectory(prefix='native-python312-') as folder:
    root = Path(folder)
    venv = root / 'venv'
    run(uv, 'venv', '--python', '3.12', venv)
    python = venv / 'bin' / 'python'
    run(uv, 'pip', 'install', '--python', python, '--index-url', 'https://download.pytorch.org/whl/cu121',
        'torch==2.5.1', 'torchvision==0.20.1')
    run(uv, 'pip', 'install', '--python', python, 'numpy>=1.26,<2', 'Pillow>=10,<12',
        'scipy>=1.11,<2', 'pycocotools>=2.0.7,<3')
    script = root / 'exercise.py'
    script.write_text(EXERCISE)
    run(python, '-u', script)
'''


def build(destination, kernel_id, commands, datasets, kernels):
    if not re.fullmatch(r'[A-Za-z0-9_-]+/[A-Za-z0-9_-]+', kernel_id):
        raise ValueError('Use owner/notebook-slug for --id')
    if not isinstance(commands, list) or not commands or any(
            not isinstance(row, list) or not row or any(not isinstance(x, str) for x in row)
            or row[0] not in COMMANDS for row in commands):
        raise ValueError('Commands must be nonempty arrays of native npp_proxy CLI arguments')
    buffer = io.BytesIO()
    paths = [*PROJECT.glob('npp_proxy/**/*.py'), *PROJECT.glob('upstream/**/*'),
             PROJECT / 'requirements.txt', PROJECT / 'LICENSE', PROJECT / 'README.md',
             PROJECT / 'docs/PROVENANCE.md']
    with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as source:
        for path in sorted(set(paths)):
            if path.is_file() and '__pycache__' not in path.parts and path.suffix != '.pyc':
                source.write(path, path.relative_to(PROJECT).as_posix())
    payload = buffer.getvalue()
    exercise = (f'SOURCE = {base64.b64encode(payload).decode()!r}\n'
                f'SOURCE_SHA256 = {hashlib.sha256(payload).hexdigest()!r}\n'
                f'COMMAND_LIST = {commands!r}\n' + EXERCISE)
    code = f'EXERCISE = {exercise!r}\n' + BOOTSTRAP
    compile(code, 'notebook', 'exec')
    compile(exercise, 'exercise', 'exec')
    notebook = {'nbformat': 4, 'nbformat_minor': 4,
                'metadata': {'language_info': {'name': 'python'},
                             'kernelspec': {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'}},
                'cells': [{'cell_type': 'code', 'execution_count': None, 'metadata': {}, 'outputs': [],
                           'source': code.splitlines(keepends=True)}]}
    metadata = {'id': kernel_id, 'title': 'Native ImageCompressor NPP Reproduction',
                'code_file': 'notebook.ipynb', 'language': 'python', 'kernel_type': 'notebook',
                'is_private': False, 'enable_gpu': True, 'enable_internet': True,
                'dataset_sources': datasets, 'kernel_sources': kernels, 'competition_sources': []}
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / 'notebook.ipynb').write_text(json.dumps(notebook), encoding='utf-8')
    (destination / 'kernel-metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    (destination / 'source-sha256.txt').write_text(hashlib.sha256(payload).hexdigest() + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--id', required=True)
    parser.add_argument('--commands', required=True, type=Path, help='JSON list of CLI argv arrays')
    parser.add_argument('--dataset', action='append', default=[])
    parser.add_argument('--kernel', action='append', default=[])
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    build(args.output, args.id, json.loads(args.commands.read_text(encoding='utf-8')),
          args.dataset, args.kernel)


if __name__ == '__main__':
    main()
