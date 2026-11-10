"""Package five native RD pretraining models with version-pinned Kaggle resume."""

import argparse
import ast
import importlib.util
import inspect
import json
from pathlib import Path
import re


_spec = importlib.util.spec_from_file_location('native_notebook_builder', Path(__file__).with_name('prepare.py'))
base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(base)
SLUG = 'imagecompressor-native-rd-pretraining'


def stage_commands(starts, run_updates, batch_size=4):
    qps = tuple(sorted(starts))
    if (not qps or not set(qps).issubset({28, 31, 34, 37, 41})
            or any(type(value) is not int or not 0 <= value <= 100000 for value in starts.values())
            or not 1 <= run_updates <= 100000 or batch_size < 1):
        raise ValueError('Require valid per-QP progress and a positive session/batch budget')
    commands = [['make-crops', '--data', '{FLICKR}', '--output', '{WORK}/crops.json']]
    for qp in qps:
        if starts[qp] == 100000:
            continue
        destination = f'{{WORK}}/rd/qp{qp}.pt'
        row = ['pretrain-proxy', '--data', '{WORK}/crops.json', '--qp', str(qp), '--device', 'cuda:0',
               '--threads', '2', '--steps', '100000', '--decay', '80000', '--run-updates', str(run_updates),
               '--batch-size', str(batch_size), '--save-every', '256', '--log-every', '128', '--seed', '2024',
               '--output', destination]
        if starts[qp]:
            row += ['--resume', destination]
        commands.append(row)
    return commands


RD_SETUP = '''
import gc
import shutil
import torch
sys.path.insert(0, str(root))
from npp_proxy.native import ImageCompressorProxy, PROXY_KIND, UPSTREAM_COMMIT

inputs = Path('/kaggle/input')
flickr = next((path for path in (inputs / 'datasets/eeshawn/flickr30k', inputs / 'flickr30k') if path.is_dir()), None)
if flickr is None:
    raise FileNotFoundError('Expected version-pinned eeshawn/flickr30k dataset')
images = [p for p in flickr.rglob('*') if p.is_file() and p.suffix.lower() in {'.jpg', '.jpeg', '.png'}]
assert len(images) >= 30000, f'Flickr30k input incomplete: {len(images)} images'
replacements['{FLICKR}'] = str(flickr)
assert torch.cuda.is_available(), 'CUDA GPU is required for native RD pretraining'
torch.set_num_threads(2)
smoke = ImageCompressorProxy().cuda().train().requires_grad_(False)
rgb = torch.rand(1, 3, 64, 64, device='cuda', requires_grad=True)
_, distortion, _, _, rate = smoke.native_forward(rgb)
loss = 1024 * distortion + rate
loss.backward()
assert torch.isfinite(loss) and rgb.grad is not None and torch.isfinite(rgb.grad).all() and rgb.grad.abs().sum() > 0
(work / 'gpu_smoke.json').write_text(json.dumps({'passed': True, 'torch': str(torch.__version__),
    'gpu': torch.cuda.get_device_name(0), 'proxy_kind': PROXY_KIND, 'upstream_commit': UPSTREAM_COMMIT,
    'estimated_bpp': rate.item(), 'input_gradient_finite': True}, indent=2))
print('NATIVE_GPU_SMOKE_PASSED', torch.cuda.get_device_name(0), flush=True)
del smoke, rgb, loss, distortion, rate
gc.collect()
torch.cuda.empty_cache()

bank = work / 'rd'
bank.mkdir(exist_ok=True)
starts = {qp: 0 for qp in QPS}
if PREVIOUS:
    owner, slug, version = PREVIOUS.split('/')
    parents = (inputs / 'notebooks' / owner, inputs / owner, inputs)
    mounts = {parent / alias for parent in parents for alias in (slug, f'{slug}-versions-{version}')
              if (parent / alias).is_dir()}
    for qp in starts:
        found = {p.resolve() for mount in mounts for p in mount.rglob(f'qp{qp}.pt')}
        if len(found) != 1:
            raise ValueError(f'Expected one native QP{qp} checkpoint in pinned source {PREVIOUS}: {found}')
        source = found.pop()
        saved = torch.load(source, map_location='cpu', weights_only=True, mmap=True)
        metadata, plan = saved['metadata'], saved['plan']
        assert metadata['stage'] == 'rd_pretrain' and metadata['target'] == 'source'
        assert metadata['proxy_kind'] == PROXY_KIND and metadata['qp'] == qp
        assert metadata['upstream_commit'] == UPSTREAM_COMMIT and metadata['channels'] == [192, 192]
        assert metadata['batch_size'] == BATCH_SIZE and metadata['seed'] == 2024
        assert plan['target_steps'] == 100000 and plan['decay_step'] == 80000
        assert plan['lr_schedule'] == {'kind': 'step', 'initial_lr': 1e-4, 'decayed_lr': 1e-5, 'decay_step': 80000}
        starts[qp] = saved['step']
        del saved
        shutil.copy2(source, bank / f'qp{qp}.pt')

COMMAND_LIST = stage_commands(starts, RUN_UPDATES, BATCH_SIZE)
(work / 'native_run.json').write_text(json.dumps({'source_sha256': SOURCE_SHA256, 'previous': PREVIOUS,
    'commands': COMMAND_LIST, 'target_per_qp': 100000, 'decay_after': 80000,
    'stage': 'native_rd_pretraining', 'is_bpg_finetuning': False}, indent=2))

def write_progress(session_complete):
    models = {}
    for qp, start in starts.items():
        path = bank / f'qp{qp}.pt'
        step, lr = start, None
        if path.is_file():
            saved = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
            step = saved['step']
            lr = saved['optimizer']['param_groups'][0]['lr']
            del saved
        models[str(qp)] = {'start_step': start, 'step': step, 'lr': lr, 'complete': step == 100000}
    report = {'stage': 'rd_pretrain', 'proxy_kind': PROXY_KIND, 'upstream_commit': UPSTREAM_COMMIT,
              'source_sha256': SOURCE_SHA256, 'target_per_qp': 100000, 'decay_after': 80000,
              'session_updates_per_model': RUN_UPDATES, 'session_complete': session_complete,
              'models': models, 'complete': all(row['complete'] for row in models.values())}
    manifest = work / 'crops.json'
    if manifest.is_file():
        report['crop_manifest_sha256'] = hashlib.sha256(manifest.read_bytes()).hexdigest()
    (work / 'progress.json').write_text(json.dumps(report, indent=2))
    print('NATIVE_RD_PROGRESS', json.dumps(report), flush=True)

write_progress(False)
'''


def build_rd(destination, owner, run_updates=4096, previous=None, batch_size=4, qps=(28, 31, 34, 37, 41)):
    if previous and not re.fullmatch(r'[A-Za-z0-9_-]+/[A-Za-z0-9_-]+/[1-9][0-9]*', previous):
        raise ValueError('Previous output must be owner/slug/version')
    if len(set(qps)) != len(qps):
        raise ValueError('Each QP can occur only once')
    starts = {qp: 0 for qp in qps}
    commands = stage_commands(starts, run_updates, batch_size)
    destination = Path(destination)
    slug = f'imagecompressor-native-rd-qp{qps[0]}' if len(qps) == 1 else SLUG
    base.build(destination, f'{owner}/{slug}', commands, ['eeshawn/flickr30k/1'], [previous] if previous else [])
    path = destination / 'notebook.ipynb'
    notebook = json.loads(path.read_text(encoding='utf-8'))
    code = ''.join(notebook['cells'][0]['source'])
    exercise = ast.literal_eval(ast.parse(code).body[0].value)
    # RD pretraining needs no standard codec. The BPG fine-tuning notebook builds it later.
    begin = exercise.index("with urllib.request.urlopen('https://bellard.org/bpg/")
    end = exercise.index('replacements = ', begin)
    exercise = exercise[:begin] + "bpg = root / 'unused-bpg'\n" + exercise[end:]
    exercise = (f'PREVIOUS = {previous!r}\nRUN_UPDATES = {run_updates!r}\nBATCH_SIZE = {batch_size!r}\nQPS = {tuple(qps)!r}\n'
                + inspect.getsource(stage_commands) + '\n' + exercise)
    exercise = exercise.replace('for command in COMMAND_LIST:', RD_SETUP + '\nfor command in COMMAND_LIST:', 1)
    exercise = exercise.replace("'--bpgenc', bpg / 'bpgenc', '--bpgdec', bpg / 'bpgdec', cwd=root)",
                                "'--bpgenc', bpg / 'bpgenc', '--bpgdec', bpg / 'bpgdec', cwd=root)\n    write_progress(False)\nwrite_progress(True)")
    compile(exercise, 'native-rd-runtime', 'exec')
    code = f'EXERCISE = {exercise!r}\n' + base.BOOTSTRAP
    compile(code, 'native-rd-notebook', 'exec')
    notebook['cells'][0]['source'] = code.splitlines(keepends=True)
    path.write_text(json.dumps(notebook), encoding='utf-8')
    metadata_path = destination / 'kernel-metadata.json'
    metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
    title = f'ImageCompressor Native RD QP{qps[0]}' if len(qps) == 1 else 'ImageCompressor Native RD Pretraining'
    metadata.update(title=title, machine_shape='NvidiaTeslaT4')
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--owner', required=True)
    parser.add_argument('--run-updates', type=int, default=4096)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--previous')
    parser.add_argument('--qps', nargs='+', type=int, default=[28, 31, 34, 37, 41])
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    build_rd(args.output, args.owner, args.run_updates, args.previous, args.batch_size, tuple(args.qps))


if __name__ == '__main__':
    main()
