"""Calibrate native RD candidates against BPG, then fine-tune a resumable QP proxy."""
import ast
import importlib.util
import json
from pathlib import Path
import re

spec = importlib.util.spec_from_file_location('native_base_builder', Path(__file__).with_name('prepare.py'))
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)


def proxy_commands(qp, updates=8192, resume=False):
    if qp not in (28,31,34,37,41) or not 1 <= updates <= 100000:
        raise ValueError('Require a paper QP and a positive session budget')
    commands = [['make-crops', '--data', '{FLICKR}', '--output', '{WORK}/crops.json']]
    common = ['--device', 'cuda:0', '--threads', '2']
    if not resume:
        commands.append(['calibrate-proxy', '--data', '{WORK}/crops.json', '--candidate-dir', '{WORK}/rd_candidates',
                         '--qps', str(qp), '--samples', '64', '--output', '{WORK}/calibration.json', *common])
    initial = ['--resume', f'{{WORK}}/proxy/qp{qp}.pt'] if resume else ['--calibration', '{WORK}/calibration.json']
    commands.append(['train-proxy', '--data', '{WORK}/crops.json', '--qp', str(qp), '--steps', '100000',
                     '--decay', '80000', '--run-updates', str(updates), '--batch-size', '4', '--seed', '2024',
                     '--save-every', '256', '--log-every', '128', '--output', f'{{WORK}}/proxy/qp{qp}.pt',
                     *common, *initial])
    commands.append(['audit-proxy', '--data', '{WORK}/crops.json', '--qps', str(qp), '--samples', '64',
                     '--proxy-dir', '{WORK}/proxy', '--output', '{WORK}/audit.json', *common])
    return commands


SETUP = '''
import shutil
import torch
sys.path.insert(0, str(root))
from npp_proxy.native import PROXY_KIND, UPSTREAM_COMMIT

assert torch.cuda.is_available(), 'Native proxy fine-tuning requires CUDA'
inputs = Path('/kaggle/input')
flickr = next((p for p in (inputs / 'datasets/eeshawn/flickr30k', inputs / 'flickr30k') if p.is_dir()), None)
if flickr is None:
    raise FileNotFoundError('Version-pinned Flickr30k input missing')
replacements['{FLICKR}'] = str(flickr)

def unique(handle, name, parent_name=None):
    owner, slug, version = handle.split('/')
    parents = (inputs / 'notebooks' / owner, inputs / owner, inputs)
    mounts = {parent / alias for parent in parents for alias in (slug, f'{slug}-versions-{version}')
              if (parent / alias).is_dir()}
    found = {p.resolve() for mount in mounts for p in mount.rglob(name)
             if parent_name is None or p.parent.name == parent_name}
    if len(found) != 1:
        raise ValueError(f'Expected one {name} ({parent_name}) in {handle}: {found}')
    return found.pop()

candidates = work / 'rd_candidates'
candidates.mkdir(exist_ok=True)
rd_hashes = {}
manifest_hash = None
for qp_text, handle in RD_SOURCES.items():
    source = unique(handle, f'qp{qp_text}.pt', 'rd')
    saved = torch.load(source, map_location='cpu', weights_only=True, mmap=True)
    metadata, plan = saved['metadata'], saved['plan']
    assert metadata['stage'] == 'rd_pretrain' and metadata['target'] == 'source'
    assert metadata['proxy_kind'] == PROXY_KIND and metadata['channels'] == [192,192]
    assert metadata['upstream_commit'] == UPSTREAM_COMMIT and metadata['qp'] == int(qp_text)
    assert saved['step'] == 100000 and plan['target_steps'] == 100000 and plan['decay_step'] == 80000
    assert plan['lr_schedule'] == {'kind':'step', 'initial_lr':1e-4, 'decayed_lr':1e-5, 'decay_step':80000}
    current_hash = metadata['crop_manifest_sha256']
    assert manifest_hash is None or manifest_hash == current_hash, 'RD candidates used different crops'
    manifest_hash = current_hash
    del saved
    # Copies preserve calibration containment and exact checkpoint hashes across sessions.
    target = candidates / f'qp{qp_text}.pt'
    shutil.copy2(source, target)
    rd_hashes[qp_text] = hashlib.sha256(target.read_bytes()).hexdigest()

bank = work / 'proxy'
bank.mkdir(exist_ok=True)
start = 0
if PREVIOUS:
    source = unique(PREVIOUS, f'qp{QP}.pt', 'proxy')
    saved = torch.load(source, map_location='cpu', weights_only=True, mmap=True)
    metadata, plan = saved['metadata'], saved['plan']
    assert metadata['proxy_kind'] == PROXY_KIND and metadata['stage'] == 'proxy'
    assert metadata['target'] == 'bpg' and metadata['qp'] == QP and metadata['codec_level'] == 8
    assert plan['target_steps'] == 100000 and plan['decay_step'] == 80000
    start = saved['step']
    assert 0 < start < 100000
    del saved
    shutil.copy2(source, bank / f'qp{QP}.pt')
    shutil.copy2(unique(PREVIOUS, 'calibration.json'), work / 'calibration.json')
    report = json.loads((work / 'calibration.json').read_text())
    assert report['manifest_sha256'] == manifest_hash and report['proxy_kind'] == PROXY_KIND
    chosen = report['points'][str(QP)]
    assert hashlib.sha256((candidates / chosen['checkpoint']).read_bytes()).hexdigest() == chosen['checkpoint_sha256']

def progress(complete_session):
    path = bank / f'qp{QP}.pt'
    step, lr = start, None
    if path.is_file():
        saved = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
        step, lr = saved['step'], saved['optimizer']['param_groups'][0]['lr']
        del saved
    result = {'stage':'proxy', 'proxy_kind':PROXY_KIND, 'upstream_commit':UPSTREAM_COMMIT,
              'source_sha256':SOURCE_SHA256, 'target_per_qp':100000, 'decay_after':80000,
              'session_complete':complete_session, 'rd_sources':RD_SOURCES, 'rd_hashes':rd_hashes,
              'models':{str(QP):{'start_step':start,'step':step,'lr':lr,'complete':step==100000}},
              'complete':step==100000}
    calibration = work / 'calibration.json'
    if calibration.is_file():
        result['initialization'] = json.loads(calibration.read_text())['points'][str(QP)]
    (work / 'progress.json').write_text(json.dumps(result, indent=2))
    print('NATIVE_BPG_PROGRESS', json.dumps(result), flush=True)

progress(False)
'''


def build_proxy(destination, owner, qp, rd_sources, updates=8192, previous=None):
    if set(rd_sources) != {'28','31','34','37','41'}:
        raise ValueError('Require all five completed native RD sources')
    for handle in [*rd_sources.values(), *([previous] if previous else [])]:
        if not re.fullmatch(r'[A-Za-z0-9_-]+/[A-Za-z0-9_-]+/[1-9][0-9]*', handle):
            raise ValueError('Checkpoint sources must be version-pinned')
    destination = Path(destination)
    commands = proxy_commands(qp, updates, bool(previous))
    slug = f'imagecompressor-native-bpg-qp{qp}'
    base.build(destination, f'{owner}/{slug}', commands, ['eeshawn/flickr30k/1'],
               list(dict.fromkeys([*rd_sources.values(), *([previous] if previous else [])])))
    path = destination / 'notebook.ipynb'
    notebook = json.loads(path.read_text(encoding='utf-8'))
    exercise = ast.literal_eval(ast.parse(''.join(notebook['cells'][0]['source'])).body[0].value)
    exercise = f'QP={qp!r}\nRD_SOURCES={rd_sources!r}\nPREVIOUS={previous!r}\n' + exercise
    exercise = exercise.replace('for command in COMMAND_LIST:', SETUP + '\nfor command in COMMAND_LIST:', 1)
    marker = "'--bpgenc', bpg / 'bpgenc', '--bpgdec', bpg / 'bpgdec', cwd=root)"
    exercise = exercise.replace(marker, marker + "\n    if command[0] == 'make-crops':\n        assert hashlib.sha256((work / 'crops.json').read_bytes()).hexdigest() == manifest_hash\n    progress(False)\nprogress(True)")
    compile(exercise, 'native-bpg-runtime', 'exec')
    code = f'EXERCISE = {exercise!r}\n' + base.BOOTSTRAP
    compile(code, 'native-bpg-notebook', 'exec')
    notebook['cells'][0]['source'] = code.splitlines(keepends=True)
    path.write_text(json.dumps(notebook), encoding='utf-8')
    metadata_path = destination / 'kernel-metadata.json'
    metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
    metadata.update(title=f'ImageCompressor Native BPG QP{qp}', machine_shape='NvidiaTeslaT4')
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding='utf-8')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--owner', required=True)
    parser.add_argument('--qp', required=True, type=int)
    parser.add_argument('--rd-sources', required=True, type=Path, help='JSON map of QP to owner/slug/version')
    parser.add_argument('--run-updates', type=int, default=8192)
    parser.add_argument('--previous')
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    build_proxy(args.output, args.owner, args.qp, json.loads(args.rd_sources.read_text(encoding='utf-8')),
                args.run_updates, args.previous)
