"""Paper losses, fixed/adaptive schedules, and resumable optimizer steps."""

import random
import tempfile
from pathlib import Path

import torch
from torch.nn import functional as F

from .schedules import learning_rate, make_schedule


PROXY_LAMBDA = {28: 4096, 31: 2048, 34: 1024, 37: 512, 41: 256, 44: 128}
TASK_LAMBDA = {28: 0.5, 31: 1., 34: 2., 37: 4., 41: 8.}
SCHEDULE = {'proxy': (100_000, 80_000), 'fixed': (30_000, 20_000), 'adaptive': (25_000, 15_000)}
BETA = 0.5
SAMPLING = 'epoch_shuffle_v1'


class EpochShuffle:
    """Index a reproducible stream of shuffled epochs by absolute sample position."""

    def __init__(self, size, seed):
        self.size = size
        self.seed = seed
        self.epoch = None
        self.order = []

    def batch(self, position, size):
        indices = []
        while len(indices) < size:
            epoch, index = divmod(position + len(indices), self.size)
            if epoch != self.epoch:
                self.order = list(range(self.size))
                random.Random(f'{self.seed}:{epoch}').shuffle(self.order)
                self.epoch = epoch
            indices.extend(self.order[index:index + size - len(indices)])
        return indices


def fixed_sample_offset(checkpoint, data, seed, task, annotations):
    metadata = checkpoint.get('metadata', {})
    expected = {'stage': 'fixed', 'data': str(data.resolve()), 'seed': seed, 'task': task,
                'annotations': str(annotations.resolve()) if annotations else None, 'sampling': SAMPLING}
    if all(metadata.get(key) == value for key, value in expected.items()):
        return metadata['sample_offset'] + checkpoint['step'] * metadata['batch_size']
    return 0


def use_value(measured, surrogate):
    """Forward equals measured. The derivative is the derivative of surrogate."""
    return measured.detach() + (surrogate - surrogate.detach())


def fit_proxy_loss(rgb, qp, proxy, codec):
    metrics = {}
    target = rgb
    if codec is not None:
        decoded, bits = zip(*(codec.roundtrip(x, qp) for x in rgb), strict=True)
        target = torch.stack(decoded)
        metrics['bpg_bpp'] = sum(bits) / (rgb.shape[0] * rgb.shape[-2] * rgb.shape[-1])
    reconstructed, bpp = proxy(rgb)
    distortion = F.mse_loss(reconstructed, target)
    loss = bpp + PROXY_LAMBDA[qp] * distortion
    return loss, {**metrics, 'proxy_bpp': bpp.detach().item(), 'target_mse': distortion.detach().item()}


def algorithm1(rgb, qp, npp, proxy, codec, task_loss):
    filtered = npp(rgb, qp)
    surrogate_rgb, surrogate_rate = proxy(filtered)
    if codec is None:
        task_rgb, rate = surrogate_rgb, surrogate_rate
    else:
        decoded, bits = zip(*(codec.roundtrip(image, qp) for image in filtered), strict=True)
        measured_rate = surrogate_rate.new_tensor(sum(bits) / (rgb.shape[0] * rgb.shape[-2] * rgb.shape[-1]))
        task_rgb = use_value(torch.stack(decoded), surrogate_rgb)
        rate = use_value(measured_rate, surrogate_rate)
    task = task_loss(task_rgb)
    pre_mse = F.mse_loss(rgb, filtered)
    loss = rate + TASK_LAMBDA[qp] * task + BETA * pre_mse
    return loss, {'bpp': rate.detach().item(), 'proxy_bpp': surrogate_rate.detach().item(),
                  'task': task.detach().item(), 'pre_mse': pre_mse.detach().item()}


def load_state(path):
    contents = torch.load(path, map_location='cpu', weights_only=True)
    return contents.get('model', contents.get('state_dict', contents))


def save_checkpoint(path, model, optimizer, step, metadata, plan=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name, suffix='.tmp', delete=False) as stream:
        temporary = Path(stream.name)
    try:
        state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                 'step': step, 'metadata': metadata}
        if plan is not None:
            state['plan'] = plan
        torch.save(state, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def optimize(args, model, stage, dataset, update, metadata, sample_offset=0):
    if stage == 'rd_pretrain':
        if args.steps is None or args.steps < 2:
            raise ValueError('RD pretraining requires an explicit --steps budget of at least 2')
        max_steps, decay = args.steps, max(1, int(args.steps * 0.8))
    else:
        max_steps, decay = SCHEDULE[stage]
    max_steps = max_steps if args.steps is None else args.steps
    decay = decay if args.decay is None else args.decay
    lr_schedule = make_schedule(getattr(args, 'lr_schedule', 'step'), max_steps, decay,
                                getattr(args, 'warmup_steps', None), getattr(args, 'cosine_start', None))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    metadata = {**metadata, 'stage': stage, 'seed': args.seed, 'batch_size': args.batch_size,
                'decay': decay, 'data': str(Path(args.data).resolve())}
    start = 0
    schedule_history = []
    sampler = EpochShuffle(len(dataset), args.seed)
    if args.resume:
        state = torch.load(args.resume, map_location='cpu', weights_only=True)
        saved = state['metadata']
        if 'sampling' not in saved:
            # Keep the exact original sampler and RNG consumption for old resumes.
            sampler = None
        elif saved['sampling'] == SAMPLING:
            sample_offset = saved['sample_offset']
        else:
            raise ValueError(f"Unknown checkpoint sampling scheme {saved['sampling']}")
    if sampler is not None:
        metadata.update(sampling=SAMPLING, sample_offset=sample_offset)
    if args.resume:
        saved_metadata = dict(state['metadata'])
        if getattr(args, 'allow_schedule_change', False):
            saved_metadata['decay'] = decay
        if saved_metadata != metadata:
            raise ValueError('Resume requires matching training data and configuration')
        previous_plan = state.get('plan', {})
        previous_lr = previous_plan.get('lr_schedule', make_schedule(
            'step', previous_plan.get('target_steps', max_steps), state['metadata']['decay']))
        lr_changed = previous_lr != lr_schedule or (
            lr_schedule['kind'] == 'cosine' and previous_plan.get('target_steps', max_steps) != max_steps)
        if lr_changed and not getattr(args, 'allow_schedule_change', False):
            raise ValueError('Changing the LR schedule on resume requires --allow-schedule-change')
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        start = state['step']
        schedule_history = list(previous_plan.get('schedule_history', []))
        if getattr(args, 'allow_schedule_change', False) and (
                previous_plan.get('target_steps', max_steps) != max_steps or state['metadata']['decay'] != decay
                or lr_changed):
            change = {'step': start, 'previous_target_steps': previous_plan.get('target_steps'),
                      'previous_decay_step': state['metadata']['decay'],
                      'target_steps': max_steps, 'decay_step': decay}
            if previous_lr['kind'] == 'cosine' or lr_schedule['kind'] == 'cosine':
                change.update(previous_lr_schedule=previous_lr, lr_schedule=lr_schedule)
            schedule_history.append(change)
            print(f'Explicit schedule change: {change}; restoring model and optimizer', flush=True)
    if max_steps <= start:
        raise ValueError(f'--steps ({max_steps}) must exceed completed steps ({start})')

    run_updates = getattr(args, 'run_updates', None)
    end = min(max_steps, start + run_updates) if run_updates is not None else max_steps
    plan = {'target_steps': max_steps, 'decay_step': decay, 'initial_lr': 1e-4, 'decayed_lr': 1e-5}
    plan['lr_schedule'] = lr_schedule
    if schedule_history:
        plan['schedule_history'] = schedule_history
    print(f'{stage} session {start + 1}..{end}, full target {max_steps}, LR schedule {lr_schedule}', flush=True)
    for step in range(start, end):
        if sampler is None:
            rng = random.Random(args.seed + step)
            torch.manual_seed(args.seed + step)
            indices = rng.choices(range(len(dataset)), k=args.batch_size)
        else:
            position = sample_offset + step * args.batch_size
            rng = random.Random(args.seed + position)
            torch.manual_seed(args.seed + position)
            indices = sampler.batch(position, args.batch_size)
        batch = [dataset[i] for i in indices]
        optimizer.param_groups[0]['lr'] = learning_rate(step + 1, max_steps, lr_schedule)
        optimizer.zero_grad(set_to_none=True)
        loss, metrics = update(batch, rng)
        if not torch.isfinite(loss):
            raise RuntimeError(f'Non-finite loss at step {step + 1}: {metrics}')
        loss.backward()
        if stage in ('proxy', 'rd_pretrain'):
            torch.nn.utils.clip_grad_value_(model.parameters(), 5)
        optimizer.step()
        completed = step + 1
        if completed == 1 or completed % args.log_every == 0 or completed == end:
            values = ' '.join(f'{key}={value:.6g}' for key, value in metrics.items())
            current_lr = optimizer.param_groups[0]['lr']
            print(f'{stage} {completed}/{max_steps} loss={loss.item():.6g} {values} lr={current_lr:.6g}', flush=True)
        if completed % args.save_every == 0 or completed == end:
            save_checkpoint(args.output, model, optimizer, completed, metadata, plan)
