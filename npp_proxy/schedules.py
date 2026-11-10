"""Learning rates indexed by completed update, independent of session boundaries."""

import math


def make_schedule(kind, target, decay, warmup_steps=None, cosine_start=None):
    if kind == 'step':
        return {'kind': 'step', 'initial_lr': 1e-4, 'decayed_lr': 1e-5, 'decay_step': decay}
    if kind != 'cosine':
        raise ValueError(f'Unknown LR schedule {kind}')
    warmup = target // 100 if warmup_steps is None else warmup_steps
    start = warmup if cosine_start is None else cosine_start
    if not 0 <= warmup <= start < target:
        raise ValueError('Cosine requires 0 <= warmup_steps <= cosine_start < target')
    return {'kind': 'cosine', 'initial_lr': 1e-4, 'min_lr': 1e-6,
            'warmup_steps': warmup, 'cosine_start': start}


def learning_rate(completed, target, schedule):
    """Rate used for update `completed` (one based), including the final update."""
    if not 1 <= completed <= target:
        raise ValueError('Completed update must be within 1..target')
    peak = schedule['initial_lr']
    if schedule['kind'] == 'step':
        return schedule['decayed_lr'] if completed > schedule['decay_step'] else peak
    if schedule['kind'] != 'cosine':
        raise ValueError(f'Unknown LR schedule {schedule["kind"]}')
    warmup, start, floor = schedule['warmup_steps'], schedule['cosine_start'], schedule['min_lr']
    if not 0 <= warmup <= start < target or not 0 < floor <= peak:
        raise ValueError('Invalid cosine schedule')
    if warmup > 1 and completed <= warmup:
        return floor + (peak - floor) * (completed - 1) / (warmup - 1)
    if completed <= start:
        return peak
    fraction = (completed - start) / (target - start)
    return floor + 0.5 * (peak - floor) * (1 + math.cos(math.pi * fraction))
