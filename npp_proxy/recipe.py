"""Explicit quarter-budget phase gates; native weights alone are not trained proxies."""

import torch

from .native import PROXY_KIND


TARGETS = {'proxy': 100000, 'fixed': 30000, 'adaptive': 25000}
DECAYS = {'proxy': 80000, 'fixed': 20000, 'adaptive': 15000}


def require_completed_native(path, stage, qp=None):
    state = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
    metadata, plan = state.get('metadata', {}), state.get('plan', {})
    if (metadata.get('stage') != stage or state.get('step') != TARGETS[stage]
            or plan.get('target_steps') != TARGETS[stage] or plan.get('decay_step') != DECAYS[stage]):
        raise ValueError(f'{path} is not a completed {stage} checkpoint for the quarter recipe')
    if plan.get('lr_schedule', {'kind': 'step'}).get('kind') != 'step':
        raise ValueError(f'{path} uses a different learning-rate schedule')
    if stage != 'proxy' and metadata.get('npp_kind') != 'author_preprocessor':
        raise ValueError(f'{path} is not an author Preprocessor checkpoint')
    if stage != 'proxy' and (metadata.get('codec_level') != 8 or metadata.get('proxy_only') is not False):
        raise ValueError(f'{path} must use real BPG level 8')
    if stage == 'fixed' and metadata.get('qps') != [34]:
        raise ValueError(f'{path} must be fixed training at QP34')
    if stage == 'proxy' and (metadata.get('proxy_kind') != PROXY_KIND or metadata.get('target') != 'bpg'
                             or metadata.get('qp') != qp or metadata.get('codec_level') != 8):
        raise ValueError(f'{path} must be a native BPG-finetuned proxy for QP {qp}, codec level 8')
    return state
