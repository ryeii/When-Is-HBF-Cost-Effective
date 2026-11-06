"""Frozen finite workload interventions for the conditional deployment sweep.

Main cases are controlled token-shape programs, not production workloads or
quality evaluations. TraceLab contributes a separate sampled length/tool-gap
case; it never supplies cache identity. Exact public text/cache experiments
remain separately reproducible in identity-workloads-v3/prefix-traffic-v4.
"""
from functools import lru_cache
import gzip
import hashlib
import json
import math
import random

from run_validation import HERE
from prefix_workloads import block_keys

WORKLOADS=['short_chat','long_rag','tool_agent','paused_personal','offline_batch']


@lru_cache(maxsize=1)
def tool_shapes():
    path=HERE/'results/causal-workloads-v2-20261104/causal-shapes.jsonl.gz'
    if not path.exists():
        path=next((HERE/'results/causal-workloads-v2-20261104').glob('*.gz'))
    selected=[]
    with gzip.open(path,'rt') as stream:
        for line in stream:
            row=json.loads(line)
            # Keep a bounded, explicit public-shape sample before design
            # evaluation. Missing identity remains missing.
            if row['split']=='heldout_workload' and row['link']['kind']=='parallel_tool_barrier':
                selected.append(row)
    # Stable cohort chosen independently of hardware and simulation outcomes.
    selected.sort(key=lambda row:hashlib.sha256(row['round_id'].encode()).digest())
    if not selected:raise ValueError('missing public held-out tool shapes')
    return selected[:8192]


def lengths(row):
    # Converter field names are checked explicitly, never guessed silently.
    return row['input_tokens'],row['output_tokens']


def make(workload,seed,roots,*,arrival_rate=None,rounds=2):
    rng=random.Random(seed)
    namespace=f'controlled-token-shape-program-v1:{seed}'
    common=random.Random(7).choices(range(128000),k=8192)
    records=[];arrival=0.
    shapes=tool_shapes() if workload=='tool_agent' else None
    for session in range(roots):
        if arrival_rate is not None and session:
            arrival+=rng.expovariate(arrival_rate)
        if workload=='short_chat':initial=512;out=64;gap=1.
        elif workload=='long_rag':initial=16384;out=64;gap=0.
        elif workload=='paused_personal':initial=4096;out=64;gap=60.
        elif workload=='offline_batch':initial=2048;out=128;gap=0.
        elif workload=='tool_agent':
            row=shapes[(seed*137+session*31)%len(shapes)]
            initial,out=lengths(row);initial=max(16,min(32768,initial));out=max(2,min(128,out))
            gap=float(row['link']['delay_seconds'])
        else:raise ValueError('unknown workload')
        tokens=rng.choices(range(128000),k=initial)
        if workload=='long_rag':tokens[:8192]=common
        for turn in range(1 if workload in ('long_rag','offline_batch') else rounds):
            output=rng.choices(range(128000),k=out)
            # Tool shapes have no token IDs in the public release. They do not
            # acquire synthetic cache eligibility through this shape program.
            keys=[] if workload=='tool_agent' else block_keys(tokens,namespace)
            completed=[] if workload=='tool_agent' else block_keys(tokens+output[:-1],namespace)
            records.append(dict(id=f's{session}-t{turn}',session=session,turn=turn,
                input_tokens=len(tokens),output_tokens=out,input_blocks=keys,completed_blocks=completed,
                root_arrival_s=arrival,predecessor=None if turn==0 else f's{session}-t{turn-1}',
                successor_gap_s=gap,source='public_shape_transfer' if shapes else 'controlled_exact_token_program'))
            tokens=tokens+output+rng.choices(range(128000),k=64 if workload=='short_chat' else 256)
    return records


def manifest():
    return dict(schema='hbf-adoption.controlled-workloads.v1',seeds=[17,29],
        short_chat=dict(prompt_tokens=512,output_tokens=64,rounds=2,think_seconds=1),
        long_rag=dict(prompt_tokens=16384,common_prefix_tokens=8192,output_tokens=64),
        paused_personal=dict(prompt_tokens=4096,output_tokens=64,rounds=2,think_seconds=60),
        offline_batch=dict(prompt_tokens=2048,output_tokens=128,deadline_seconds=3600),
        tool_agent=dict(source='TraceLab 8192 lowest round-id hashes among held-out parallel-tool transitions; deterministic index sample',
                        input_cap=32768,output_cap=128,cache_identity=False),
        bounds='Finite deterministic programs; no production arrival, model quality or steady-state claim.')
