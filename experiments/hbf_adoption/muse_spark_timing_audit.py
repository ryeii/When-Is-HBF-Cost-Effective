"""Check the Muse timing adapter with public H200 measurements and shape tests."""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
from statistics import median

import pyarrow.parquet as pq

from muse_spark_timing import (MuseTiming, SYSTEMS, component_estimate, deployment_model,
                               local_attention, reservation_bytes)
from projection_timing import Layout
from run_validation import fingerprint

TABLES = SYSTEMS/'data/h200_sxm'
MOE = TABLES/'moe/vllm/0.30.0/moe_perf.parquet'
ATTENTION = TABLES/'attention/vllm/0.30.0/generation_attention_perf.parquet'


def summarize(errors):
    errors = sorted(errors)
    return dict(n=len(errors), median_ape=median(errors),
                p95_ape=errors[math.ceil(.95*len(errors))-1])


def expert_transfer():
    groups = defaultdict(list)
    for row in pq.read_table(MOE).to_pylist():
        if (row['moe_dtype'], row['distribution'], row['num_experts'], row['topk'],
                row['moe_ep_size']) != ('bfloat16', 'balanced', 256, 8, 1):
            continue
        if row['moe_tp_size'] not in (1, 2, 4, 8) or row['latency'] <= 0:
            continue
        shape = row['hidden_size'], row['inter_size']
        groups[shape, row['moe_tp_size'], row['num_tokens']].append(row['latency'])
    # Hold out a smaller BF16 expert shape with the same top-k/expert count.
    # The operator estimator scales with the product of matrix dimensions.
    # This cross-shape check cannot measure error at Muse's 4096x4096.
    donor, target = (3072, 1024), (4096, 2048)
    errors = []
    for (shape, tp, tokens), data in groups.items():
        key = donor, tp, tokens
        if shape == target and key in groups:
            prediction = median(groups[key])*math.prod(target)/math.prod(donor)
            errors.append(abs(prediction/median(data)-1))
    return dict(donor=donor, holdout=target, **summarize(errors))


def sliding_transfer():
    groups = defaultdict(list)
    for row in pq.read_table(ATTENTION).to_pylist():
        if (row['head_dim'], row['kv_cache_dtype'], row['attn_dtype']) != (64, 'bfloat16', 'bfloat16'):
            continue
        key = tuple(row[k] for k in ('kernel_source', 'batch_size', 'num_heads', 'num_key_value_heads'))
        groups[key, row['window_size'], row['isl']+row['step']].append(row['latency'])
    errors = []
    for (key, window, length), values in groups.items():
        if window != 128 or length <= window:
            continue
        capped = groups.get((key, 0, window))
        if capped:
            errors.append(abs(median(capped)/median(values)-1))
    return dict(held_out_window=128, **summarize(errors))


def check_graph():
    model = deployment_model()
    layout = Layout(1, 1, 1, 1, 'h3_cascaded', 8, 4)
    assert MuseTiming(model, 'h200', layout).stage_weight_bytes(0) == 2*model['parameters']
    assert reservation_bytes(model, 32768, 1, 1) == (16*32768+46*2048)*4096
    assert reservation_bytes(dict(model, archive_prefix_kv=True), 32768, 1, 1) == 62*32768*4096
    audit = []
    for tp in (1, 2, 4, 8):
        for prefill, b, q, ctx in ((True, 1, 512, 512), (True, 4, 16384, 33152),
                                  (False, 1, 1, 512), (False, 16, 1, 32768)):
            for gpu in ('h200', 'rtx4090'):
                values, sources = component_estimate(gpu, tp, prefill, b, q, ctx)
                assert all(math.isfinite(x) and x > 0 for x in values)
                audit.append(dict(gpu=gpu, tp=tp, prefill=prefill, batch=b, query=q,
                                  context=ctx, values_s=values, sources=sources))
        assert local_attention(tp, False, 1, 1, 8192, 2048) == local_attention(tp, False, 1, 1, 32768, 2048)
    return audit


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', required=True, type=Path)
    a = p.parse_args()
    report = dict(scope='Cross-geometry and cross-window transfer checks; not Muse serving validation.',
                  expert_transfer=expert_transfer(), sliding_decode_transfer=sliding_transfer(),
                  graph_checks=check_graph(), inputs=[fingerprint(MOE), fingerprint(ATTENTION)])
    with a.out.open('x') as f:
        json.dump(report, f, indent=2)
        f.write('\n')
    print(json.dumps({k: report[k] for k in ('expert_transfer', 'sliding_decode_transfer')}))


if __name__ == '__main__':
    main()
