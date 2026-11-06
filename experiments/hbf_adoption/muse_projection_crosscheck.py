"""Compare NVIDIA GEMM queries to public H200 Muse projection measurements."""
import argparse
import json
from pathlib import Path

from muse_spark_timing import database_engine

SOURCE = 'https://thethings.ai/p/okc/muse-bf16-proj-report'
BENCHMARK = 'https://openkernelchallenge.dev/problems/muse_bf16_proj'
# Per-load SGLang baseline, public judge check 0fdd01df8a014f82. The benchmark
# times five local GPU projections without attention, experts or collectives.
REFERENCE_US = {1: 59.7, 8: 75.1, 16: 77.3, 32: 78.4, 128: 86.9, 256: 100.1}


def compare():
    from aiconfigurator_core.sdk import common, operations
    from aiconfigurator_core.sdk.engine import build_ops_json

    engine = database_engine()
    comparisons = []
    for tokens, measured in REFERENCE_US.items():
        shapes = {'fused_qkv_gate': (2304, 8192), 'o': (8192, 1024),
                  'pooler': (4096, 8192), 'unpooler': (1024 if tokens <= 8 else 8192, 4096),
                  'router_bf16_proxy': (256, 8192)}
        spec = build_ops_json([operations.GEMM(name, 1, n, k, common.GEMMQuantMode.bfloat16)
                               for name, (n, k) in shapes.items()])
        rows = engine.evaluate_ops_json(spec, is_context=False, batch_size=tokens, s=1, x=tokens)
        estimated = sum(r[1] for r in rows)*1000
        comparisons.append(dict(tokens=tokens, measured_us=measured, estimated_us=estimated,
                                signed_relative_error=estimated/measured-1,
                                operators_us={name: latency*1000 for name, latency, _, _ in rows},
                                sources={r[0]: r[3] for r in rows}))
    return comparisons


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', required=True, type=Path)
    args = p.parse_args()
    report = dict(scope='Independent local projection cross-check on the benchmark TP8 shapes. '
                       'The served TP+EP schedule and precision differ from the study. '
                       'The TF32/FP32 router is approximated by BF16 in these queries.',
                  benchmark=BENCHMARK, reported_measurements=SOURCE, rows=compare())
    with args.out.open('x') as f:
        json.dump(report, f, indent=2)
        f.write('\n')
    for row in report['rows']:
        print(f"{row['tokens']:3d} tokens: {row['estimated_us']:.1f} vs {row['measured_us']:.1f} µs")


if __name__ == '__main__':
    main()
