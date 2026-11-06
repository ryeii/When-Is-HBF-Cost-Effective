"""Independent, standard-library audit of the released serving measurements.

Nothing here fits simulator parameters. Released simulations and new reruns are
kept separate. Epoch/monotonic conversion is estimated, never called measured.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / 'packages/serving/LLMServingSim'


def artifact(path: Path) -> dict:
    return {'path': str(path.resolve().relative_to(ROOT)),
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'bytes': path.stat().st_size}


def quantile(values: list[float], q: float) -> float:
    """Linear interpolation, equivalent to NumPy's default percentile."""
    if not values:
        raise ValueError('empty sample')
    ordered = sorted(values)
    p = (len(ordered) - 1) * q
    low = math.floor(p)
    return ordered[low] + (ordered[math.ceil(p)] - ordered[low]) * (p - low)


def compare(observed: list[float], predicted: list[float]) -> dict:
    if len(observed) != len(predicted) or not observed or min(observed) <= 0:
        raise ValueError('comparison requires matched, positive observations')
    errors = [(p / o - 1) * 100 for o, p in zip(observed, predicted)]
    return {
        'n': len(observed), 'measured_mean': statistics.mean(observed),
        'simulated_mean': statistics.mean(predicted),
        'mean_signed_relative_error_percent': statistics.mean(errors),
        'mean_absolute_relative_error_percent': statistics.mean(map(abs, errors)),
        'median_absolute_relative_error_percent': statistics.median(map(abs, errors)),
        'aggregate_mean_bias_percent': (statistics.mean(predicted) / statistics.mean(observed) - 1) * 100,
        'measured_p95': quantile(observed, .95), 'simulated_p95': quantile(predicted, .95),
        'measured_p99': quantile(observed, .99), 'simulated_p99': quantile(predicted, .99),
        'p95_bias_percent': (quantile(predicted, .95) / quantile(observed, .95) - 1) * 100,
        'p99_bias_percent': (quantile(predicted, .99) / quantile(observed, .99) - 1) * 100,
    }


def load_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def audit(example: Path, sim_path: Path | None = None) -> dict:
    meta_path = example / 'vllm/meta.json'
    requests_path = example / 'vllm/requests.jsonl'
    sim_path = sim_path or example / 'outputs/sim.csv'
    meta = json.loads(meta_path.read_text())
    if meta.get('resolve_only') or (example / 'vllm/ABORTED.json').exists():
        raise ValueError('incomplete benchmark')
    audits = meta.get('step_audit') or []
    for item in audits if isinstance(audits, list) else [audits]:
        if (item.get('kernel_trace_steps') or item.get('idle_ms') or item.get('error')
                or item.get('end_to_end_control_eligible') is False):
            raise ValueError('instrumented benchmark cannot establish end-to-end fidelity')
    measured = load_rows(requests_path)
    with sim_path.open(newline='') as stream:
        simulated = list(csv.DictReader(stream))
    # The released bench emits bench-N, the simulator emits N. Reject any other
    # namespace rather than pairing by row order or sorting independent samples.
    by_id = {}
    for row in measured:
        rid = row['request_id']
        if not rid.startswith('bench-') or not rid[6:].isdigit() or rid in by_id:
            raise ValueError(f'unsupported/duplicate request identifier {rid}')
        by_id[rid] = row
    if len(simulated) != len(measured) or len(measured) != meta['num_requests']:
        raise ValueError('request population incomplete')
    offsets = [r['queued_ts'] - r['arrival_time'] for r in measured]
    offset = min(offsets)
    observations = {k: [] for k in ('ttft_ms', 'tpot_ms', 'latency_ms')}
    predictions = {k: [] for k in observations}
    aligned, seen = [], set()
    mismatches = []
    for sim in simulated:
        rid = 'bench-' + sim['request id']
        if rid in seen or rid not in by_id:
            raise ValueError(f'unmatched/duplicate simulated identifier {rid}')
        seen.add(rid)
        obs = by_id[rid]
        if (int(sim['input']) != obs['input_toks'] or int(sim['output']) != obs['output_toks']):
            mismatches.append({'id': rid, 'measured_input': obs['input_toks'],
                               'simulated_input': int(sim['input']),
                               'measured_output': obs['output_toks'],
                               'simulated_output': int(sim['output'])})
        arrival = obs['arrival_time'] + offset
        first, last = obs['first_token_ts'], obs['last_token_ts']
        # Subtracting epoch-sized doubles loses <1 microsecond of precision.
        # This tolerance admits only that rounding, not reordered engine events.
        if not arrival <= obs['queued_ts'] + 1e-6 or not obs['queued_ts'] <= first <= last:
            raise ValueError(f'invalid request clock {rid}')
        ttft, latency = (first - arrival) * 1000, (last - arrival) * 1000
        observations['ttft_ms'].append(ttft)
        predictions['ttft_ms'].append(float(sim['TTFT']) / 1e6)
        observations['latency_ms'].append(latency)
        predictions['latency_ms'].append(float(sim['latency']) / 1e6)
        tpot = None
        if obs['output_toks'] > 1:
            tpot = (last - first) * 1000 / (obs['output_toks'] - 1)
            observations['tpot_ms'].append(tpot)
            predictions['tpot_ms'].append(float(sim['TPOT']) / 1e6)
        # There are no measured per-token timestamps here. Never synthesize
        # p99 ITL from average TPOT or from the simulated ITL samples.
        aligned.append({'request_id': rid, 'measured_ttft_estimated_ms': ttft,
                        'measured_ttft_queued_lower_bound_ms': (first - obs['queued_ts']) * 1000,
                        'simulated_ttft_ms': float(sim['TTFT']) / 1e6,
                        'measured_tpot_ms': tpot,
                        'simulated_tpot_ms': float(sim['TPOT']) / 1e6})
    stats = {k: compare(observations[k], predictions[k]) for k in observations}
    measured_duration = max(r['last_token_ts'] for r in measured) - min(r['arrival_time'] + offset for r in measured)
    simulated_duration = (max(float(r['end_time']) for r in simulated) - min(float(r['arrival']) for r in simulated)) / 1e9
    measured_throughput = sum(r['output_toks'] for r in measured) / measured_duration
    simulated_throughput = sum(int(r['output']) for r in simulated) / simulated_duration
    dataset_path = PACKAGE / meta['dataset_path']
    dataset_matches = hashlib.sha256(dataset_path.read_bytes()).hexdigest() == meta['dataset_hash']
    gate = (not mismatches and dataset_matches
            and all(stats[k]['mean_absolute_relative_error_percent'] <= 10 for k in ('ttft_ms', 'tpot_ms', 'latency_ms'))
            and abs(simulated_throughput / measured_throughput - 1) <= .1)
    return {
        'example': str(example.relative_to(PACKAGE / 'bench/examples')),
        'kind': 'released_measurement_vs_' + ('released_simulation' if sim_path == example / 'outputs/sim.csv' else 'new_simulation'),
        'new_hardware_measurement': False, 'hardware': meta['hardware'],
        'measured_at': meta['started_at'], 'vllm_version': meta['vllm_version'],
        'artifacts': [artifact(p) for p in (meta_path, requests_path, sim_path, example / 'config.json', dataset_path)],
        'dataset_hash_matches_metadata': dataset_matches,
        'length_mismatches': mismatches, 'clock_offset_estimate_seconds': offset,
        'clock_qualification': 'minimum queued-minus-arrival estimator; unknown nonnegative minimum engine-pickup delay; arrival TTFT is a lower bound without a synchronized clock measurement',
        'queued_minus_estimated_arrival_ms': {'median': statistics.median((v-offset)*1000 for v in offsets), 'max': max((v-offset)*1000 for v in offsets)},
        'metrics': stats,
        'throughput': {'measured_output_tokens_per_second': measured_throughput,
                       'simulated_output_tokens_per_second': simulated_throughput,
                       'relative_error_percent': (simulated_throughput / measured_throughput - 1) * 100},
        'provisional_10_percent_mean_absolute_relative_error_gate': 'pass' if gate else 'fail',
        'adoption_gate': 'not_established',
        'missing_validation': ['per-request p99 inter-token intervals', 'synchronized arrival clocks',
                               'independent held-out workloads', 'HBF-coupled GPU execution',
                               'distributed HBF/network coupling beyond released node configurations'],
        'aligned_requests': aligned,
    }


def report(results: list[dict]) -> str:
    lines = ['# Serving evidence audit', '',
             'This independently recomputes comparisons from released raw request records. It does not run GPU inference or fit parameters. A small error in an aggregate mean does not establish small errors on individual requests.', '',
             '| Released configuration | Requests | TTFT MARE | TPOT MARE | E2E MARE | Output throughput error | Provisional gate |',
             '| --- | ---: | ---: | ---: | ---: | ---: | --- |']
    for r in results:
        m = r['metrics']
        lines.append(f"| {r['example']} | {m['ttft_ms']['n']} | {m['ttft_ms']['mean_absolute_relative_error_percent']:.2f}% | {m['tpot_ms']['mean_absolute_relative_error_percent']:.2f}% | {m['latency_ms']['mean_absolute_relative_error_percent']:.2f}% | {r['throughput']['relative_error_percent']:+.2f}% | {r['provisional_10_percent_mean_absolute_relative_error_gate']} |")
    lines += ['', 'MARE means the mean of each matched request’s absolute relative error. The provisional gate requires ≤10% MARE for TTFT, average TPOT and end-to-end latency, ≤10% throughput error, identical lengths and a matching workload hash. It is necessary, not sufficient, for the planned adoption validation gate.', '',
              'Arrival times use epoch seconds; engine times use a monotonic clock. Following the source’s documented estimator, this audit uses the minimum queued-minus-arrival difference. Unknown minimum queue-pickup delay biases estimated TTFT downward; results are conditional on that clock conversion. Average TPOT is not a tail inter-token latency.', '',
              'These are released validation workloads, not an independent holdout. Every adoption decision remains unvalidated. See the JSON for signed errors, p95/p99 request distributions, source hashes, token-length mismatches and aligned request residuals.', '']
    return '\n'.join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--example')
    parser.add_argument('--sim-csv', type=Path)
    args = parser.parse_args()
    if args.sim_csv and not args.example:
        parser.error('--sim-csv requires --example')
    examples = ([PACKAGE / 'bench/examples' / args.example] if args.example else
                sorted(p.parent for p in (PACKAGE / 'bench/examples').glob('*/*/config.json')))
    results = [audit(p, args.sim_csv) for p in examples]
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / 'audit.json').write_text(json.dumps({'schema': 'hbf-adoption.serving-evidence.v1', 'results': results}, indent=2, allow_nan=False) + '\n')
    (args.out / 'REPORT.md').write_text(report(results))
    print(report(results))


if __name__ == '__main__':
    main()
