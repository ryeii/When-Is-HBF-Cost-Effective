"""Audit Muse trajectories and compute the study's matched adoption frontiers."""
import argparse
from collections import Counter, defaultdict
import gzip
from itertools import product
import json
import math
from pathlib import Path
from unittest.mock import patch

import analyze_deployments as adoption
from muse_spark_timing import deployment_model
from projection_sweep import COUNTS, WORKLOADS, design_key
from projection_engine import Design
from projection_timing import GPUS, Layout
from run_validation import ROOT, fingerprint


def verify(record):
    path = ROOT/record['path']
    if fingerprint(path) != record:
        raise ValueError(f'Changed Muse result: {path}')
    return path


def audit_template(path):
    with gzip.open(path, 'rt') as stream:
        run = json.load(stream)
    requests = {r['id']: r for r in run['requests']}
    if (run['status'] != 'simulated_projection' or run['validated']
            or len(requests) != run['completed_requests'] or len(requests) != len(run['requests'])):
        raise ValueError(f'Incomplete Muse execution: {path}')
    for request in requests.values():
        tokens = request['token_times_s']
        if (len(tokens) != request['expected_output_tokens'] or tokens[0] < request['arrival_s']
                or any(b < a for a, b in zip(tokens, tokens[1:]))):
            raise ValueError(f'Noncausal token trajectory: {path}')
        if request['predecessor'] is not None:
            previous = requests[request['predecessor']]
            expected = previous['token_times_s'][-1]+previous['successor_gap_s']
            if not math.isclose(request['arrival_s'], expected, rel_tol=0., abs_tol=1e-7):
                raise ValueError(f'Noncausal successor release: {path}')
    if any(resource['utilization'] > 1.000001 for resource in run['resources'].values()):
        raise ValueError(f'Resource overbooked: {path}')
    return len(requests), run['events']


def key_for_design(data):
    return design_key(Design(**{**data, 'layout': Layout(**data['layout'])}))


def expected_outcomes(screen):
    expected = set()
    for origin, designs in (('matched_central_screen', screen['central']),
                            ('fixed_four_user_secondary_screen', screen['additions'])):
        for design in designs:
            layout = design['layout']
            wide_tool = screen['workload'] == 'tool_agent' and layout['tp']*layout['pp'] > 8
            views = ('fixed_users',)
            if origin == 'matched_central_screen' and not wide_tool:
                views += tuple(f'users_per_gpu_{n}' for n in (1, 4, 16))
            expected.update((key_for_design(design), view, seed, origin)
                            for view, seed in product(views, (17, 29)))
    return expected


def collect(matrix):
    manifest = json.loads((matrix/'manifest.json').read_text())
    expected = {f'{gpu}-{w}-{n}' for gpu, w, n in product(GPUS, WORKLOADS, COUNTS)}
    if manifest['status'] != 'completed' or set(manifest['cells']) != expected:
        raise ValueError('A complete 90-cell Muse matrix is required')
    groups = defaultdict(list)
    integrity = Counter()
    templates = {}
    for key, cell in manifest['cells'].items():
        worker_file = verify(cell['receipt'])
        worker = json.loads(worker_file.read_text())
        if worker['status'] != 'completed' or worker['cells_completed'] != 1:
            raise ValueError(f'Incomplete cell {key}')
        outputs = {p.name: p for p in map(verify, worker['outputs'])}
        screen = json.loads(outputs['screening.jsonl'].read_text())
        if key != f"{screen['gpu']}-{screen['workload']}-{screen['n']}":
            raise ValueError(f'Mismatched cell {key}')
        required = expected_outcomes(screen)
        actual = set()
        for line in outputs['results.jsonl'].read_text().splitlines():
            row = json.loads(line)
            design, view, seed = (row[k] for k in ('design_id', 'view', 'seed'))
            identity = design, view, seed, row['candidate_origin']
            if identity in actual or identity not in required:
                raise ValueError(f'Duplicate or unexpected trajectory {key}: {identity}')
            actual.add(identity)
            status = row['status']
            if status == 'simulated_projection':
                for t in row['templates']:
                    path = worker_file.parent/'runs'/t['file']
                    templates[path] = None
            elif status != 'conditional_service_bound_excluded' or row['numerically_simulated_throughput'] is not None:
                raise ValueError(f'Invalid outcome {key}: {status}')
            integrity[status] += 1
            groups[row['gpu'], row['workload'], design, view].append(row)
        if required != actual or len(actual) != worker['result_rows']:
            raise ValueError(f'Missing trajectories {key}')
    for path in templates:
        requests, events = audit_template(path)
        templates[path] = fingerprint(path)
        integrity['replayed_requests'] += requests
        integrity['events'] += events
    integrity.update(cells=len(expected), replayed_templates=len(templates))
    return groups, dict(integrity), list(templates.values())


def paired_designs(groups):
    for (gpu, workload, design_id, view), rows in groups.items():
        if len(rows) != 2 or {r['seed'] for r in rows} != {17, 29}:
            raise ValueError('Missing paired workload windows')
        design = rows[0]['design']
        layout = design['layout']
        installed = layout['installed_gpus']
        stacks = installed*layout['hbf_stacks']
        metrics = {}
        for slo in ('responsive', 'balanced', 'relaxed', 'offline'):
            feasible = all(r['status']=='simulated_projection' and (
                r['offline_deadline_pass'] if slo == 'offline' else r['metrics'][slo]['slo_pass']) for r in rows)
            rates = [(r['output_tokens']/r['elapsed_s'] if slo == 'offline'
                      else r['metrics'][slo]['good_output_tokens_per_second']) for r in rows] if feasible else []
            metrics[slo] = dict(feasible=feasible, good_tokens_per_second=min(rates, default=None))
        yield dict(model=rows[0]['model'], gpu=gpu, workload=workload, design_id=design_id,
                   view=view, design=design, n=installed, architecture=layout['architecture'],
                   comparator=design['comparator'], hbf_TB=stacks*512*2**30/1e12,
                   installed_stacks=stacks, metrics=metrics)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--matrix', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    matrix = a.matrix.resolve()
    groups, integrity, templates = collect(matrix)
    designs = list(paired_designs(groups))
    minimum = adoption.minimum_frontiers(designs)
    with patch.object(adoption, 'catalog', lambda: [deployment_model()]):
        scale = adoption.scale_frontiers(designs)
    results = dict(status='completed', timing_evidence='muse_operator_projection', validated=False,
                   integrity=integrity, matrix=fingerprint(matrix/'manifest.json'),
                   templates=templates, fixed_demand_frontiers=minimum, scale_frontiers=scale,
                   price_curves=adoption.price_curves(designs, minimum), paired_designs=designs)
    with a.out.open('x') as stream:
        json.dump(results, stream, indent=2)
        stream.write('\n')
    print(json.dumps(integrity, indent=2))
    print(Counter(r['classification'] for r in minimum))


if __name__ == '__main__':
    main()
