"""Run Muse Spark through the study's matched and fixed-demand screens.

This entry point binds the Muse-specific timing and byte functions only within
its own process. The archived eight-model runner and input hashes stay intact.
Run one GPU/workload job per directory so finished jobs can be retained.
"""
import argparse
from contextlib import ExitStack, contextmanager
from dataclasses import asdict
from functools import lru_cache
import importlib.metadata
import json
from pathlib import Path
import time
from unittest.mock import patch
import weakref

from boundary_replay import BoundarySimulator
import fixed_demand_expansion as expansion
from model_catalog import kv_bytes as standard_kv_bytes
from muse_spark_timing import (MuseTiming, SYSTEMS, deployment_model, layer_kv_bytes,
                               parameters_for_rank, reservation_bytes)
import projection_engine
from projection_timing import GPUS, layer_parameters as standard_layer_parameters
import projection_sweep as sweep
from run_validation import HERE, ROOT, fingerprint


def layer_parameters(model, layer, tp):
    if 'full_layers' in model:
        return parameters_for_rank(model, tp)
    return standard_layer_parameters(model, layer, tp)


def kv_bytes(model, tokens, tp=1, pp=1):
    if 'full_layers' in model:
        return reservation_bytes(model, tokens, tp, pp)
    return standard_kv_bytes(model, tokens, tp, pp)


class MuseSimulator(BoundarySimulator):
    def __init__(self, model, design, requests):
        model = dict(model, archive_prefix_kv=design.prefix)
        super().__init__(model, design, requests)
        owner = weakref.ref(self)

        @lru_cache(maxsize=2048)
        def route(layer, tokens):
            return super(MuseSimulator, owner())._route(layer, tokens)

        self._route = route

    def _memory_plan(self, requests):
        self.timings = MuseTiming(self.model, self.d.gpu, self.layout, gpu_scale=self.d.gpu_scale)
        super()._memory_plan(requests)

    def boundary_fraction(self, stage):
        plan = self.plans[stage]
        layers = len(self.layout.layers(self.model, stage))
        tensors = plan['weight_bytes']-2*layers*sum(parameters_for_rank(self.model, self.layout.tp))
        return plan['boundary_cold_bytes']/tensors

    def next_layer(self, batch):
        index = batch['layer_index']
        if index == len(batch['layer_list']):
            self.stage_finish(batch)
            return
        layer = batch['layer_list'][index]
        batch['layer_index'] += 1
        cold_attn, cold_ffn, _, _ = self.layer_plan[layer]
        routed, fraction = self._route(layer, batch['tokens'])
        attn_p, ffn_p, _ = parameters_for_rank(self.model, self.layout.tp)
        local_kv = layer_kv_bytes(self.model, layer, batch['context'], self.layout.tp)
        costs = batch['components']
        key = 'full_attention_layer' if layer in self.model['full_layers'] else 'sliding_attention_layer'
        attention = max(costs[key], (2*attn_p+local_kv*len(batch['requests']))/self.timings.near_Bps)
        ffn = max(costs['ffn_layer'], (2*ffn_p+routed)/self.timings.near_Bps)
        if batch['prefill']:
            attention = max(attention, self.timings.eager_layer_floor(True)-ffn)

        def after_attention():
            self.stream(batch, cold_ffn+routed, ffn, 1. if cold_ffn else fraction,
                        lambda: self.next_layer(batch), cpu_allowed=routed > 0)

        self.stream(batch, cold_attn, attention, 0., after_attention, cpu_allowed=False)


@contextmanager
def use_muse(model):
    """Scope adapters to one sweep. Each worker uses a separate process."""
    sweep.local_screen.cache_clear()
    sweep.model_map.cache_clear()
    expansion.local_proxy.cache_clear()
    with ExitStack() as stack:
        stack.enter_context(patch.object(projection_engine, 'kv_bytes', kv_bytes))
        stack.enter_context(patch.object(projection_engine, 'layer_parameters', layer_parameters))
        stack.enter_context(patch.object(sweep, 'Simulator', MuseSimulator))
        stack.enter_context(patch.object(sweep, 'catalog', lambda: [model]))
        stack.enter_context(patch.object(expansion, 'BoundarySimulator', MuseSimulator))
        stack.enter_context(patch.object(expansion, 'MODELS', {model['id']: model}))
        stack.enter_context(patch.object(expansion, 'kv_bytes', kv_bytes))
        stack.enter_context(patch.object(expansion, 'layer_parameters', layer_parameters))
        yield
    sweep.local_screen.cache_clear()
    sweep.model_map.cache_clear()
    expansion.local_proxy.cache_clear()


def evaluate_cell(model, runner, gpu, workload, count, views, seeds, *, secondary):
    central, scores = sweep.shortlist(model, count, gpu, workload)
    central_ids = {sweep.design_key(d) for d in central}
    additions, secondary_scores = [], []
    if secondary and 'fixed' in views:
        proxy_model = dict(model, archive_prefix_kv=workload in ('long_rag', 'paused_personal'))
        with patch.dict(expansion.MODELS, {model['id']: proxy_model}):
            extra, secondary_scores = expansion.selected(model, count, gpu, workload)
        additions = [d for d in extra if sweep.design_key(d) not in central_ids]
    selection = dict(model=model['id'], gpu=gpu, workload=workload, n=count,
                     candidates=scores, central=[asdict(d) for d in central],
                     secondary_candidates=secondary_scores, additions=[asdict(d) for d in additions])
    yield selection
    for d in central+additions:
        added = sweep.design_key(d) not in central_ids
        for view in ('fixed',) if added else views:
            if workload == 'tool_agent' and d.layout.width > 8 and view != 'fixed':
                continue
            for seed in seeds:
                row = runner.execute(model, d, workload, seed, None if view == 'fixed' else int(view))
                row['candidate_origin'] = 'fixed_four_user_secondary_screen' if added else 'matched_central_screen'
                row['timing_evidence'] = 'muse_operator_projection'
                yield row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--gpu', choices=GPUS, required=True)
    parser.add_argument('--workload', choices=sweep.WORKLOADS, required=True)
    parser.add_argument('--counts', type=int, choices=sweep.COUNTS, nargs='+', default=sweep.COUNTS)
    parser.add_argument('--views', nargs='+', choices=('fixed', '1', '4', '16'), default=['fixed', '1', '4', '16'])
    parser.add_argument('--seeds', nargs='+', type=int, choices=(17, 29), default=[17, 29])
    parser.add_argument('--event-budget', type=int, default=2_000_000_000)
    parser.add_argument('--central-only', action='store_true')
    args = parser.parse_args()
    if args.event_budget < 1:
        parser.error('--event-budget must be positive')
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out/'runs').mkdir()
    model = deployment_model()
    receipt = dict(status='running', arguments=vars(args)|{'out': str(out)}, cells_completed=0,
                   result_rows=0, started_unix=time.time(), model=model,
                   aisimulate_version=importlib.metadata.version('aisimulate'),
                   inputs=[fingerprint(path) for path in (
                       SYSTEMS/'manifest.json',
                       HERE/'results/device-service-20261104/manifest.json',
                       HERE/'results/projection-profiles-20261104/profiles.json',
                       HERE/'results/causal-workloads-v2-20261104/manifest.json',
                       HERE/'results/model-metadata-20261104/muse-architecture.json',
                       *sorted(HERE.glob('muse_spark_*.py')),
                       *sorted(HERE.glob('projection_*.py')),
                       HERE/'boundary_replay.py', HERE/'fixed_demand_expansion.py')])

    def save_receipt():
        temporary = out/'manifest.tmp'
        temporary.write_text(json.dumps(receipt, indent=2, default=str)+'\n')
        temporary.replace(out/'manifest.json')

    save_receipt()
    sweep.Events.run.__kwdefaults__['max_events'] = args.event_budget
    runner = sweep.Runner(out)
    # Disk-backed traces avoid retaining large request/token arrays across
    # layouts. Cached templates still use the same event replay and requests.
    from memory_cache import DiskPrograms, DiskRuns
    runner.cache = DiskRuns(out)
    runner.workload_cache = DiskPrograms(sweep.make)
    try:
        with use_muse(model), (out/'screening.jsonl').open('w') as screens, (out/'results.jsonl').open('w') as results:
            for count in args.counts:
                outcomes = evaluate_cell(model, runner, args.gpu, args.workload, count,
                                         args.views, args.seeds, secondary=not args.central_only)
                screens.write(json.dumps(next(outcomes), separators=(',', ':'))+'\n')
                screens.flush()
                for row in outcomes:
                    results.write(json.dumps(row, separators=(',', ':'))+'\n')
                    results.flush()
                    receipt['result_rows'] += 1
                receipt['cells_completed'] += 1
                save_receipt()
                print(f'{args.gpu} {args.workload} GPUs={count} cells={receipt["cells_completed"]} '
                      f'rows={receipt["result_rows"]} replayed_templates={runner.executions}', flush=True)
        receipt.update(status='completed', outputs=[fingerprint(out/f)
                       for f in ('screening.jsonl', 'results.jsonl')],
                       replayed_templates=runner.executions)
    except BaseException as exc:
        receipt.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        receipt['elapsed_seconds'] = time.time()-receipt['started_unix']
        save_receipt()


if __name__ == '__main__':
    main()
