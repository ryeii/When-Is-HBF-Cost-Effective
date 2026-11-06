"""Study decision rules, independent of any simulator's headline summaries.

All clocks are seconds in one domain. Incomplete requests stay in the offered
population. Tail ITL requires actual token completion events, not average TPOT.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

from evidence import quantile


@dataclass(frozen=True)
class SLO:
    name: str
    ttft_seconds: float
    mean_itl_seconds: float
    p99_itl_seconds: float
    required_joint_success: float = .95

    def __post_init__(self):
        if (not self.name or any(not math.isfinite(x) or x <= 0 for x in
                (self.ttft_seconds, self.mean_itl_seconds, self.p99_itl_seconds))
                or not 0 < self.required_joint_success <= 1):
            raise ValueError('invalid service target')


SLOS = {
    'responsive': SLO('responsive', 2, .05, .1),
    'balanced': SLO('balanced', 5, .1, .2),
    'relaxed': SLO('relaxed', 10, .2, .4),
}


def assess_requests(requests: Iterable[dict], slo: SLO, window_start: float,
                    window_end: float) -> dict:
    """Score the declared finite cohort; drain time belongs in the window.

    Each record has id, arrival_s, expected_output_tokens, token_times_s. A
    failed/censored request supplies an empty/partial token list. No warmup
    request is silently dropped. The caller must declare any warmup cohort.
    """
    if not all(math.isfinite(v) for v in (window_start, window_end)) or window_end <= window_start:
        raise ValueError('invalid observation window')
    seen, results, output_tokens, good_output_tokens = set(), [], 0, 0
    for request in requests:
        rid = request['id']
        if rid in seen:
            raise ValueError(f'duplicate request {rid}')
        seen.add(rid)
        expected = request['expected_output_tokens']
        if isinstance(expected, bool) or not isinstance(expected, int) or expected < 1:
            raise ValueError('positive integer output target required')
        arrival = request['arrival_s']
        tokens = request['token_times_s']
        if not math.isfinite(arrival) or not window_start <= arrival <= window_end:
            raise ValueError('offered request outside declared window')
        if len(tokens) > expected:
            raise ValueError('more completions than requested tokens')
        if any(not math.isfinite(v) or v < arrival for v in tokens):
            raise ValueError('invalid token completion')
        if any(b < a for a, b in zip(tokens, tokens[1:])):
            raise ValueError('token completions are not causal')
        observed = [v for v in tokens if v <= window_end]
        output_tokens += len(observed)
        complete = len(observed) == expected
        itls = [b - a for a, b in zip(observed, observed[1:])]
        ttft = observed[0] - arrival if observed else None
        mean_itl = sum(itls) / len(itls) if itls else None
        p99 = quantile(itls, .99) if itls else None
        # A one-token request has no inter-token interval constraint. It still
        # needs to finish and pass TTFT. Report absent ITL as null, never zero.
        passed = bool(complete and ttft <= slo.ttft_seconds
                      and (not itls or (mean_itl <= slo.mean_itl_seconds
                                        and p99 <= slo.p99_itl_seconds)))
        if passed:
            good_output_tokens += len(observed)
        results.append({'id': rid, 'complete': complete, 'joint_pass': passed,
                        'ttft_s': ttft, 'mean_itl_s': mean_itl, 'p99_itl_s': p99})
    if not results:
        raise ValueError('empty offered cohort')
    successes = sum(r['joint_pass'] for r in results)
    duration = window_end - window_start
    fraction = successes / len(results)
    # Wilson interval is descriptive. Preregistered pass rule is empirical 95%.
    z, n = 1.959963984540054, len(results)
    center = (fraction + z*z/(2*n)) / (1+z*z/n)
    half = z*math.sqrt(fraction*(1-fraction)/n+z*z/(4*n*n))/(1+z*z/n)
    return {'offered_requests': n, 'completed_requests': sum(r['complete'] for r in results),
            'joint_successes': successes, 'joint_success_fraction': fraction,
            'joint_success_wilson95': [center-half, min(1, center+half)],
            'slo_pass': fraction >= slo.required_joint_success,
            'output_tokens_per_second': output_tokens / duration,
            'good_output_tokens_per_second': good_output_tokens / duration,
            'good_requests_per_second': successes / duration,
            'window_seconds': duration, 'requests': results}


def classify_adoption(baseline: dict | None, hbf: dict | None, *,
                      search_complete: bool, validation_passed: bool) -> dict:
    """Missing/failed execution is unresolved, distinct from proven infeasibility.

    Inputs represent minimum total installed cost at equal model, workload,
    precision and SLO. None means no feasible candidate found in that class.
    """
    if not search_complete or not validation_passed:
        return {'classification': 'unresolved', 'saving_fraction': None}
    for candidate in (baseline, hbf):
        if candidate is not None and (not math.isfinite(candidate['total_cost']) or candidate['total_cost'] <= 0):
            raise ValueError('cost must be finite and positive')
    if hbf is None:
        return {'classification': 'neither_feasible' if baseline is None else 'baseline_only', 'saving_fraction': None}
    if baseline is None:
        return {'classification': 'enabled_within_search_domain', 'saving_fraction': None}
    savings = 1 - hbf['total_cost'] / baseline['total_cost']
    return {'classification': 'cost_effective' if savings > 0 else 'no_cost_advantage',
            'saving_fraction': max(0, savings), 'signed_cost_reduction_fraction': savings}


def price_headroom(baseline_cost: float, candidates: Iterable[dict],
                   integration_premium_per_stack: float = 0) -> dict:
    """Maximum affordable flash price after reselecting the feasible design.

    Costs use one declared currency/scenario unit; capacity is decimal TB.
    Only SLO-feasible, validated designs may be passed by the caller. Negative
    headroom is retained: even zero flash price cannot rescue such a design.
    """
    if (not math.isfinite(baseline_cost) or baseline_cost <= 0
            or not math.isfinite(integration_premium_per_stack) or integration_premium_per_stack < 0):
        raise ValueError('invalid economic input')
    scored = []
    for candidate in candidates:
        c, tb, stacks = (candidate[k] for k in ('cost_excluding_hbf', 'hbf_capacity_TB', 'hbf_stacks'))
        if (not all(math.isfinite(v) for v in (c, tb, stacks)) or c < 0 or tb <= 0
                or isinstance(stacks, bool) or int(stacks) != stacks or stacks <= 0):
            raise ValueError('invalid installed design cost/capacity')
        if candidate.get('validated') is not True or candidate.get('slo_pass') is not True:
            raise ValueError('unvalidated or SLO-infeasible candidate')
        scored.append({'id': candidate['id'], 'price_per_TB':
                       (baseline_cost-c-integration_premium_per_stack*stacks)/tb})
    best = max(scored, key=lambda item: (item['price_per_TB'], item['id'])) if scored else None
    return {'maximum_affordable_price_per_TB': None if best is None else best['price_per_TB'],
            'design': None if best is None else best['id'], 'candidates': scored}


def node_occupancy(gpus: int, gpus_per_node: int = 8) -> list[int]:
    if any(isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in (gpus, gpus_per_node)):
        raise ValueError('positive GPU/node counts required')
    full, remainder = divmod(gpus, gpus_per_node)
    return [gpus_per_node] * full + ([remainder] if remainder else [])
