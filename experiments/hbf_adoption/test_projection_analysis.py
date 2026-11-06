"""Economic decision counterexamples for the executed projection analysis."""
import unittest

from analyze_deployments import minimum_frontiers, price_curves, scale_frontiers


def candidate(identifier, architecture, n, stacks, rate):
    # Two same-count flash layouts can pass the same target while using
    # different installed capacity. The faster witness need not set its price.
    return dict(model='fixture', gpu='h200', workload='short_chat',
        design_id=identifier, view='fixed_users', n=n, architecture=architecture,
        comparator='resident' if architecture == 'baseline' else 'gpu_staging',
        hbf_TB=n*stacks*512*2**30/1e12, installed_stacks=n*stacks,
        metrics={name: dict(feasible=True, good_tokens_per_second=rate)
                 for name in ('responsive', 'balanced', 'relaxed')})


class ProjectionEconomicDecisions(unittest.TestCase):
    def test_absent_matched_candidates_remain_explicitly_missing(self):
        rows=scale_frontiers([])
        self.assertEqual(len(rows),5616)
        self.assertTrue(all(r['tested_useful_good_tokens_per_second'] is None for r in rows))
        self.assertTrue(all(r['design_id'] is None for r in rows))

    def test_faster_minimum_gpu_witness_does_not_set_the_price_ceiling(self):
        pool=[candidate('base', 'baseline', 8, 0, 50),
              candidate('fast', 'h3_cascaded', 2, 4, 100),
              candidate('compact', 'h3_cascaded', 2, 2, 90)]
        rows=minimum_frontiers(pool)
        row=next(r for r in rows if r['architecture']=='h3_cascaded' and r['slo']=='balanced')
        self.assertEqual(row['hbf_design'], 'fast')
        self.assertEqual(row['ceiling_design'], 'compact')
        self.assertAlmostEqual(row['maximum_net_premium_gpu_units_per_TB'],
                               6/(4*512*2**30/1e12))
        point=next(r for r in price_curves(pool, rows) if r['architecture']=='h3_cascaded'
                   and r['slo']=='balanced' and r['premium_gpu_units_per_stack']==.5)
        self.assertEqual(point['selected_design'], 'compact')
        self.assertEqual(point['hbf_cost_gpu_units'], 4)

    def test_missing_reference_cannot_become_infinite_savings(self):
        row=next(r for r in minimum_frontiers([candidate('hbf', 'shared_site', 1, 2, 10)])
                 if r['architecture']=='shared_site' and r['slo']=='balanced')
        self.assertEqual(row['classification'], 'reference_not_found_in_pool')
        self.assertIsNone(row['affordable_total_premium_gpu_units'])
        self.assertIsNone(row['maximum_net_premium_gpu_units_per_TB'])

    def test_more_gpus_preserves_negative_headroom(self):
        row=next(r for r in minimum_frontiers([
            candidate('base', 'baseline', 2, 0, 10),
            candidate('hbf', 'shared_site', 4, 2, 20)])
                 if r['architecture']=='shared_site' and r['slo']=='balanced')
        self.assertEqual(row['classification'], 'no_gpu_acquisition_headroom')
        self.assertEqual(row['affordable_total_premium_gpu_units'], -2)
        self.assertLess(row['maximum_net_premium_gpu_units_per_TB'], 0)


if __name__=='__main__':
    unittest.main()
