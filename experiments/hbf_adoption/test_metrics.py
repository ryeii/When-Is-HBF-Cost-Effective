"""Scientific contract tests: censoring, joint SLOs, tails and design selection."""
import unittest

from evidence import compare, quantile
from metrics import SLOS, assess_requests, classify_adoption, node_occupancy, price_headroom


class ScientificContracts(unittest.TestCase):
    def test_aggregate_agreement_does_not_imply_request_agreement(self):
        result = compare([1, 9], [9, 1])
        self.assertEqual(result['aggregate_mean_bias_percent'], 0)
        self.assertGreater(result['mean_absolute_relative_error_percent'], 400)

    def test_tail_is_not_average(self):
        # One 300 ms stall is hidden by a 16 ms mean over 100 token gaps.
        gaps = [.01] * 97 + [.3] * 3
        tokens = [0.1]
        for gap in gaps:
            tokens.append(tokens[-1] + gap)
        result = assess_requests([dict(id='stall', arrival_s=0, expected_output_tokens=101,
                                       token_times_s=tokens)], SLOS['responsive'], 0, 10)
        self.assertLess(result['requests'][0]['mean_itl_s'], .05)
        self.assertGreater(result['requests'][0]['p99_itl_s'], .1)
        self.assertFalse(result['slo_pass'])

    def test_failures_and_censoring_remain_in_denominator(self):
        requests = [dict(id=str(i), arrival_s=0, expected_output_tokens=2,
                         token_times_s=[.1, .12]) for i in range(18)]
        requests += [dict(id='failed', arrival_s=0, expected_output_tokens=2, token_times_s=[]),
                     dict(id='censored', arrival_s=0, expected_output_tokens=2, token_times_s=[.1, 11])]
        result = assess_requests(requests, SLOS['responsive'], 0, 10)
        self.assertEqual(result['joint_success_fraction'], .9)
        self.assertEqual(result['completed_requests'], 18)
        self.assertEqual(result['good_output_tokens_per_second'], 3.6)
        self.assertFalse(result['slo_pass'])

    def test_joint_pass_cannot_use_separate_marginals(self):
        requests = [dict(id=str(i), arrival_s=0, expected_output_tokens=2,
                         token_times_s=[.1, .12]) for i in range(18)]
        requests += [dict(id='late-first', arrival_s=0, expected_output_tokens=2, token_times_s=[3, 3.01]),
                     dict(id='slow-decode', arrival_s=0, expected_output_tokens=2, token_times_s=[.1, .3])]
        self.assertFalse(assess_requests(requests, SLOS['responsive'], 0, 10)['slo_pass'])

    def test_one_token_has_no_fabricated_itl(self):
        result = assess_requests([dict(id='one', arrival_s=0, expected_output_tokens=1,
                                      token_times_s=[.1])], SLOS['responsive'], 0, 1)
        self.assertTrue(result['slo_pass'])
        self.assertIsNone(result['requests'][0]['p99_itl_s'])

    def test_reject_future_or_reversed_clock(self):
        with self.assertRaises(ValueError):
            assess_requests([dict(id='bad', arrival_s=1, expected_output_tokens=2,
                                  token_times_s=[2, .5])], SLOS['balanced'], 0, 5)

    def test_unrun_is_never_infeasible_or_enabled(self):
        self.assertEqual(classify_adoption(None, {'total_cost': 1}, search_complete=False,
                                          validation_passed=True)['classification'], 'unresolved')
        result = classify_adoption(None, {'total_cost': 1}, search_complete=True, validation_passed=True)
        self.assertEqual(result['classification'], 'enabled_within_search_domain')
        self.assertIsNone(result['saving_fraction'])

    def test_hundred_gpu_topology_does_not_invent_tp100(self):
        self.assertEqual(node_occupancy(100), [8]*12 + [4])
        self.assertEqual(node_occupancy(72), [8]*9)
        self.assertEqual(sum(node_occupancy(1)), 1)

    def test_reoptimization_changes_price_boundary(self):
        designs = [dict(id='large', cost_excluding_hbf=2, hbf_capacity_TB=8, hbf_stacks=8,
                        validated=True, slo_pass=True),
                   dict(id='small', cost_excluding_hbf=6, hbf_capacity_TB=1, hbf_stacks=1,
                        validated=True, slo_pass=True)]
        result = price_headroom(10, designs)
        self.assertEqual(result['design'], 'small')
        self.assertEqual(result['maximum_affordable_price_per_TB'], 4)
        designs[1]['validated'] = False
        with self.assertRaises(ValueError):
            price_headroom(10, designs)

    def test_negative_headroom_is_not_clamped_to_fake_break_even(self):
        result = price_headroom(10, [dict(id='too-expensive', cost_excluding_hbf=12,
                                         hbf_capacity_TB=2, hbf_stacks=1, validated=True, slo_pass=True)])
        self.assertEqual(result['maximum_affordable_price_per_TB'], -1)

    def test_percentile_method(self):
        self.assertAlmostEqual(quantile([0, 100], .99), 99)


if __name__ == '__main__':
    unittest.main()
