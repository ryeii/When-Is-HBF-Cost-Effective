import unittest

from model_catalog import catalog,kv_bytes


class ModelGeometry(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.models={row['id']:row for row in catalog()}

    def test_official_qwen8_has_thirty_six_layers(self):
        model=self.models['qwen3_8b']
        self.assertEqual(model['layers'],36)
        self.assertEqual(kv_bytes(model,1),147456)

    def test_mla_is_not_divided_by_tensor_parallel_width(self):
        model=self.models['deepseek_v3']
        self.assertEqual(kv_bytes(model,128,1),kv_bytes(model,128,8))
        self.assertLess(kv_bytes(model,128,8,4),kv_bytes(model,128,8,1))

    def test_gqa_head_replication_beyond_kv_head_count(self):
        model=self.models['qwen3_235b_a22b']
        self.assertEqual(model['kv_heads'],4)
        self.assertEqual(kv_bytes(model,128,4),kv_bytes(model,128,8))

    def test_moe_capacity_and_active_work_are_separate(self):
        for model in self.models.values():
            self.assertEqual(model['parameters'],model['shared_parameters']+model['routed_parameters'])
            if model['family']=='moe':
                self.assertLess(model['active_parameters'],model['parameters'])
                self.assertAlmostEqual(model['active_parameters'],model['shared_parameters']+
                    model['routed_parameters']*model['top_k']/model['experts'])

    def test_core_text_counts_match_reported_order_without_nominal_overrides(self):
        self.assertTrue(670e9<self.models['deepseek_v3']['parameters']<672e9)
        self.assertTrue(1020e9<self.models['kimi_k2']['parameters']<1030e9)
        self.assertTrue(405e9<self.models['llama31_405b']['parameters']<407e9)


if __name__=='__main__':unittest.main()
