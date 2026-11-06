import unittest
from functools import lru_cache
from model_catalog import catalog
from projection_engine import Simulator,Design
from projection_timing import Layout
from projection_workloads import make


class MemoizationContracts(unittest.TestCase):
    def test_complete_clocks_and_resources_are_identical(self):
        models={m['id']:m for m in catalog()}
        for model_id,workload,prefix in [('qwen3_30b_a3b','short_chat',False),
                ('qwen3_32b','short_chat',False),('qwen3_8b','paused_personal',True)]:
            d=Design(Layout(1,1,1,1,'h3_cascaded',8,4),'h200',batch_cap=4,prefix=prefix)
            records=make(workload,17,2)
            for r in records:r['output_tokens']=3
            original=Simulator(models[model_id],d,records).run(retain_tokens=True)
            sim=Simulator(models[model_id],d,records)
            sim._route=lru_cache(maxsize=8192)(sim._route)
            cached=sim.run(retain_tokens=True)
            self.assertEqual(original,cached)


if __name__=='__main__':unittest.main()
