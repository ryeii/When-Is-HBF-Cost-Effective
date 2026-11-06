import unittest
from functools import lru_cache
from boundary_replay import BoundarySimulator
from projection_engine import Simulator,Design
from projection_timing import Layout
from projection_workloads import make
from model_catalog import catalog


class BoundaryContracts(unittest.TestCase):
    def test_resident_boundary_has_identical_complete_clocks(self):
        m=next(m for m in catalog() if m['id']=='qwen3_8b')
        d=Design(Layout(1,1,1,1,'h3_cascaded',8,4),'h200',batch_cap=4)
        records=make('short_chat',17,2)
        for r in records:r['output_tokens']=2
        original=Simulator(m,d,records).run(retain_tokens=True)
        corrected=BoundarySimulator(m,d,records).run(retain_tokens=True)
        self.assertEqual(original,corrected)

    def test_logits_fetch_follows_last_layer_and_bytes_conserve(self):
        m=next(m for m in catalog() if m['id']=='qwen3_8b')
        d=Design(Layout(1,1,1,1,'h3_cascaded',8,4),'h200')
        records=make('short_chat',17,1,rounds=1);records[0]['output_tokens']=1
        class Capture(BoundarySimulator):
            def stream(self,batch,nbytes,*args,**kwargs):
                if nbytes:self.observed.append((self.events.now,batch['layer_index'],nbytes))
                return super().stream(batch,nbytes,*args,**kwargs)
        sim=Capture(m,d,records);sim.observed=[]
        # Force a small boundary eviction in an otherwise resident test model.
        sim.plans[0]['boundary_cold_bytes']=64*2**20
        result=sim.run(retain_tokens=True)
        self.assertEqual(len(sim.observed),2)
        self.assertEqual([r[2] for r in sim.observed],[32*2**20,32*2**20])
        self.assertEqual([r[1] for r in sim.observed],[0,m['layers']])
        self.assertGreater(sim.observed[1][0],sim.observed[0][0])
        self.assertEqual(result['traffic']['weight_read_bytes'],64*2**20)


if __name__=='__main__':unittest.main()
