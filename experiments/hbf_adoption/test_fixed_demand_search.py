import tempfile
import unittest
from pathlib import Path
from dataclasses import replace
from fixed_demand_expansion import selected,proxy,MODELS
from projection_engine import Design
from projection_timing import Layout
from projection_sweep import Runner


class FixedDemandSearchContracts(unittest.TestCase):
    def test_screen_recovers_the_resident_four_user_batch(self):
        model=MODELS['llama31_405b'];chosen,_=selected(model,8,'h200','short_chat')
        d=next(d for d in chosen if d.comparator=='resident' and d.layout.tp==8 and d.layout.pp==1 and d.batch_cap==4)
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);(out/'runs').mkdir()
            row=Runner(out).execute(model,d,'short_chat',17)
        self.assertEqual(row['status'],'simulated_projection')
        self.assertTrue(row['metrics']['responsive']['slo_pass'])

    def test_configured_workspace_is_not_reduced_to_available_users(self):
        model=MODELS['qwen3_8b']
        d=Design(Layout(1,1,1,1,'h3_cascaded',8,4),'rtx4090',batch_cap=4,chunk=16384,prefix=True)
        four=proxy(model,d,'long_rag');sixteen=proxy(model,replace(d,batch_cap=16),'long_rag')
        self.assertEqual(four['effective_batch'],4)
        self.assertEqual(sixteen['effective_batch'],4)
        self.assertGreaterEqual(sixteen['ttft_s'],four['ttft_s'])


if __name__=='__main__':unittest.main()
