import math
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from model_catalog import catalog
from projection_timing import Layout,layer_parameters,GiB
from projection_engine import Simulator,Design,CapacityError
from projection_workloads import make,tool_shapes
from projection_sweep import Runner


class ProjectionContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.models={m['id']:m for m in catalog()}

    def design(self,*,n=1,tp=1,pp=1,architecture='h3_cascaded',prefix=False):
        return Design(Layout(n,tp,pp,n//(tp*pp),architecture,8,4 if architecture!='baseline' else 0),
                      'h200',prefix=prefix,batch_cap=4,host_gib=4096)

    def test_physical_payload_conservation_across_pp(self):
        for model in self.models.values():
            for pp in (1,2,4,8):
                d=self.design(n=pp,pp=pp)
                from projection_native import NativeTiming
                t=NativeTiming(model,'h200',d.layout)
                self.assertAlmostEqual(sum(t.stage_weight_bytes(s) for s in range(pp)),
                                       model['bf16_weight_bytes'],delta=1)

    def test_tp_replication_cannot_create_a_smaller_total_payload(self):
        for model in self.models.values():
            single=sum(sum(layer_parameters(model,i,1)) for i in range(model['layers']))
            for tp in (2,4,8):
                split=tp*sum(sum(layer_parameters(model,i,tp)) for i in range(model['layers']))
                self.assertGreaterEqual(split,single)

    def test_one_hundred_gpu_node_mapping_has_a_real_partial_node(self):
        layout=Layout(100,1,1,100,'h3_cascaded',8,4)
        self.assertEqual(layout.nodes,13)
        self.assertEqual([layout.node(r,0) for r in range(96,100)],[12]*4)

    def test_native_mla_never_calls_gqa_attention_surrogate(self):
        m=self.models['deepseek_v3'];d=self.design()
        s=Simulator(m,d,make('short_chat',17,1))
        s.timings.attention_seconds=lambda *a:(_ for _ in ()).throw(AssertionError('GQA substituted'))
        records=make('short_chat',17,1,rounds=1)
        records[0]['output_tokens']=2
        s=Simulator(m,d,records);r=s.run(retain_tokens=True)
        self.assertEqual(len(r['requests'][0]['token_times_s']),2)
        self.assertGreater(r['traffic']['weight_read_bytes'],0)

    def test_successors_release_after_simulated_completion_and_gap(self):
        d=self.design();s=Simulator(self.models['qwen3_8b'],d,make('short_chat',17,2))
        result=s.run(retain_tokens=True)
        rows={r['id']:r for r in result['requests']}
        for r in rows.values():
            if r['predecessor']:
                previous=rows[r['predecessor']]
                self.assertAlmostEqual(r['arrival_s'],previous['token_times_s'][-1]+previous['successor_gap_s'])
        self.assertTrue(all(abs(used)<1 for used in s.kv_used))
        self.assertTrue(all(r['utilization']<=1.000001 for r in result['resources'].values()))

    def test_resident_control_does_not_fetch_weights(self):
        d=replace(self.design(architecture='baseline'),comparator='resident')
        result=Simulator(self.models['qwen3_8b'],d,make('short_chat',17,2)).run()
        self.assertEqual(result['traffic']['weight_read_bytes'],0)
        self.assertEqual(result['traffic']['gpu_staging_bytes'],0)

    def test_capacity_exclusion_is_explicit(self):
        d=replace(self.design(),gpu='rtx4090',layout=Layout(1,1,1,1,'shared_site',4,4))
        records=make('long_rag',17,1);records[0]['input_tokens']=32768
        with self.assertRaises(CapacityError):
            Simulator(self.models['llama31_405b'],d,records)

    def test_write_programming_is_charged_and_far_capacity_is_finite(self):
        d=self.design(prefix=True)
        records=make('paused_personal',17,2)
        s=Simulator(self.models['qwen3_8b'],d,records);result=s.run()
        traffic=result['traffic']
        self.assertGreater(traffic['prefix_write_logical_bytes'],0)
        self.assertGreaterEqual(traffic['flash_program_physical_bytes'],traffic['prefix_write_logical_bytes']*1.3)
        self.assertTrue(all(len(c.rows)<=c.capacity for c in s.far_caches))
        self.assertFalse(any(s.pending_prefix))
        self.assertFalse(any(any(busy) for busy in s.save_busy))

    def test_cross_node_pipeline_completes_with_nic_service(self):
        d=self.design(n=16,tp=8,pp=2)
        records=make('short_chat',29,4,rounds=1)
        for r in records:r['output_tokens']=4
        s=Simulator(self.models['llama31_70b'],d,records);result=s.run()
        self.assertEqual(result['completed_requests'],4)
        self.assertGreater(result['traffic']['network_bytes'],0)
        self.assertTrue(any(key.startswith('nic-out:') for key in result['resources']))
        self.assertEqual(s.inflight,[0])

    def test_two_node_group_composition_matches_explicit_shared_node_execution(self):
        # All records have equal shapes and no prefix caching. Each node's
        # full eight-GPU host/CPU contention is identical under this program.
        d=self.design(n=16)
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp);(out/'runs').mkdir()
            # Runner's receipts use workspace-relative fingerprints only in
            # main(); this local equivalence test needs no external writes.
            runner=Runner(out)
            composed=runner.execute(self.models['qwen3_8b'],d,'short_chat',17,1)
            explicit=Simulator(self.models['qwen3_8b'],d,make('short_chat',17,16)).run()
        self.assertAlmostEqual(composed['elapsed_s'],explicit['elapsed_s'],places=9)
        self.assertAlmostEqual(composed['metrics']['balanced']['output_tokens_per_second'],
                               explicit['summaries']['balanced']['output_tokens_per_second'],places=7)

    def test_public_tools_have_no_prefix_identity(self):
        self.assertEqual(len(tool_shapes()),8192)
        for r in make('tool_agent',17,4):
            self.assertFalse(r['input_blocks']);self.assertFalse(r['completed_blocks'])
            self.assertGreaterEqual(r['successor_gap_s'],0)

    def test_service_lower_bound_never_exceeds_observed_token_or_resource_clocks(self):
        for model_id in ('qwen3_8b','qwen3_30b_a3b','deepseek_v3','llama31_405b'):
            for arch,near,stacks,control in [('baseline',8,0,'gpu_staging'),('h3_cascaded',8,4,'gpu_staging'),
                                            ('baseline',8,0,'cpu_oracle')]:
                if control=='cpu_oracle' and self.models[model_id]['family']=='dense':continue
                d=Design(Layout(1,1,1,1,arch,near,stacks),'h200',control,host_gib=4096,
                         batch_cap=4,chunk=16384,prefix=False)
                records=make('short_chat',17,2,rounds=1)
                records=[dict(r,output_tokens=3) for r in records]
                s=Simulator(self.models[model_id],d,records);bound=s.service_bounds();result=s.run(retain_tokens=True)
                gaps=[b-a for r in result['requests'] for a,b in zip(r['token_times_s'],r['token_times_s'][1:])]
                self.assertLessEqual(bound['decode_interval_s'],min(gaps)+1e-7)
                self.assertLessEqual(bound['offline_makespan_s'],result['elapsed_s']+1e-7)

    def test_cpu_bound_still_transfers_shared_attention_weights_to_gpu(self):
        d=replace(self.design(),comparator='cpu_oracle',layout=Layout(1,1,1,1,'baseline'),host_gib=4096)
        s=Simulator(self.models['qwen3_30b_a3b'],d,make('short_chat',17,1))
        hops=s._source_hops(0,0,128*2**20,weight=True)
        self.assertTrue(any(h.resource.name.startswith('ingress:') and h.duration>0 for h in hops))

    def test_bandwidth_exclusion_has_no_fabricated_token_throughput(self):
        d=replace(self.design(),gpu='rtx4090')
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp);(out/'runs').mkdir()
            result=Runner(out).execute(self.models['llama31_405b'],d,'short_chat',17)
        self.assertEqual(result['status'],'conditional_service_bound_excluded')
        self.assertIsNone(result['metrics']);self.assertIsNone(result['numerically_simulated_throughput'])

    def test_optimistic_root_bound_dominates_explicit_completion_count(self):
        d=replace(self.design(),batch_cap=1,chunk=16384)
        s=Simulator(self.models['qwen3_32b'],d,make('short_chat',17,32,rounds=1))
        bound=s.service_bounds();result=s.run(retain_tokens=True)
        observed=sum(r['token_times_s'][-1]<=bound['root_response_deadline_s']+1e-7 for r in result['requests'])
        self.assertLess(bound['passing_roots_upper_bound'],32)
        self.assertLessEqual(observed,bound['passing_roots_upper_bound'])

    def test_overloaded_dense_case_is_certified_without_hitting_event_budget(self):
        d=Design(Layout(8,8,1,1,'h3_cascaded',8,4),'rtx4090',batch_cap=1,chunk=16384,prefix=False,host_gib=256)
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp);(out/'runs').mkdir();runner=Runner(out)
            result=runner.execute(self.models['llama31_405b'],d,'short_chat',17,16)
        self.assertEqual(result['status'],'conditional_service_bound_excluded')
        self.assertLess(result['root_pass_upper_bound'],result['required_root_passes'])
        self.assertEqual(runner.executions,0)

    def test_reuse_rejects_any_changed_event_physics(self):
        import json,shutil
        from run_validation import HERE
        from projection_sweep import reusable_rows
        with tempfile.TemporaryDirectory(dir=HERE/'results') as temp:
            base=Path(temp);old=base/'old';new=base/'new';(old/'sources').mkdir(parents=True);(new/'runs').mkdir(parents=True)
            (old/'manifest.json').write_text(json.dumps(dict(source_inputs=[])))
            (old/'results.jsonl').write_text('')
            for name in ('projection_resources.py','projection_timing.py','projection_native.py','projection_workloads.py',
                         'projection_engine.py','projection_sweep.py','model_catalog.py','metrics.py','evidence.py'):
                shutil.copy2(HERE/name,old/'sources'/name)
            rows,receipt=reusable_rows(old,new,[])
            self.assertEqual(rows,{})
            file=old/'sources/projection_engine.py';file.write_text(file.read_text().replace('3.5*GiB','3.0*GiB'))
            with self.assertRaisesRegex(ValueError,'event-model methods changed'):reusable_rows(old,new,[])


if __name__=='__main__':unittest.main()
