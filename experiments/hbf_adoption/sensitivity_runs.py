"""Frozen physical/discrepancy corners, reselecting the central candidate pool."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
from dataclasses import asdict,replace
import json
from pathlib import Path
import shutil
import sys
import time

from run_validation import ROOT,HERE,fingerprint,execute
from model_catalog import catalog
from projection_timing import Layout
from projection_engine import Design
from boundary_replay import BoundarySimulator as Simulator,affected
import projection_sweep as sweep

CASES=[('h200','llama31_405b','short_chat'),('h200','kimi_k2','short_chat'),
       ('rtx4090','qwen3_32b','short_chat'),('rtx4090','qwen3_235b_a22b','short_chat'),
       ('rtx4090','qwen3_8b','long_rag'),('h200','qwen3_8b','paused_personal')]
SCENARIOS=['link_64GBs','link_256GBs','hbf_adverse','conventional_fast','moe_skewed',
           'program_slow','program_fast','thermal_10W','combined_adverse']
ACTIVE={}


class ScenarioSimulator(Simulator):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        if self.layout.hbf_stacks:
            self.device.read_Bps*=ACTIVE.get('read_factor',1.)
            if ACTIVE.get('program_Bps_per_stack') is not None:
                power=max(0,self.d.thermal_watts-2)*self.layout.hbf_stacks
                self.device.write_Bps=min(ACTIVE['program_Bps_per_stack']*self.layout.hbf_stacks,
                    power/(8*150e-12),self.d.hbf_link_Bps)

    def run(self,**kwargs):
        result=super().run(**kwargs)
        result['scenario_settings']=dict(ACTIVE)
        result['actual_hbf_read_Bps']=self.device.read_Bps
        result['actual_hbf_program_Bps']=self.device.write_Bps
        return result


def variant(d,name,family):
    global ACTIVE
    ACTIVE={};hbf=d.layout.hbf_stacks>0
    if name in ('link_64GBs','link_256GBs') and hbf:
        d=replace(d,hbf_link_Bps=64e9 if name=='link_64GBs' else 256e9)
    if name in ('hbf_adverse','combined_adverse') and hbf:
        ACTIVE['read_factor']=.75;d=replace(d,gpu_scale=1.5,cold_compute_scale=2.)
    if name in ('conventional_fast','combined_adverse') and not hbf:
        d=replace(d,io_scale=.5,gpu_scale=.75)
    if name=='combined_adverse' and hbf:d=replace(d,hbf_link_Bps=256e9)
    if name=='moe_skewed' and family=='moe':d=replace(d,route_mode='skewed')
    if name=='program_slow' and hbf:
        ACTIVE['program_Bps_per_stack']=2e9;d=replace(d,write_amplification=4.)
    if name=='program_fast' and hbf:
        ACTIVE['program_Bps_per_stack']=15e9;d=replace(d,write_amplification=1.)
    if name=='thermal_10W' and hbf:d=replace(d,thermal_watts=10.)
    return d,dict(ACTIVE)


def wait_for_job(path):
    start=time.time()
    while time.time()-start<12*3600:
        if path.exists():
            try:record=json.loads(path.read_text())
            except json.JSONDecodeError:time.sleep(10);continue
            if record['status']=='completed':return record
            if record['status'] in ('completed_with_failures','superseded_partial'):
                raise RuntimeError(f'central job not eligible: {path}')
        time.sleep(10)
    raise TimeoutError('central job did not finish')


def case(args):
    central=args.input.resolve()/f'{args.model}-{args.gpu}'
    wait_for_job(central/'manifest.json')
    expanded=args.expansion.resolve()/central.name
    wait_for_job(expanded/'manifest.json')
    source=central/'results.jsonl';source_fp=fingerprint(source)
    expansion_source=fingerprint(expanded/'results.jsonl')
    rows=[];row_sources={};row_directories={}
    for directory,record in ((central,source_fp),(expanded,expansion_source)):
        for line in (directory/'results.jsonl').read_text().splitlines():
            row=json.loads(line)
            if row['workload']!=args.workload or row['view']!='fixed_users' or row['seed']!=17:continue
            if row['design_id'] in row_sources:raise ValueError('duplicate original/expanded candidate')
            rows.append(row);row_sources[row['design_id']]=record;row_directories[row['design_id']]=directory
    if not rows:raise ValueError('central candidate pool is empty')
    boundary_cold={old['design_id']:(row_directories[old['design_id']]==central and affected(central,old)) for old in rows}
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=False);(out/'runs').mkdir()
    shutil.copy2(__file__,out/'runner.py')
    model=next(m for m in catalog() if m['id']==args.model)
    sweep.Simulator=ScenarioSimulator
    manifest=dict(schema='hbf-adoption.sensitivity-case.v1',status='running',model=args.model,gpu=args.gpu,
        workload=args.workload,source=source_fp,expansion_source=expansion_source,protocol=fingerprint(HERE/'SENSITIVITY_PROTOCOL.md'),
        search_protocol=fingerprint(HERE/'FIXED_DEMAND_SEARCH_AUDIT.md'),
        source_inputs=[fingerprint(HERE/name) for name in ('projection_engine.py','projection_sweep.py','projection_timing.py','projection_resources.py','projection_native.py','projection_workloads.py','boundary_replay.py')],
        scenarios=SCENARIOS,seed=17,scope='Reoptimization within the central screened pool; finite scenario stability, not a confidence interval.')
    failures=0;count=0
    with (out/'results.jsonl').open('w') as output:
        for scenario in SCENARIOS:
            runner=sweep.Runner(out)
            for old in rows:
                payload=dict(old['design']);layout=Layout(**payload.pop('layout'));original=Design(layout,**payload)
                d,settings=variant(original,scenario,model['family'])
                inactive_program=scenario in ('program_slow','program_fast') and not d.prefix
                unchanged=((d==original and not settings) or inactive_program) and not boundary_cold[old['design_id']]
                try:
                    if unchanged:
                        result=dict(old,reused_central_source=row_sources[old['design_id']],
                            reuse_reason='program mechanism inactive in this read-only workload' if inactive_program else 'scenario leaves this comparator unchanged')
                        result['design']=asdict(d);result['design_id']=sweep.design_key(d)
                    else:result=runner.execute(model,d,args.workload,17)
                    result.update(scenario=scenario,scenario_settings=settings)
                except Exception as e:
                    failures+=1;result=dict(status='failed',scenario=scenario,error=f'{type(e).__name__}: {e}',
                        design=asdict(d),model=args.model,gpu=args.gpu,workload=args.workload)
                output.write(json.dumps(result,separators=(',',':'))+'\n');output.flush();count+=1
            print(args.model,args.gpu,args.workload,scenario,'rows',count,'failed',failures,flush=True)
    manifest.update(status='completed' if failures==0 else 'completed_with_failures',result_rows=count,failures=failures,
                    outputs=[fingerprint(out/'results.jsonl')])
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    if failures:raise SystemExit(1)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,required=True)
    p.add_argument('--expansion',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--model');p.add_argument('--gpu')
    p.add_argument('--workload');p.add_argument('--workers',type=int,default=2)
    p.add_argument('--compiled',action='store_true')
    p.add_argument('--low-memory',action='store_true')
    args=p.parse_args()
    if args.model:case(args);return
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=False);shutil.copy2(__file__,out/'runner.py')
    manifest=dict(schema='hbf-adoption.sensitivity.v1',status='running',protocol=fingerprint(HERE/'SENSITIVITY_PROTOCOL.md'),jobs=[])
    def run(gpu,model,workload):
        name=f'{model}-{gpu}-{workload}'
        wrapper=HERE/('memory_entry.py' if args.low_memory else 'compiled_entry.py' if args.compiled else 'runtime_runner.py')
        argv=[sys.executable,str(wrapper),str(Path(__file__).resolve()),'--input',str(args.input.resolve()),'--out',str(out/name),
              '--model',model,'--gpu',gpu,'--workload',workload,'--expansion',str(args.expansion.resolve())]
        extra=[]
        if args.low_memory:
            from memory_entry import runtime_inputs
            extra=runtime_inputs()
        elif args.compiled:
            from compiled_entry import runtime_inputs
            extra=runtime_inputs()
        receipt=execute(argv,ROOT,out/(name+'-execution'),timeout=18*3600,
                        inputs=[Path(__file__).resolve(),HERE/'SENSITIVITY_PROTOCOL.md',HERE/'FIXED_DEMAND_SEARCH_AUDIT.md',HERE/'runtime_runner.py',HERE/'boundary_replay.py',*extra])
        return dict(id=name,status=receipt['status'],receipt=fingerprint(out/(name+'-execution')/'receipt.json'))
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(run,*item) for item in CASES]
        for future in as_completed(futures):
            manifest['jobs'].append(future.result())
            (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    manifest['status']='completed' if all(j['status']=='pass' for j in manifest['jobs']) else 'completed_with_failures'
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    if manifest['status']!='completed':raise SystemExit(1)


if __name__=='__main__':main()
