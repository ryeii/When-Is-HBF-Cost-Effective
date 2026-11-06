"""Receipted finite-domain conditional adoption sweep (no hardware validation).

Independent, identically driven node groups can be replayed once and copied
exactly; each template still simulates all its GPUs and shared host/SSD/NIC
resources. Cross-node PP groups are never replaced by one-GPU multiplication.
Copied cohorts are not independent statistical observations.
"""
from __future__ import annotations
import argparse
import ast
from dataclasses import asdict,replace
from functools import lru_cache
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import time
import traceback

from run_validation import HERE,ROOT,fingerprint
from model_catalog import catalog
from projection_timing import Layout,GiB
from projection_engine import Simulator,Design,CapacityError
from projection_resources import Events
from projection_workloads import make,WORKLOADS,manifest as workload_manifest
from metrics import SLOS,assess_requests
from evidence import quantile

COUNTS=[1,2,4,8,16,32,64,72,100]


def layouts(n,architecture):
    memory=[(8,0)] if architecture=='baseline' else ([(6,2),(4,4)] if architecture=='shared_site' else [(8,2),(8,4)])
    for tp in (1,2,4,8):
        for pp in (1,2,4,8,16):
            width=tp*pp
            if width>n:continue
            for near,stacks in memory:
                yield Layout(n,tp,pp,n//width,architecture,near,stacks)


def candidates(n,gpu,workload):
    host=256 if gpu=='rtx4090' else 4096
    for architecture in ('baseline','shared_site','h3_cascaded'):
        for layout in layouts(n,architecture):
            controls=('resident','gpu_staging','cpu_oracle') if architecture=='baseline' else ('gpu_staging',)
            for comparator in controls:
                for batch in (1,4,16):
                    for chunk in (512,16384):
                        yield Design(layout,gpu,comparator,host_gib=host,batch_cap=batch,chunk=chunk,
                                     prefix=workload in ('long_rag','paused_personal'))


@lru_cache(maxsize=8)
def screen_records(workload):return make(workload,17,1)


def screen(model,design,workload):
    """Declared static ranking only; never an E2E or SLO result.

    Score capacity-legal candidates using native costs plus serial source and
    staging service. Final outcomes come from the causal event replay. Keep
    both the latency and throughput extrema in every comparator class.
    """
    if design.comparator=='cpu_oracle' and model['family']!='moe':
        raise CapacityError('CPU expert bound does not apply to dense layers')
    records=screen_records(workload)
    if workload=='tool_agent':
        # Capacity screening uses the entire frozen shape envelope, before
        # any hardware outcomes; a short sampled root cannot admit a design
        # unable to hold another member's active KV.
        records=records+[dict(records[0],id='capacity-envelope',input_tokens=33152,output_tokens=128)]
    sim=Simulator(model,design,records)
    l=design.layout;t=sim.timings
    context=max(r['input_tokens'] for r in records)
    batch=design.batch_cap
    decode,_=t.components(False,batch,1,context)
    prefill,_=t.components(True,batch,min(design.chunk,context),context)
    stages=[];prefill_stages=[]
    for stage in range(l.pp):
        node=l.node(0,stage);fraction=sim.host_fraction[node]
        node_groups=max(1,min(l.replicas,8//min(8,l.width)))
        def source(nbytes,tokens,cpu):
            if not nbytes:return 0.
            if l.hbf_stacks:return sim.device.seconds(nbytes)
            allbytes=nbytes*l.tp
            hostbw=100e9 if design.gpu=='rtx4090' else 460.8e9
            missing=allbytes*(1-fraction)
            seconds=missing/(design.drives*design.ssd_Bps_per_drive)+(allbytes+missing)/hostbw
            if not cpu:seconds+=nbytes/t.gpu['ingress_Bps']
            return seconds*node_groups
        cost=0.;pf=0.
        for layer in l.layers(model,stage):
            a,f,_,_=sim.layer_plan[layer]
            routed,cold_fraction=sim._route(layer,batch)
            full,_=sim._route(layer,batch*min(design.chunk,context))
            cpu=design.comparator=='cpu_oracle' and f==0
            cost+=source(a+f+routed,batch,cpu)+(a+f+routed)/t.near_Bps
            pf+=source(a+f+full,batch*min(design.chunk,context),cpu)+(a+f+full)/t.near_Bps
        layer_fraction=len(list(l.layers(model,stage)))/model['layers']
        resident=(decode['attention']+decode['ffn'])*layer_fraction
        resident_pf=(prefill['attention']+prefill['ffn'])*layer_fraction
        if design.comparator=='cpu_oracle':resident*=.5;resident_pf*=.5
        stages.append(cost+resident)
        prefill_stages.append(pf+max(resident_pf,t.eager_layer_floor(True)*len(list(l.layers(model,stage)))))
    latency=sum(stages)+t.cpu_seconds(False)
    ttft=sum(prefill_stages)*math.ceil(context/design.chunk)+t.cpu_seconds(True)
    throughput=l.replicas*batch/max(stages)
    return dict(latency_s=latency,ttft_s=ttft,throughput_score=throughput,
                memory_plans=sim.plans)


@lru_cache(maxsize=1)
def model_map():return {m['id']:m for m in catalog()}


@lru_cache(maxsize=20000)
def local_screen(model_id,design,workload):
    try:return screen(model_map()[model_id],design,workload),None
    except CapacityError as e:return None,str(e)


def shortlist(model,n,gpu,workload):
    groups={};all_rows=[]
    for design in candidates(n,gpu,workload):
        row=dict(design=asdict(design))
        try:
            width=design.layout.width
            local_replicas=min(design.layout.replicas,max(1,8//width))
            local_layout=replace(design.layout,installed_gpus=width*local_replicas,replicas=local_replicas)
            local=replace(design,layout=local_layout)
            cached,error=local_screen(model['id'],local,workload)
            if error:raise CapacityError(error)
            score=dict(cached,throughput_score=cached['throughput_score']*design.layout.replicas/local_replicas)
            row.update(status='ranked',**score)
            key=(design.layout.architecture,design.comparator)
            groups.setdefault(key,[]).append((design,score))
        except CapacityError as e:row.update(status='capacity_excluded',reason=str(e))
        all_rows.append(row)
    chosen=[]
    for key,rows in groups.items():
        # Two predeclared extrema, without selecting on replayed outcomes.
        low=min(rows,key=lambda x:(x[1]['latency_s']+x[1]['ttft_s']/64,x[0].layout.width))
        high=max(rows,key=lambda x:(x[1]['throughput_score'],-x[0].layout.width))
        middle=max(rows,key=lambda x:(x[1]['throughput_score']/max(1,x[1]['latency_s']/.1,x[1]['ttft_s']/5),
                                     -x[0].layout.width))
        for item in (low,high,middle):
            if item[0] not in chosen:chosen.append(item[0])
    return chosen,all_rows


def design_key(design):
    return hashlib.sha256(json.dumps(asdict(design),sort_keys=True).encode()).hexdigest()[:20]


class Runner:
    def __init__(self,out):
        self.out=out;self.cache={};self.workload_cache={};self.bound_cache={};self.executions=0

    def execute(self,model,design,workload,seed,users_per_gpu=None,fixed_users=4):
        l=design.layout
        if users_per_gpu is None:
            parts=[(design,1,fixed_users)]
        else:
            group=max(8,l.width)
            full=l.installed_gpus//group;partial=l.installed_gpus%group
            parts=[]
            if full:
                local=replace(l,installed_gpus=group,replicas=group//l.width)
                parts.append((replace(design,layout=local),full,int(users_per_gpu*local.used_gpus)))
            if l.width<=8 and partial>=l.width:
                local=replace(l,installed_gpus=partial,replicas=partial//l.width)
                parts.append((replace(design,layout=local),1,int(users_per_gpu*local.used_gpus)))
            if not parts:raise RuntimeError('empty independent-node decomposition')
            # Proportional demand is based on INSTALLED GPUs. Idle chips in a
            # uniform layout do not silently remove their share of demand.
            extra=int(users_per_gpu*(l.installed_gpus-l.used_gpus))
            copies_total=sum(p[1] for p in parts)
            average,remainder=divmod(extra,copies_total)
            balanced=[]
            for local,copies,roots in parts:
                raised=min(copies,remainder)
                if raised:balanced.append((local,raised,roots+average+1))
                if copies>raised:balanced.append((local,copies-raised,roots+average))
                remainder-=raised
            parts=balanced
            if sum(copies*roots for local,copies,roots in parts)!=int(users_per_gpu*l.installed_gpus):
                raise AssertionError('proportional demand lost installed-GPU users')
        prepared={};certificates=[]
        for local,copies,roots in parts:
            key=(model['id'],design_key(local),workload,seed,roots)
            workload_key=(workload,seed,roots)
            if workload_key not in self.workload_cache:self.workload_cache[workload_key]=make(workload,seed,roots)
            records=self.workload_cache[workload_key]
            if key not in self.bound_cache:
                sim=Simulator(model,local,records)
                self.bound_cache[key]=sim.service_bounds();prepared[key]=sim
            certificates.append(dict(bounds=self.bound_cache[key],copies=copies,requests=len(records),
                                     all_have_itl=all(r['output_tokens']>=2 for r in records)))
        population=sum(c['copies']*c['requests'] for c in certificates)
        excluded=sum(c['copies']*c['requests'] for c in certificates
                     if c['all_have_itl'] and c['bounds']['decode_interval_s']>.2+1e-9)
        interactive_excluded=workload!='offline_batch' and excluded/population>.05+1e-9
        roots=sum(c['copies']*c['bounds']['root_requests'] for c in certificates)
        root_upper=sum(c['copies']*c['bounds']['passing_roots_upper_bound'] for c in certificates)
        required_roots=max(0,roots-math.floor(.05*population+1e-8))
        root_excluded=workload!='offline_batch' and root_upper<required_roots
        offline_excluded=workload=='offline_batch' and any(c['bounds']['offline_makespan_s']>3600+1e-9 for c in certificates)
        if interactive_excluded or root_excluded or offline_excluded:
            return dict(status='conditional_service_bound_excluded',validated=False,model=model['id'],
                workload=workload,gpu=design.gpu,seed=seed,
                view='fixed_users' if users_per_gpu is None else f'users_per_gpu_{users_per_gpu}',
                design=asdict(design),design_id=design_key(design),metrics=None,requests=population,
                certificate=certificates,
                root_pass_upper_bound=root_upper,required_root_passes=required_roots,
                reason='mandatory resource work exceeds the offline deadline' if offline_excluded else
                       ('too few roots can finish by the necessary relaxed-SLO response deadline' if root_excluded else
                        'more than 5% of requests must exceed every interactive mean-ITL target'),
                numerically_simulated_throughput=None)
        runs=[]
        for local,copies,roots in parts:
            key=(model['id'],design_key(local),workload,seed,roots)
            if key not in self.cache:
                workload_key=(workload,seed,roots)
                if workload_key not in self.workload_cache:self.workload_cache[workload_key]=make(workload,seed,roots)
                records=self.workload_cache[workload_key]
                simulation=prepared.get(key) or Simulator(model,local,records)
                run=simulation.run(retain_tokens=True)
                bound=self.bound_cache[key]
                gaps=[b-a for r in run['requests'] for a,b in zip(r['token_times_s'],r['token_times_s'][1:])]
                if gaps and min(gaps)<bound['decode_interval_s']-1e-6:
                    raise AssertionError('decode interval violates the supposedly optimistic service bound')
                if run['elapsed_s']<bound['offline_makespan_s']-1e-6:
                    raise AssertionError('resource work bound exceeds actual elapsed time')
                observed_roots=sum(r['predecessor'] is None and r['token_times_s'][-1]<=bound['root_response_deadline_s']+1e-7
                                   for r in run['requests'])
                if observed_roots>bound['passing_roots_upper_bound']:
                    raise AssertionError('observed root completions violate optimistic root-cohort bound')
                run['service_bounds']=bound
                if any(r['utilization']>1.000001 for r in run['resources'].values()):
                    raise AssertionError('resource overbooked')
                name=f"run-{hashlib.sha256(repr(key).encode()).hexdigest()[:20]}.json.gz"
                with gzip.GzipFile(filename=str(self.out/'runs'/name),mode='wb',mtime=0) as stream:
                    stream.write(json.dumps(run,separators=(',',':')).encode())
                self.cache[key]=(run,name);self.executions+=1
            run,name=self.cache[key];runs.append((run,copies,name))
        # Replicate exact independent node/group executions. Each full template
        # models shared host bandwidth, copies, SSD queues and PP links.
        all_requests=[];traffic={key:0. for key in runs[0][0]['traffic']}
        elapsed=max(run['elapsed_s'] for run,copies,name in runs)
        for group,(run,copies,name) in enumerate(runs):
            for copy in range(copies):
                for r in run['requests']:
                    all_requests.append({**r,'id':f'g{group}-c{copy}-{r["id"]}'})
            for key,value in run['traffic'].items():traffic[key]+=value*copies
        metrics={}
        for name,slo in SLOS.items():
            scored=assess_requests(all_requests,slo,0,max(elapsed,1e-9));details=scored.pop('requests')
            for field in ('ttft_s','mean_itl_s','p99_itl_s'):
                values=[r[field] for r in details if r[field] is not None]
                scored[field+'_p50']=quantile(values,.5) if values else None
                scored[field+'_p99']=quantile(values,.99) if values else None
            metrics[name]=scored
        return dict(status='simulated_projection',validated=False,model=model['id'],workload=workload,
            gpu=design.gpu,seed=seed,view='fixed_users' if users_per_gpu is None else f'users_per_gpu_{users_per_gpu}',
            design=asdict(design),design_id=design_key(design),metrics=metrics,traffic=traffic,
            output_tokens=sum(r['expected_output_tokens'] for r in all_requests),
            elapsed_s=elapsed,requests=len(all_requests),
            offline_deadline_pass=all(run['offline_deadline_pass'] for run,copies,name in runs),
            templates=[dict(file=name,copies=copies,installed_gpus=run['design']['layout']['installed_gpus']) for run,copies,name in runs],
            installed_hbf_stacks=l.hbf_stacks*l.installed_gpus,
            installed_hbf_TB=l.hbf_stacks*l.installed_gpus*512*GiB/1e12,
            accelerator_thermal_budget_watts=l.installed_gpus*(sim_tdp(design.gpu)+l.hbf_stacks*design.thermal_watts),
            idle_installed_gpus=l.installed_gpus-l.used_gpus)


def sim_tdp(gpu):return 450 if gpu=='rtx4090' else 700


def reusable_rows(previous,out,current_inputs):
    """Reuse completed rows from an interrupted worker, with checked physics.

    Only service-bound code and orchestration may differ. Every event-model
    method and every timing/workload/resource primitive must be identical.
    Failed/truncated rows are retried, never converted into zero throughput.
    """
    if previous is None or not previous.exists():return {},None
    previous=previous.resolve();old=json.loads((previous/'manifest.json').read_text())
    if old['source_inputs']!=current_inputs:raise ValueError('reuse inputs differ')
    checked=[]
    for name in ('projection_resources.py','projection_timing.py','projection_native.py','projection_workloads.py'):
        source=previous/'sources'/name
        if source.read_bytes()!=(HERE/name).read_bytes():raise ValueError(f'reuse physics changed: {name}')
        checked.append(fingerprint(source))
    def normalized(path):
        tree=ast.parse(path.read_text())
        for node in tree.body:
            if isinstance(node,ast.ClassDef) and node.name=='Simulator':
                node.body=[n for n in node.body if not isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) or n.name!='service_bounds']
        return ast.dump(tree,include_attributes=False)
    source=previous/'sources/projection_engine.py'
    if normalized(source)!=normalized(HERE/'projection_engine.py'):raise ValueError('event-model methods changed')
    checked.append(fingerprint(source))
    recovered=[]
    for name in ('model_catalog.py','metrics.py','evidence.py'):
        snapshot=previous/'sources'/name
        if snapshot.read_bytes()!=(HERE/name).read_bytes():raise ValueError(f'auxiliary physics changed: {name}')
        recovered.append(fingerprint(snapshot))
    old_tree=ast.parse((previous/'sources/projection_sweep.py').read_text())
    new_tree=ast.parse((HERE/'projection_sweep.py').read_text())
    names=('layouts','candidates','screen_records','screen','local_screen','shortlist','design_key')
    for name in names:
        a=next(n for n in old_tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
        b=next(n for n in new_tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
        if ast.dump(a)!=ast.dump(b):raise ValueError(f'candidate search changed: {name}')
    source_results=previous/'results.jsonl';source_hash=fingerprint(source_results)
    rows={};truncated=0;failed=0
    for number,line in enumerate(source_results.read_text().splitlines(),1):
        try:row=json.loads(line)
        except json.JSONDecodeError:truncated+=1;continue
        if row['status'] not in ('simulated_projection','conditional_service_bound_excluded'):
            failed+=1;continue
        for template in row.get('templates',[]):
            original=previous/'runs'/template['file'];destination=out/'runs'/template['file']
            if not destination.exists():os.link(original,destination)
        row=dict(row,reused_from=dict(source=source_hash,line=number,event_physics_ast_equal=True))
        rows[(row['workload'],row['design_id'],row['seed'],row['view'])]=row
    return rows,dict(source=source_hash,rows=len(rows),truncated_rows=truncated,failed_rows_retried=failed,
                     equivalent_physics=checked,recovered_auxiliary_snapshots=recovered,
                     scope='Only completed rows reused; old source snapshots retained as execution lineage.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--models',nargs='*')
    parser.add_argument('--counts',type=int,nargs='*',default=COUNTS)
    parser.add_argument('--gpus',nargs='*',default=['rtx4090','h200'])
    parser.add_argument('--workloads',nargs='*',default=WORKLOADS)
    parser.add_argument('--views',nargs='*',default=['fixed','1','4','16'])
    parser.add_argument('--seeds',type=int,nargs='*',default=[17,29])
    parser.add_argument('--reuse-from',type=Path)
    parser.add_argument('--event-budget',type=int,default=200_000_000,
                        help='Computational guard only; does not change event times')
    args=parser.parse_args();out=args.out.resolve();out.mkdir(parents=True,exist_ok=False)
    if args.event_budget<1:raise ValueError('positive event budget required')
    Events.run.__kwdefaults__['max_events']=args.event_budget
    (out/'runs').mkdir();(out/'sources').mkdir()
    sources=[]
    for file in sorted(HERE.glob('*.py')):
        shutil.copy2(file,out/'sources'/file.name);sources.append(fingerprint(file))
    inputs=[HERE/'results/native-profiles-v4-20261104/manifest.json',HERE/'results/device-service-20261104/manifest.json',
            HERE/'results/model-metadata-20261104/manifest.json',HERE/'results/causal-workloads-v2-20261104/manifest.json',
            HERE/'results/projection-profiles-20261104/profiles.json']
    manifest=dict(schema='hbf-adoption.deployment-projection.v1',arguments=vars(args)|{'out':str(out)},
        source_inputs=[fingerprint(p) for p in inputs],sources=sources,started_unix=time.time(),
        workload=workload_manifest(),sources_snapshotted=True,status='running',
        screening='Power-of-two TP/PP layouts and batch caps 1/4/16; latency, throughput and balanced-SLO static extrema per comparator class are replayed. This is a declared screening policy, not an exhaustive E2E optimality proof.',
        conventional_cost_control='Host memory and eight SSDs are free in GPU-equivalent economic comparisons; CPUs use a separate zero-arithmetic bound.',
        precision='BF16 core text architecture; DeepSeek/Kimi are format counterfactuals of FP8-released checkpoints.',
        views='Four fixed closed-loop users; proportional 1/4/16 users per installed GPU. Maximum useful demand is the best SLO-passing tested concurrency, not an unbounded steady-state maximum.')
    reused,reuse_receipt=reusable_rows(args.reuse_from,out,manifest['source_inputs'])
    manifest['reused_execution']=reuse_receipt
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2,default=str)+'\n')
    runner=Runner(out);failures=0;cells=0;rows=0
    models=[m for m in catalog() if not args.models or m['id'] in args.models]
    with (out/'results.jsonl').open('w') as output,(out/'screening.jsonl').open('w') as screening:
        for model in models:
            for gpu in args.gpus:
                for workload in args.workloads:
                    for n in args.counts:
                        designs,scores=shortlist(model,n,gpu,workload)
                        screening.write(json.dumps(dict(model=model['id'],gpu=gpu,workload=workload,n=n,
                            candidates=scores,chosen=[design_key(d) for d in designs]),separators=(',',':'))+'\n')
                        screening.flush()
                        for d in designs:
                            for seed in args.seeds:
                                for view in args.views:
                                    try:
                                        row_key=(workload,design_key(d),seed,'fixed_users' if view=='fixed' else f'users_per_gpu_{view}')
                                        result=reused.get(row_key)
                                        if result is None:result=runner.execute(model,d,workload,seed,None if view=='fixed' else int(view))
                                    except Exception as e:
                                        failures+=1
                                        result=dict(status='failed',error=f'{type(e).__name__}: {e}',
                                            traceback=traceback.format_exc(),model=model['id'],gpu=gpu,workload=workload,
                                            n=n,design=asdict(d),seed=seed,view=view)
                                    output.write(json.dumps(result,separators=(',',':'))+'\n');output.flush();rows+=1
                        cells+=1
                        print(model['id'],gpu,workload,n,'candidates',len(scores),'replayed',len(designs),
                              'rows',rows,'native template runs',runner.executions,'failed',failures,flush=True)
                        # A bounded cache preserves reusable template receipts;
                        # it need not retain earlier workload token arrays.
                    runner.cache.clear()
                    runner.workload_cache.clear()
                    runner.bound_cache.clear()
    manifest.update(status='completed' if not failures else 'completed_with_failures',
                    cells=cells,result_rows=rows,failed_rows=failures,native_template_runs=runner.executions,
                    elapsed_seconds=time.time()-manifest['started_unix'],
                    outputs=[fingerprint(out/'results.jsonl'),fingerprint(out/'screening.jsonl')])
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2,default=str)+'\n')
    if failures:raise SystemExit(1)


if __name__=='__main__':main()
