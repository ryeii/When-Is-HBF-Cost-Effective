"""Apply the fixed-four-user secondary screen uniformly to the entire matrix."""
from __future__ import annotations
import argparse
from dataclasses import asdict,replace
from functools import lru_cache
import json
import math
from pathlib import Path
import shutil
import time
from boundary_replay import BoundarySimulator
from projection_engine import CapacityError
from projection_timing import layer_parameters
from projection_workloads import make,WORKLOADS
from model_catalog import catalog,kv_bytes
import projection_sweep as sweep
from projection_resources import Events
from run_validation import ROOT,HERE,fingerprint

MODELS={m['id']:m for m in catalog()}


@lru_cache(maxsize=20000)
def local_proxy(model_id,d,workload,users,active_on_node):
    model=MODELS[model_id];records=make(workload,17,4)
    if workload=='tool_agent':records=records+[dict(records[0],id='capacity-envelope',input_tokens=33152,output_tokens=128)]
    if d.comparator=='cpu_oracle' and model['family']!='moe':raise CapacityError('CPU bound is MoE-only')
    sim=BoundarySimulator(model,d,records);l=d.layout;t=sim.timings
    context=max(r['input_tokens'] for r in records)
    one=kv_bytes(model,max(r['input_tokens']+r['output_tokens'] for r in records),l.tp,l.pp)
    batch=min(d.batch_cap,users,max(1,int(sim.kv_capacity//one)))
    result=[]
    for prefill in (False,True):
        query=min(d.chunk,context) if prefill else 1
        components,_=t.components(prefill,batch,query,context)
        stages=[]
        for stage in range(l.pp):
            value=0.
            for layer in l.layers(model,stage):
                a,f,_,_=sim.layer_plan[layer]
                routed,fraction=sim._route(layer,batch*query)
                attn_p,ffn_p,_=layer_parameters(model,layer,l.tp)
                attention=max(components['attention']/model['layers'],2*attn_p/t.near_Bps)
                ffn=max(components['ffn']/model['layers'],(2*ffn_p+routed)/t.near_Bps)
                if prefill and attention+ffn<t.eager_layer_floor(True):attention=t.eager_layer_floor(True)-ffn
                oracle=d.comparator=='cpu_oracle' and routed>0
                cold_fraction=1. if f else fraction
                gpu_ffn=ffn*(1-cold_fraction) if oracle else ffn*(1+cold_fraction*(d.cold_compute_scale-1))
                gpu_attn=attention*(d.cold_compute_scale if a else 1.)
                copy=(a+(0. if oracle else f+routed))/t.near_Bps
                def source(payload,cpu):
                    activations=batch*query*model['hidden']*2*2*model['top_k']*l.tp if cpu else None
                    hops=sim._source_hops(0,stage,payload,cpu_activation_bytes=activations)
                    seconds=sum(h.duration for h in hops)
                    return seconds*(active_on_node if not l.hbf_stacks else 1)
                # Ranking proxy only: overlap each source with GPU service,
                # retain stage/path dependencies and charge staged consumption.
                value+=max(gpu_attn+a/t.near_Bps,source(a,False))
                value+=max(gpu_ffn+(0. if oracle else (f+routed)/t.near_Bps),source(f+routed,oracle))
            boundary=sim.plans[stage]['boundary_cold_bytes']
            value+=source(boundary,False) if boundary else 0.
            stages.append(value)
        path=sum(stages)+components['embedding']+components['logits']+t.cpu_seconds(prefill)
        occupancy=max(stages)*math.ceil(users/batch)
        result.append(max(path,occupancy))
    return dict(decode_s=result[0],ttft_s=result[1]*math.ceil(context/min(d.chunk,context)),effective_batch=batch)


def proxy(model,d,w):
    l=d.layout;users=math.ceil(4/l.replicas)
    local_replicas=min(l.replicas,max(1,8//l.width))
    local=replace(d,layout=replace(l,installed_gpus=l.width*local_replicas,replicas=local_replicas))
    active=min(4,l.replicas,max(1,8//l.width))
    return local_proxy(model['id'],local,w,users,active)


def selected(model,n,gpu,workload):
    groups={};scores=[]
    for d in sweep.candidates(n,gpu,workload):
        try:
            score=proxy(model,d,workload)
            groups.setdefault((d.layout.architecture,d.comparator),[]).append((d,score))
            scores.append(dict(design_id=sweep.design_key(d),status='ranked',**score))
        except CapacityError as e:scores.append(dict(design_id=sweep.design_key(d),status='capacity_excluded',reason=str(e)))
    chosen=[]
    def tie(item):return (item[0].batch_cap,item[0].layout.width,item[0].layout.hbf_stacks,item[0].chunk)
    for rows in groups.values():
        winners=[min(rows,key=lambda x:(x[1]['decode_s'],*tie(x))),min(rows,key=lambda x:(x[1]['ttft_s'],*tie(x)))]
        for ttft,itl in ((2,.05),(5,.1),(10,.2)):
            winners.append(min(rows,key=lambda x:(max(x[1]['ttft_s']/ttft,x[1]['decode_s']/itl),*tie(x))))
        for d,score in winners:
            if d not in chosen:chosen.append(d)
    return chosen,scores


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();out=a.out.resolve();out.mkdir(parents=True,exist_ok=False);shutil.copy2(__file__,out/'runner.py')
    manifest=dict(status='running',cells=0,jobs=[],protocol=fingerprint(HERE/'FIXED_DEMAND_SEARCH_AUDIT.md'))
    sweep.Simulator=BoundarySimulator;Events.run.__kwdefaults__['max_events']=200_000_000
    for model in sorted(MODELS.values(),key=lambda m:-m['parameters']):
        for gpu in ('h200','rtx4090'):
            folder=out/f"{model['id']}-{gpu}";folder.mkdir();(folder/'runs').mkdir()
            runner=sweep.Runner(folder);count=0;cells=0
            with (folder/'results.jsonl').open('w') as output,(folder/'screening.jsonl').open('w') as scoring:
                for w in WORKLOADS:
                    for n in sweep.COUNTS:
                        old,_=sweep.shortlist(model,n,gpu,w);old_ids={sweep.design_key(d) for d in old}
                        candidates,scores=selected(model,n,gpu,w)
                        added=[d for d in candidates if sweep.design_key(d) not in old_ids]
                        scoring.write(json.dumps(dict(model=model['id'],gpu=gpu,workload=w,n=n,scores=scores,
                            added=[asdict(d) for d in added]),separators=(',',':'))+'\n');scoring.flush()
                        for d in added:
                            for seed in (17,29):
                                row=runner.execute(model,d,w,seed);row['candidate_origin']='fixed_four_user_secondary_screen'
                                output.write(json.dumps(row,separators=(',',':'))+'\n');output.flush();count+=1
                        cells+=1
                        print(model['id'],gpu,w,n,'added',len(added),'rows',count,flush=True)
                    runner.cache.clear();runner.workload_cache.clear();runner.bound_cache.clear()
            record=dict(status='completed',model=model['id'],gpu=gpu,cells=cells,result_rows=count,
                sources=[fingerprint(HERE/'fixed_demand_expansion.py'),fingerprint(HERE/'boundary_replay.py'),
                    *[fingerprint(p) for p in HERE.glob('projection_*.py')]],
                outputs=[fingerprint(folder/'results.jsonl'),fingerprint(folder/'screening.jsonl')])
            (folder/'manifest.json').write_text(json.dumps(record,indent=2)+'\n')
            manifest['jobs'].append(dict(id=folder.name,manifest=fingerprint(folder/'manifest.json')));manifest['cells']+=cells
            (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
            local_proxy.cache_clear()
    manifest['status']='completed';(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')


if __name__=='__main__':main()
