"""Correct nonresident embedding/logit scheduling without discarding valid runs.

Only rows with nonzero boundary-cold payload need a new token trajectory.
Unchanged designs and optimistic bounds remain valid. Raw originals are kept.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
import gzip
import json
from pathlib import Path
import shutil
import time
from model_catalog import catalog
from projection_engine import Simulator,Design
from projection_timing import Layout
from projection_resources import Events
import projection_sweep as sweep
from run_validation import ROOT,HERE,fingerprint


class BoundarySimulator(Simulator):
    def boundary_fraction(self,stage):
        p=self.plans[stage]
        # Reconstruct the total boundary tensor allocation. The plan pins a
        # common proportion of embedding/head on a one-stage replica.
        layers=list(self.layout.layers(self.model,stage))
        from projection_timing import layer_parameters
        total=p['weight_bytes']-sum(2*sum(layer_parameters(self.model,i,self.layout.tp)) for i in layers)
        return p['boundary_cold_bytes']/max(1.,total)

    def stage_begin(self,batch,stage):
        cold=self.plans[stage]['boundary_cold_bytes']
        if cold<=0:return super().stage_begin(batch,stage)
        replica=batch['replica']
        if stage and self.stage_busy[replica][stage]:
            self.stage_queues[replica][stage].append(batch);return
        self.stage_busy[replica][stage]=True
        current=dict(batch,stage=stage,layer_list=list(self.layout.layers(self.model,stage)),layer_index=0)
        embedding=batch['components']['embedding'] if stage==0 else 0.
        payload=(cold/2 if self.layout.pp==1 else cold) if stage==0 else 0.
        if payload:
            self.stream(current,payload,embedding,self.boundary_fraction(stage),lambda:self.next_layer(current),cpu_allowed=False)
        else:self.resource(f'gpu:{replica}:{stage}').submit(embedding,self.next_layer,current)

    def stage_finish(self,batch):
        stage=batch['stage'];cold=self.plans[stage]['boundary_cold_bytes']
        if stage!=self.layout.pp-1 or cold<=0:return super().stage_finish(batch)
        payload=cold/2 if self.layout.pp==1 else cold
        # Last-layer dependencies are complete now. The stream consumes the
        # logits operation; the base finish then releases credits with zero
        # remaining logits work. No output-head bytes are prefetched early.
        current=dict(batch,components={**batch['components'],'logits':0.})
        self.stream(batch,payload,batch['components']['logits'],self.boundary_fraction(stage),
            lambda:super(BoundarySimulator,self).stage_finish(current),cpu_allowed=False)


def affected(folder,row):
    if row['status']!='simulated_projection':return False
    for template in row['templates']:
        with gzip.open(folder/'runs'/template['file'],'rt') as stream:run=json.load(stream)
        if any(p['boundary_cold_bytes']>0 for p in run['memory_plans']):return True
    return False


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args();out=a.out.resolve();out.mkdir(parents=True,exist_ok=False)
    shutil.copy2(__file__,out/'runner.py');(out/'runs').mkdir()
    models={m['id']:m for m in catalog()};Events.run.__kwdefaults__['max_events']=200_000_000
    sweep.Simulator=BoundarySimulator;runner=sweep.Runner(out);count=0;manifest=dict(status='running',sources=[],outputs=[])
    # Process each completed worker as soon as its immutable result is available.
    jobs=[f"{m['id']}-{g}" for m in sorted(models.values(),key=lambda m:-m['parameters']) for g in ('h200','rtx4090')]
    with (out/'results.jsonl').open('w') as stream:
        for name in jobs:
            folder=a.input.resolve()/name
            while True:
                try:record=json.loads((folder/'manifest.json').read_text())
                except (FileNotFoundError,json.JSONDecodeError):time.sleep(10);continue
                if record['status']=='completed':break
                if record['status']=='completed_with_failures':raise ValueError('central worker failed')
                time.sleep(10)
            source=fingerprint(folder/'results.jsonl');manifest['sources'].append(source)
            for line in (folder/'results.jsonl').read_text().splitlines():
                row=json.loads(line)
                if not affected(folder,row):continue
                d=dict(row['design']);design=Design(Layout(**d.pop('layout')),**d)
                users=None if row['view']=='fixed_users' else int(row['view'].rsplit('_',1)[1])
                result=runner.execute(models[row['model']],design,row['workload'],row['seed'],users)
                result['corrected_from']=dict(source=source,design_id=row['design_id'],seed=row['seed'],view=row['view'],
                    reason='Nonresident embedding and logits consumed at their causal boundaries; proportional pinning on PP1.')
                stream.write(json.dumps(result,separators=(',',':'))+'\n');stream.flush();count+=1
            runner.cache.clear();runner.workload_cache.clear();runner.bound_cache.clear()
            print('Boundary audit',name,'corrected rows',count,flush=True)
    manifest.update(status='completed',corrected_rows=count,
        inputs=[fingerprint(p) for p in sorted(HERE.glob('projection_*.py'))]+[fingerprint(Path(__file__))],
        outputs=[fingerprint(out/'results.jsonl')])
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')


if __name__=='__main__':main()
