"""Current-batch held-out routing intervention; no future-route prefetch."""
from __future__ import annotations
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
from model_catalog import catalog
from projection_engine import Simulator,Design
from projection_timing import Layout
import projection_sweep as sweep
from run_validation import ROOT,HERE,fingerprint,execute

ROUTES=HERE/'results/routing-20261104/general.npz'


class RouteSimulator(Simulator):
    data=None
    ranking=None
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        if RouteSimulator.data is None:
            with np.load(ROUTES) as z:RouteSimulator.data=next(z[k] for k in z if z[k].ndim==3)
            count=np.array([np.bincount(layer[:2000].reshape(-1),minlength=128) for layer in self.data])
            RouteSimulator.ranking=np.argsort(-count,axis=1,kind='stable')
        self.active_batch=None

    def next_layer(self,batch):
        self.active_batch=batch
        return super().next_layer(batch)

    def _route(self,layer,tokens):
        batch=self.active_batch
        if batch is None:return super()._route(layer,tokens)
        selected=[]
        for rid,q in zip(batch['requests'],batch['queries']):
            r=self.requests[rid]
            offset=int(hashlib.sha256(f"heldout:{r['session']}".encode()).hexdigest()[:12],16)%6000
            position=r['processed'] if batch['prefill'] else r['input_tokens']+len(r['token_times_s'])-1
            indices=2000+(offset+position+np.arange(q))%6000
            selected.append(self.data[layer,indices].reshape(-1))
        assignments=np.concatenate(selected)
        cache=self.layer_plan[layer][2]
        hot=self.ranking[layer,:cache]
        cold=assignments[~np.isin(assignments,hot)]
        return len(np.unique(cold))*self.layer_plan[layer][3],len(cold)/max(1,len(assignments))

    def service_bounds(self):
        # Mean-field concavity certificates do not apply to these correlated
        # routes. Disable exclusions and execute every intervention trajectory.
        roots=[r for r in self.requests.values() if r['predecessor'] is None]
        return dict(decode_interval_s=0.,per_replica_decode_s=[0.]*self.layout.replicas,
            offline_makespan_s=0.,mandatory_resource_work_s={},mandatory_work_by_replica={},
            decode_tokens_by_replica={},root_requests=len(roots),passing_roots_upper_bound=len(roots),
            root_response_deadline_s=max(10+.2*(r['output_tokens']-1) for r in roots),
            scope='No mean-field exclusion applied to correlated held-out route intervention.')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();out=args.out.resolve();out.mkdir(parents=True,exist_ok=False)
    (out/'runs').mkdir();shutil.copy2(__file__,out/'runner.py')
    model=next(m for m in catalog() if m['id']=='qwen3_30b_a3b')
    sweep.Simulator=RouteSimulator
    rows=[]
    for gpu in ('rtx4090','h200'):
        for n in (1,8,32):
            chosen,_=sweep.shortlist(model,n,gpu,'short_chat')
            runner=sweep.Runner(out)
            for d in chosen:
                exact=runner.execute(model,d,'short_chat',17)
                rows.append(exact)
            print(gpu,n,'complete',len(rows),flush=True)
    (out/'results.jsonl').write_text(''.join(json.dumps(r,separators=(',',':'))+'\n' for r in rows))
    manifest=dict(status='completed',schema='hbf-adoption.route-intervention.v1',
        inputs=[fingerprint(ROUTES),fingerprint(HERE/'SENSITIVITY_PROTOCOL.md'),
                *[fingerprint(p) for p in sorted(HERE.glob('projection_*.py'))]],
        runner=fingerprint(out/'runner.py'),outputs=[fingerprint(out/'results.jsonl')],
        counts=[1,8,32],seed=17,training_tokens=2000,heldout_tokens=6000,
        interpretation='Q4 route identities transferred to BF16 payload; fixed-four-user program; no BF16 routing or request-tail validation claim.')
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')


if __name__=='__main__':main()
