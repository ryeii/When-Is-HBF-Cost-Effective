"""Parallel orchestration of the unchanged fixed-demand screen and replay."""
from concurrent.futures import ThreadPoolExecutor,as_completed
import argparse
import json
import os
from pathlib import Path
import shutil
import sys
from model_catalog import catalog
from run_validation import ROOT,HERE,execute,fingerprint


def worker(args):
    import fixed_demand_expansion as task
    import projection_sweep as sweep
    if args.reuse_root and (args.reuse_root/'runner.py').read_bytes()!=(HERE/'fixed_demand_expansion.py').read_bytes():
        raise ValueError('secondary-screen implementation changed')
    reused={};sources={}
    for gpu in ('h200','rtx4090') if args.reuse_root else ():
        folder=args.reuse_root/f'{args.worker}-{gpu}';path=folder/'results.jsonl'
        if not path.exists():continue
        source=fingerprint(path);sources[gpu]=source
        for number,line in enumerate(path.read_text().splitlines(),1):
            try:r=json.loads(line)
            except json.JSONDecodeError:continue
            if r['status'] not in ('simulated_projection','conditional_service_bound_excluded'):continue
            reused[(gpu,r['workload'],r['design_id'],r['seed'])]=(r,folder,source,number)
    original=sweep.Runner.execute
    def replay(self,model,design,workload,seed,*extra,**kwargs):
        key=(design.gpu,workload,sweep.design_key(design),seed)
        if key not in reused:return original(self,model,design,workload,seed,*extra,**kwargs)
        row,folder,source,number=reused[key]
        for template in row.get('templates',[]):
            destination=self.out/'runs'/template['file']
            if not destination.exists():shutil.copyfile(folder/'runs'/template['file'],destination)
        return dict(row,reused_from=dict(source=source,line=number,event_physics_ast_equal=True,
            scope='Identical frozen secondary-screen code and event inputs; only model-job orchestration changed.'))
    sweep.Runner.execute=replay
    task.MODELS={args.worker:task.MODELS[args.worker]}
    sys.argv=[str(HERE/'fixed_demand_expansion.py'),'--out',str(args.out.resolve())]
    task.main()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--reuse-root',type=Path);p.add_argument('--workers',type=int,default=4);p.add_argument('--worker')
    p.add_argument('--low-memory',action='store_true')
    args=p.parse_args();args.reuse_root=args.reuse_root.resolve() if args.reuse_root else None
    if args.worker:worker(args);return
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=False);(out/'parts').mkdir();shutil.copy2(__file__,out/'runner.py')
    if args.low_memory:from memory_entry import runtime_inputs
    else:from compiled_entry import runtime_inputs
    manifest=dict(status='running',cells=0,jobs=[],protocol=fingerprint(HERE/'FIXED_DEMAND_SEARCH_AUDIT.md'),
        orchestration='Independent model pairs at normal process priority; unchanged ranking/replay implementation.')
    def run(model):
        part=out/'parts'/model
        wrapper=HERE/('memory_entry.py' if args.low_memory else 'compiled_entry.py')
        argv=[sys.executable,str(wrapper),str(Path(__file__).resolve()),
            '--worker',model,'--out',str(part)]
        if args.reuse_root:
            argv+=['--reuse-root',str(args.reuse_root)]
        receipt=execute(argv,ROOT,out/(model+'-execution'),timeout=18*3600,
            inputs=[Path(__file__).resolve(),HERE/'fixed_demand_expansion.py',HERE/'boundary_replay.py',
                HERE/'FIXED_DEMAND_SEARCH_AUDIT.md',*HERE.glob('projection_*.py'),*runtime_inputs()])
        if receipt['status']!='pass':raise ValueError('secondary-screen worker failed')
        top=json.loads((part/'manifest.json').read_text())
        if top['status']!='completed' or top['cells']!=90:raise ValueError('incomplete model pair')
        published=[]
        for job in top['jobs']:
            source=part/job['id'];destination=out/job['id']
            shutil.copytree(source,destination,copy_function=os.link)
            record=json.loads((source/'manifest.json').read_text())
            # Replace only the copied manifest so original execution receipts
            # and original manifests remain immutable and independently usable.
            (destination/'manifest.json').unlink()
            record.update(relocated_from=fingerprint(source/'manifest.json'),
                execution_receipt=fingerprint(out/(model+'-execution')/'receipt.json'),
                outputs=[fingerprint(destination/'results.jsonl'),fingerprint(destination/'screening.jsonl')])
            (destination/'manifest.json').write_text(json.dumps(record,indent=2)+'\n')
            published.append(dict(id=job['id'],manifest=fingerprint(destination/'manifest.json')))
        return published
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(run,m['id']) for m in sorted(catalog(),key=lambda m:-m['parameters'])]
        for future in as_completed(futures):
            manifest['jobs'].extend(future.result());manifest['cells']=45*len(manifest['jobs'])
            (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    manifest['status']='completed';(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')


if __name__=='__main__':main()
