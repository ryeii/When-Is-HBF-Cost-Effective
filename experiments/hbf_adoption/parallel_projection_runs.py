"""Run independent model/platform sweeps in bounded CPU processes."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import shutil
import sys
import time

from model_catalog import catalog
from run_validation import ROOT,HERE,execute,fingerprint


def completed_jobs(roots):
    """Find intact matched-domain components with current execution inputs."""
    completed={}
    for root in roots:
        try:jobs=json.loads((root/'manifest.json').read_text())['jobs']
        except (OSError,ValueError,KeyError):continue
        for job in jobs:
            if job.get('status')!='pass' or job['id'] in completed:continue
            try:
                manifest,receipt=(ROOT/job[key]['path'] for key in ('manifest','receipt'))
                if any(fingerprint(ROOT/job[k]['path'])!=job[k] for k in ('manifest','receipt')):continue
                worker=json.loads(manifest.read_text());proof=json.loads(receipt.read_text())
                if (worker['status']!='completed' or worker['cells']!=45 or worker['failed_rows']
                    or 'matched_tool_scope' not in worker or proof['status']!='pass'
                    or not proof['input_hashes_unchanged']):continue
                if not {'matched_projection_sweep.py','projection_engine.py'}<={
                    Path(r['path']).name for r in proof['inputs']}:continue
                if any(fingerprint(ROOT/r['path'])!=r for r in
                       proof['inputs']+worker['source_inputs']+worker['outputs']):continue
            except (OSError,ValueError,KeyError,TypeError):continue
            completed[job['id']]=job
    return completed


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--workers',type=int,default=4)
    p.add_argument('--reuse-root',type=Path)
    p.add_argument('--completed-from',type=Path,action='append',default=[])
    p.add_argument('--fallback-root',type=Path);p.add_argument('--compiled',action='store_true')
    p.add_argument('--low-memory',action='store_true')
    p.add_argument('--matched-domain',action='store_true')
    p.add_argument('--event-budget',type=int,default=200_000_000)
    args=p.parse_args();out=args.out.resolve();out.mkdir(parents=True,exist_ok=False)
    shutil.copy2(__file__,out/'runner.py')
    start=time.time()
    manifest=dict(schema='hbf-adoption.parallel-deployment.v1',started_utc=datetime.now(timezone.utc).isoformat(),
                  workers=args.workers,status='running',jobs=[])
    completed=completed_jobs(args.completed_from) if args.matched_domain else {}
    def run(model,gpu):
        name=f'{model}-{gpu}';destination=out/name
        if name in completed:
            job=completed[name]
            source=(ROOT/job['manifest']['path']).parent
            shutil.copytree(source,destination,copy_function=os.link)
            print(f'{name}: carrying forward 45 completed cells',flush=True)
            return job
        wrapper=HERE/('memory_entry.py' if args.low_memory else 'compiled_entry.py' if args.compiled else 'runtime_runner.py')
        target=HERE/('matched_projection_sweep.py' if args.matched_domain else 'projection_sweep.py')
        argv=[sys.executable,str(wrapper),str(target),'--out',str(destination),
              '--models',model,'--gpus',gpu,'--event-budget',str(args.event_budget)]
        for base in (args.reuse_root,args.fallback_root):
            if base and (base/name/'manifest.json').exists():
                argv+=['--reuse-from',str((base/name).resolve())];break
        extra=[]
        if args.low_memory:
            from memory_entry import runtime_inputs
            extra=runtime_inputs()
        elif args.compiled:
            from compiled_entry import runtime_inputs
            extra=runtime_inputs()
        receipt=execute(argv,ROOT,out/(name+'-execution'),timeout=12*3600,
                        inputs=[*HERE.glob('projection_*.py'),target,HERE/'runtime_runner.py',*extra])
        return dict(id=name,receipt=fingerprint(out/(name+'-execution')/'receipt.json'),
                    status=receipt['status'],manifest=fingerprint(destination/'manifest.json')
                    if (destination/'manifest.json').exists() else None)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        work=[(m['id'],gpu) for m in sorted(catalog(),key=lambda m:m['parameters'],reverse=True)
              for gpu in ('rtx4090','h200')]
        work.sort(key=lambda item: '-'.join(item) not in completed)
        jobs=[pool.submit(run,*item) for item in work]
        for future in as_completed(jobs):
            manifest['jobs'].append(future.result())
            (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    manifest.update(status='completed' if all(j['status']=='pass' for j in manifest['jobs']) else 'completed_with_failures',
                    elapsed_seconds=time.time()-start)
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    if manifest['status']!='completed':raise SystemExit(1)


if __name__=='__main__':main()
