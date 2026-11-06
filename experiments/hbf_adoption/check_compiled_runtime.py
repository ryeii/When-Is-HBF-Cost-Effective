"""Check an optional compiled execution of identical, untyped Python sources."""
import argparse
from functools import lru_cache
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import time
import unittest
from run_validation import ROOT,HERE,fingerprint

BUILD=ROOT/'.cache/hbf-adoption/compiled-event-model'


def install():
    for name in ('projection_resources','projection_engine'):
        if (BUILD/(name+'.py')).read_bytes()!=(HERE/(name+'.py')).read_bytes():raise ValueError('compiled source differs')
        binary=next(BUILD.glob(name+'.*.so'))
        spec=importlib.util.spec_from_file_location(name,binary);module=importlib.util.module_from_spec(spec)
        sys.modules[name]=module;spec.loader.exec_module(module)


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--tests',action='store_true')
    a=p.parse_args();out=a.out.resolve();out.mkdir(parents=True,exist_ok=False);shutil.copy2(__file__,out/'runner.py')
    import projection_engine as pure
    from projection_timing import Layout
    from projection_workloads import make
    from model_catalog import catalog
    models={m['id']:m for m in catalog()};install();import projection_engine as compiled
    cases=[('qwen3_8b','short_chat','h200',16,1,False,64),
        ('qwen3_30b_a3b','short_chat','rtx4090',4,4,False,8),
        ('llama31_405b','offline_batch','h200',4,4,False,8),
        ('qwen3_8b','paused_personal','h200',4,4,True,8)]
    rows=[]
    for model,workload,gpu,roots,batch,prefix,output in cases:
        records=make(workload,17,roots)
        for r in records:r['output_tokens']=output
        results=[];seconds=[]
        def run(module):
            d=module.Design(Layout(1,1,1,1,'h3_cascaded',8,4),gpu,batch_cap=batch,prefix=prefix)
            start=time.perf_counter();s=module.Simulator(models[model],d,records);s._route=lru_cache(maxsize=8192)(s._route)
            r=s.run(retain_tokens=True)
            return time.perf_counter()-start,json.dumps(r,sort_keys=True,separators=(',',':'))
        for module in (pure,compiled):run(module)  # Warm native query/inclusion caches for both.
        samples=[[],[]];texts=[[],[]]
        for index in (0,1,1,0,0,1):
            elapsed,encoded=run((pure,compiled)[index]);samples[index].append(elapsed);texts[index].append(encoded)
        for index in (0,1):
            if len(set(texts[index]))!=1:raise ValueError('runtime is nondeterministic')
            seconds.append(sorted(samples[index])[1]);results.append(texts[index][0])
        if results[0]!=results[1]:raise ValueError(f'compiled execution changed complete trajectory: {model}')
        rows.append(dict(model=model,workload=workload,gpu=gpu,roots=roots,batch=batch,
            trajectory_sha256=hashlib.sha256(results[0].encode()).hexdigest(),identical=True,
            python_seconds=seconds[0],compiled_seconds=seconds[1],speedup=seconds[0]/seconds[1]))
        print(rows[-1],flush=True)
    tests=None
    if a.tests:
        suite=unittest.defaultTestLoader.discover(str(HERE),pattern='test_*.py')
        result=unittest.TextTestRunner(verbosity=2).run(suite)
        tests=dict(tests=result.testsRun,failures=len(result.failures),errors=len(result.errors))
        if not result.wasSuccessful():raise ValueError('compiled runtime contract checks failed')
    (out/'manifest.json').write_text(json.dumps(dict(status='completed',checks=rows,tests=tests,
        inputs=[fingerprint(p) for p in sorted(BUILD.glob('*')) if p.is_file()]+[fingerprint(HERE/'projection_resources.py'),fingerprint(HERE/'projection_engine.py')],
        interpretation='Cython 3.2.1 with annotation_typing=False and infer_types=False; no physics/source change. Warmed native/inclusion caches; ABBAAB timing order; median of three per runtime. Clock/resource/traffic dictionaries match exactly on declared cases.'),indent=2)+'\n')


if __name__=='__main__':main()
