"""Execute the declared matched analysis domain using unchanged physics.

This main loop is derived from projection_sweep.main. Only unmatched wide
pipeline proportional tool views are omitted; no status or clock is invented.
"""
from projection_sweep import *


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
    runner=Runner(out);failures=0;cells=0;rows=0;omissions=[]
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
                                    if workload=='tool_agent' and d.layout.width>8 and view!='fixed':
                                        omissions.append(dict(workload=workload,n=n,design_id=design_key(d),seed=seed,view=view,reason='Outside matched heterogeneous tool capacity domain; fixed-user comparisons remain included.'))
                                        continue
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
    (out/'scope-omissions.json').write_text(json.dumps(omissions,indent=2)+'\n')
    manifest['matched_tool_scope']=dict(omitted_views=len(omissions),reason='Replica widths above eight are outside the matched heterogeneous tool useful-capacity comparison; no SLO outcome is assigned.')
    manifest.update(status='completed' if not failures else 'completed_with_failures',
                    cells=cells,result_rows=rows,failed_rows=failures,native_template_runs=runner.executions,
                    elapsed_seconds=time.time()-manifest['started_unix'],
                    outputs=[fingerprint(out/'results.jsonl'),fingerprint(out/'screening.jsonl'),fingerprint(out/'scope-omissions.json')])
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2,default=str)+'\n')
    if failures:raise SystemExit(1)


if __name__=='__main__':main()
