"""Verify complete projection coverage and generate conditional frontiers.

Certificates exclude SLOs under the model; they never supply throughput.
No validated-adoption classifier is weakened to accept projections.
"""
from __future__ import annotations
import argparse
from collections import defaultdict,Counter
from functools import lru_cache
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from run_validation import ROOT,fingerprint
from model_catalog import catalog
from result_rows import Corrections,Expansion

ARCHES=['baseline','shared_site','h3_cascaded']
NAMES={'baseline':'Conventional','shared_site':'Shared site','h3_cascaded':'Cascaded'}
COLORS={'baseline':'#343e48','shared_site':'#c47b21','h3_cascaded':'#176ea4'}
SHORT={'qwen3_8b':'Qwen 8B','qwen3_32b':'Qwen 32B','llama31_70b':'Llama 70B',
       'llama31_405b':'Llama 405B','qwen3_30b_a3b':'Qwen 30/A3','qwen3_235b_a22b':'Qwen 235/A22',
       'deepseek_v3':'DeepSeek V3','kimi_k2':'Kimi K2'}


def savefig(fig,out,name):
    for ext in ('pdf','png','svg'):fig.savefig(out/(name+'.'+ext),bbox_inches='tight',dpi=180)
    plt.close(fig)


@lru_cache(maxsize=None)
def checked_fingerprint(path):return fingerprint(ROOT/path)


def verify(record):
    if checked_fingerprint(record['path'])!=record:raise ValueError(f"changed input {record['path']}")


def load(root,corrections,expansion):
    manifest=json.loads((root/'manifest.json').read_text())
    if manifest['status']!='completed' or len(manifest['jobs'])!=16:
        raise ValueError('complete successful 16-job projection receipt required')
    raw=[];inputs=[fingerprint(root/'manifest.json'),fingerprint(corrections.path/'manifest.json'),fingerprint(corrections.path/'results.jsonl')];templates={};cells=0;reused=0;scope_omissions=0
    inputs.extend(expansion.inputs)
    for job in manifest['jobs']:
        if job['status']!='pass':raise ValueError('failed worker')
        verify(job['receipt']);verify(job['manifest'])
        receipt=json.loads((ROOT/job['receipt']['path']).read_text())
        if receipt['status']!='pass' or not receipt['input_hashes_unchanged']:raise ValueError('worker inputs changed')
        folder=root/job['id'];worker=json.loads((folder/'manifest.json').read_text())
        if worker['status']!='completed' or worker['failed_rows']:raise ValueError('incomplete worker coverage')
        cells+=worker['cells']
        scope_omissions+=worker.get('matched_tool_scope',{}).get('omitted_views',0)
        for row in worker['outputs']:verify(row);inputs.append(row)
        for row in worker['sources']:
            snapshot=folder/'sources'/Path(row['path']).name
            if hashlib.sha256(snapshot.read_bytes()).hexdigest()!=row['sha256']:raise ValueError('source snapshot mismatch')
        for row in corrections.load(folder)+expansion.load(folder.name):
            if row['status'] not in ('simulated_projection','conditional_service_bound_excluded'):
                raise ValueError('unknown/failed projection is not an exclusion')
            if 'reused_from' in row:reused+=1;verify(row['reused_from']['source'])
            if row['status']=='simulated_projection':
                for t in row['templates']:
                    path=Path(row['_run_directory'])/'runs'/t['file'];templates[str(path)]=path
            raw.append(row)
        inputs.append(fingerprint(folder/'manifest.json'))
    corrections.complete()
    if cells!=720:raise ValueError(f'expected 720 model/platform/workload/count cells; got {cells}')
    # Recheck finite token populations, causal successor release, resource work
    # and physical service bounds on every distinct executed template.
    counts=Counter();template_hashes=[]
    for path in templates.values():
        with gzip.open(path,'rt') as stream:run=json.load(stream)
        audited={corrections.path,*expansion.jobs.values()}
        if any(p['boundary_cold_bytes']>0 for p in run['memory_plans']) and path.parent.parent not in audited:
            raise ValueError('uncorrected nonresident boundary trajectory')
        if run['status']!='simulated_projection' or run['validated'] is not False:raise ValueError('wrong evidence level')
        by_id={r['id']:r for r in run['requests']}
        if len(by_id)!=run['completed_requests']:raise ValueError('duplicate/missing request')
        for r in by_id.values():
            ts=r['token_times_s']
            if len(ts)!=r['expected_output_tokens'] or any(b<a for a,b in zip(ts,ts[1:])):
                raise ValueError('incomplete/noncausal token trajectory')
            if ts[0]<r['arrival_s']:raise ValueError('token precedes request')
            if r['predecessor'] is not None:
                previous=by_id[r['predecessor']]
                if not math.isclose(r['arrival_s'],previous['token_times_s'][-1]+previous['successor_gap_s'],abs_tol=1e-7):
                    raise ValueError('noncausal successor release')
        if any(r['utilization']>1.000001 for r in run['resources'].values()):raise ValueError('resource overbooked')
        counts['requests']+=len(by_id);counts['events']+=run['events'];counts['templates']+=1
        template_hashes.append(fingerprint(path))
    groups=defaultdict(list)
    for row in raw:groups[(row['model'],row['gpu'],row['workload'],row['design_id'],row['view'])].append(row)
    designs=[]
    for key,rows in groups.items():
        if len(rows)!=2 or {r['seed'] for r in rows}!={17,29}:raise ValueError('paired seed coverage missing')
        m,g,w,did,view=key;design=rows[0]['design'];layout=design['layout']
        summary=dict(model=m,gpu=g,workload=w,design_id=did,view=view,design=design,
            n=layout['installed_gpus'],architecture=layout['architecture'],
            comparator=design['comparator'],hbf_TB=layout['installed_gpus']*layout['hbf_stacks']*512*2**30/1e12,
            candidate_origin=rows[0].get('candidate_origin','original_static_pool'),
            installed_stacks=layout['installed_gpus']*layout['hbf_stacks'],seeds=[17,29],metrics={})
        for slo in ('responsive','balanced','relaxed','offline'):
            feasible=all(r['status']=='simulated_projection' and
                (r['offline_deadline_pass'] if slo=='offline' else r['metrics'][slo]['slo_pass']) for r in rows)
            values=[r for r in rows if r['status']=='simulated_projection']
            rate=min((r['output_tokens']/r['elapsed_s'] if slo=='offline' else
                      r['metrics'][slo]['good_output_tokens_per_second']) for r in values) if feasible else None
            metrics=dict(feasible=feasible,good_tokens_per_second=rate,
                certificate_windows=sum(r['status']=='conditional_service_bound_excluded' for r in rows))
            if slo!='offline' and values:
                for name in ('ttft_s_p50','ttft_s_p99','mean_itl_s_p50','p99_itl_s_p99','joint_success_fraction'):
                    # Report the adverse window: latency is larger, but
                    # joint attainment is smaller. Feasibility above already
                    # requires every window to pass independently.
                    reduce=min if name=='joint_success_fraction' else max
                    metrics[name]=reduce(r['metrics'][slo][name] for r in values)
            summary['metrics'][slo]=metrics
        designs.append(summary)
    return designs,dict(raw_rows=len(raw),statuses=dict(Counter(r['status'] for r in raw)),
        paired_design_views=len(designs),coverage_cells=cells,reused_completed_rows=reused,
        unmatched_tool_views_omitted=scope_omissions,
        corrected_boundary_rows=len(corrections.rows),correction_manifest=fingerprint(corrections.path/'manifest.json'),
        expansion_rows=sum('candidate_origin' in r for r in raw),expansion_manifest=fingerprint(expansion.path/'manifest.json'),
        template_checks=dict(counts),template_hashes=template_hashes,inputs=inputs)


def best(pool,slo,*,cost_first=False):
    eligible=[d for d in pool if d['metrics'][slo]['feasible']]
    if not eligible:return None
    return min(eligible,key=lambda d:(d['n'],-d['metrics'][slo]['good_tokens_per_second'],d['design_id'])) if cost_first else \
           max(eligible,key=lambda d:(d['metrics'][slo]['good_tokens_per_second'],-d['n'],d['design_id']))


def minimum_frontiers(designs):
    groups=defaultdict(list)
    for d in designs:
        if d['view']=='fixed_users':groups[(d['model'],d['gpu'],d['workload'])].append(d)
    rows=[]
    for (model,gpu,workload),pool in groups.items():
        for slo in (['offline'] if workload=='offline_batch' else ['responsive','balanced','relaxed']):
            choices={a:best([d for d in pool if d['architecture']==a],slo,cost_first=True) for a in ARCHES}
            gpu_only=best([d for d in pool if d['architecture']=='baseline' and d['comparator']!='cpu_oracle'],slo,cost_first=True)
            base=choices['baseline']
            for arch in ('shared_site','h3_cascaded'):
                hbf=choices[arch]
                eligible=[d for d in pool if d['architecture']==arch and d['metrics'][slo]['feasible']]
                ceiling=max(eligible,key=lambda d:((base['n']-d['n'])/d['hbf_TB'],-d['n'])) if base and eligible else None
                price=None if ceiling is None else (base['n']-ceiling['n'])/ceiling['hbf_TB']
                classification=('reference_not_found_in_pool' if base is None else
                    'no_passing_hbf_design_in_pool' if hbf is None else
                    'positive_headroom_within_pool' if hbf['n']<base['n'] else 'no_gpu_acquisition_headroom')
                rows.append(dict(model=model,gpu=gpu,workload=workload,slo=slo,architecture=arch,
                    baseline_n=None if base is None else base['n'],baseline_design=None if base is None else base['design_id'],
                    baseline_control=None if base is None else base['comparator'],
                    gpu_staging_or_resident_n=None if gpu_only is None else gpu_only['n'],
                    hbf_n=None if hbf is None else hbf['n'],hbf_design=None if hbf is None else hbf['design_id'],
                    hbf_stacks=None if hbf is None else hbf['installed_stacks'],
                    hbf_TB=None if hbf is None else hbf['hbf_TB'],
                    affordable_total_premium_gpu_units=None if base is None or hbf is None else base['n']-hbf['n'],
                    maximum_net_premium_gpu_units_per_TB=price,
                    ceiling_design=None if ceiling is None else ceiling['design_id'],classification=classification))
    return rows


def scale_frontiers(designs):
    groups=defaultdict(list)
    # Retain every declared cell, including a class with no matched candidate.
    # Its explicit missing value must not disappear from plots/tables or become
    # zero physical capacity.
    for model in catalog():
        for gpu in ('rtx4090','h200'):
            for workload in ('short_chat','long_rag','tool_agent','paused_personal','offline_batch'):
                for n in (1,2,4,8,16,32,64,72,100):
                    for architecture in ARCHES:
                        groups[(model['id'],gpu,workload,n,architecture)]
    for d in designs:
        if d['view']=='fixed_users':continue
        # Exact node-group copying gives matched controlled shapes. Heterogeneous
        # public tool shapes have different group samples when width>8; restrict
        # this one maximum-capacity view to the matched <=8-GPU replica domain.
        if d['workload']=='tool_agent' and d['design']['layout']['tp']*d['design']['layout']['pp']>8:continue
        groups[(d['model'],d['gpu'],d['workload'],d['n'],d['architecture'])].append(d)
    rows=[]
    for (m,g,w,n,a),pool in groups.items():
        for slo in (['offline'] if w=='offline_batch' else ['responsive','balanced','relaxed']):
            choice=best(pool,slo)
            rows.append(dict(model=m,gpu=g,workload=w,n=n,architecture=a,slo=slo,
                tested_useful_good_tokens_per_second=None if choice is None else choice['metrics'][slo]['good_tokens_per_second'],
                design_id=None if choice is None else choice['design_id'],
                selected_view=None if choice is None else choice['view']))
    return rows


def price_curves(designs,frontiers):
    """Re-select every feasible design at each installed per-stack premium."""
    groups=defaultdict(list)
    for d in designs:
        if d['view']=='fixed_users':groups[(d['model'],d['gpu'],d['workload'],d['architecture'])].append(d)
    output=[]
    for row in frontiers:
        pool=[d for d in groups[(row['model'],row['gpu'],row['workload'],row['architecture'])]
              if d['metrics'][row['slo']]['feasible']]
        for premium in (0,.01,.025,.05,.1,.25,.5,1,2,4):
            chosen=min(pool,key=lambda d:(d['n']+premium*d['installed_stacks'],d['design_id'])) if pool else None
            cost=None if chosen is None else chosen['n']+premium*chosen['installed_stacks']
            output.append(dict(model=row['model'],gpu=row['gpu'],workload=row['workload'],slo=row['slo'],
                architecture=row['architecture'],premium_gpu_units_per_stack=premium,
                selected_design=None if chosen is None else chosen['design_id'],hbf_cost_gpu_units=cost,
                baseline_cost_gpu_units=row['baseline_n'],
                net_savings_gpu_units=None if cost is None or row['baseline_n'] is None else row['baseline_n']-cost))
    return output


def fmt(n):return '--' if n is None else str(n)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--corrections',type=Path,required=True)
    p.add_argument('--expansion',type=Path,required=True)
    args=p.parse_args();out=args.out.resolve();out.mkdir(parents=True,exist_ok=False)
    shutil.copy2(__file__,out/'runner.py')
    designs,integrity=load(args.input.resolve(),Corrections(args.corrections),Expansion(args.expansion));frontiers=minimum_frontiers(designs);scale=scale_frontiers(designs)
    original_frontiers=minimum_frontiers([d for d in designs if d['candidate_origin']=='original_static_pool'])
    original_lookup={(r['model'],r['gpu'],r['workload'],r['slo'],r['architecture']):r for r in original_frontiers}
    search_audit=[]
    for r in frontiers:
        old=original_lookup[(r['model'],r['gpu'],r['workload'],r['slo'],r['architecture'])]
        search_audit.append({k:r[k] for k in ('model','gpu','workload','slo','architecture')}|
            dict(original_baseline_n=old['baseline_n'],expanded_baseline_n=r['baseline_n'],
                 original_hbf_n=old['hbf_n'],expanded_hbf_n=r['hbf_n']))
    unique_baselines={(r['model'],r['gpu'],r['workload'],r['slo']):r for r in search_audit}
    integrity['search_audit']=dict(baseline_comparisons=len(unique_baselines),
        changed_baseline_minima=sum(r['original_baseline_n']!=r['expanded_baseline_n'] for r in unique_baselines.values()),
        changed_hbf_minima=sum(r['original_hbf_n']!=r['expanded_hbf_n'] for r in search_audit))
    prices=price_curves(designs,frontiers)
    (out/'aggregate.json').write_text(json.dumps(dict(designs=designs,minimum_frontiers=frontiers,scale_frontiers=scale,price_curves=prices),separators=(',',':'))+'\n')
    (out/'integrity.json').write_text(json.dumps(integrity,indent=2)+'\n')
    with (out/'minimum-frontiers.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(frontiers[0]));writer.writeheader();writer.writerows(frontiers)
    with (out/'scale-frontiers.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(scale[0]));writer.writeheader();writer.writerows(scale)
    with (out/'price-curves.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(prices[0]));writer.writeheader();writer.writerows(prices)
    with (out/'search-audit-frontiers.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(search_audit[0]));writer.writeheader();writer.writerows(search_audit)
    models=[m['id'] for m in catalog()]
    fig,axes=plt.subplots(1,2,figsize=(11,4.3),sharey=True)
    for ax,gpu in zip(axes,['rtx4090','h200']):
        for arch,offset,marker in [('baseline',-.18,'o'),('shared_site',0.,'s'),('h3_cascaded',.18,'^')]:
            rows=[next(r for r in frontiers if r['model']==m and r['gpu']==gpu and r['workload']=='short_chat' and r['slo']=='balanced' and r['architecture']==('shared_site' if arch=='baseline' else arch)) for m in models]
            ys=[r['baseline_n'] if arch=='baseline' else r['hbf_n'] for r in rows]
            ax.scatter([i+offset for i,y in enumerate(ys) if y is not None],[y for y in ys if y is not None],
                       color=COLORS[arch],marker=marker,label=NAMES[arch],s=45)
        ax.set_xticks(range(8),[SHORT[m] for m in models],rotation=45,ha='right')
        ax.set(title='RTX4090-scale projection' if gpu=='rtx4090' else 'H200 projection',yscale='log',ylim=(.8,140))
        ax.grid(axis='y',alpha=.2)
    axes[0].set_ylabel('Minimum evaluated installed GPUs');axes[1].legend(frameon=False,fontsize=8)
    fig.suptitle('Four fixed users; balanced short-chat SLO');fig.tight_layout();savefig(fig,out,'fixed-demand-frontier')
    fig,axes=plt.subplots(2,2,figsize=(10,7),sharex=True)
    for ax,m in zip(axes.flat,['qwen3_8b','qwen3_235b_a22b','llama31_405b','kimi_k2']):
        for arch in ARCHES:
            rows=sorted([r for r in scale if r['model']==m and r['gpu']=='h200' and r['workload']=='short_chat'
                         and r['slo']=='balanced' and r['architecture']==arch],key=lambda r:r['n'])
            ax.plot([r['n'] for r in rows],[np.nan if r['tested_useful_good_tokens_per_second'] is None else r['tested_useful_good_tokens_per_second'] for r in rows],
                    marker='o',markersize=4,color=COLORS[arch],label=NAMES[arch])
        ax.set(title=SHORT[m],xscale='log',yscale='log',xlabel='Installed GPUs',ylabel='Passing output tokens/s')
        ax.grid(alpha=.2)
    axes[0,0].legend(frameon=False,fontsize=8);fig.tight_layout();savefig(fig,out,'scale-frontiers')
    fig,axes=plt.subplots(1,2,figsize=(11,4.3),sharey=True)
    for ax,gpu in zip(axes,['rtx4090','h200']):
        for arch,offset in [('shared_site',-.12),('h3_cascaded',.12)]:
            vals=[];xs=[]
            for i,m in enumerate(models):
                r=next(r for r in frontiers if r['model']==m and r['gpu']==gpu and r['workload']=='short_chat' and r['slo']=='balanced' and r['architecture']==arch)
                if r['maximum_net_premium_gpu_units_per_TB'] is not None:xs.append(i+offset);vals.append(r['maximum_net_premium_gpu_units_per_TB'])
            ax.scatter(xs,vals,marker='s' if arch=='shared_site' else '^',label=NAMES[arch],color=COLORS[arch],s=45)
        ax.axhline(0,color='gray',linewidth=1);ax.set(title='RTX4090-scale' if gpu=='rtx4090' else 'H200')
        ax.set_xticks(range(8),[SHORT[m] for m in models],rotation=45,ha='right');ax.grid(axis='y',alpha=.2)
    axes[0].set_ylabel('Affordable net installed premium\n(GPU-equivalent units per installed TB)')
    axes[1].legend(frameon=False,fontsize=8);fig.tight_layout();savefig(fig,out,'price-headroom')
    fig,axes=plt.subplots(2,2,figsize=(10,7),sharex=True)
    for ax,(gpu,m) in zip(axes.flat,[('rtx4090','qwen3_32b'),('rtx4090','qwen3_235b_a22b'),('h200','llama31_405b'),('h200','kimi_k2')]):
        base=next(r for r in frontiers if r['gpu']==gpu and r['model']==m and r['workload']=='short_chat' and r['slo']=='balanced')['baseline_n']
        if base is not None:ax.axhline(base,color=COLORS['baseline'],label='Conventional minimum')
        for arch in ('shared_site','h3_cascaded'):
            points=sorted([r for r in prices if r['gpu']==gpu and r['model']==m and r['workload']=='short_chat' and r['slo']=='balanced' and r['architecture']==arch],key=lambda r:r['premium_gpu_units_per_stack'])
            ax.plot([r['premium_gpu_units_per_stack'] for r in points],
                [np.nan if r['hbf_cost_gpu_units'] is None else r['hbf_cost_gpu_units'] for r in points],
                marker='o',markersize=3,color=COLORS[arch],label=NAMES[arch])
        ax.set(title=gpu+' / '+SHORT[m],xlabel='Premium per installed stack (GPU units)',ylabel='Minimum evaluated cost (GPU units)')
        ax.set_xscale('symlog',linthresh=.01)
        ax.grid(alpha=.2)
    axes[0,0].legend(frameon=False,fontsize=8);fig.tight_layout();savefig(fig,out,'price-curves')
    statuses=Counter(r['classification'] for r in frontiers)
    report=['# When is HBF worth it? Conditional simulation results','',
        'These results compare the evaluated designs under the frozen workload, precision, service, cost and physical scenarios. They are not validated shipping-product forecasts.','',
        f"Coverage: {integrity['coverage_cells']} model/platform/workload/count cells; {integrity['raw_rows']:,} outcomes; {integrity['template_checks']['templates']:,} distinct complete event templates.",'',
        f"Fixed-four-user architecture/workload/SLO classifications: `{dict(statuses)}`.",'',
        'Conventional host RAM and eight SSDs are free in the GPU-equivalent cost control. The strongest conventional class includes a zero-arithmetic CPU bound. Unknown HBF price is represented by affordable total installed net premium. Missing references are not infinite savings.','',
        '![Minimum installed compute](fixed-demand-frontier.png)','','![Scale-dependent useful goodput](scale-frontiers.png)','','![Signed installed premium](price-headroom.png)','',
        'The machine-readable minimum and scale frontiers contain every evaluated case. Service-bound exclusions have no fabricated throughput. Static candidate screening limits optimality; public tool-shape useful-capacity comparisons are restricted to matched node-template widths.','']
    (out/'REPORT.md').write_text('\n'.join(report))
    (out/'output-manifest.json').write_text(json.dumps(dict(inputs=integrity['inputs'],
        outputs=[fingerprint(p) for p in sorted(out.iterdir()) if p.is_file()]),indent=2)+'\n')
    print('Analyzed',integrity['raw_rows'],'outcomes;',dict(statuses),flush=True)


if __name__=='__main__':main()
