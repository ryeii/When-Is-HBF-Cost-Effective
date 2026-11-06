"""Finite-scenario stability, exact-route intervention and lifetime arithmetic."""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
import csv
import gzip
import json
from pathlib import Path
import shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from analyze_deployments import SHORT,NAMES,verify,savefig
from run_validation import ROOT,fingerprint
from result_rows import Corrections,Expansion


def readlines(path):return [json.loads(line) for line in path.read_text().splitlines()]


def passes(row,slo):
    return row['status']=='simulated_projection' and row['metrics'][slo]['slo_pass']


def frontiers(rows,scenario):
    output=[]
    for slo in ('responsive','balanced','relaxed'):
        base=[r for r in rows if r['design']['layout']['architecture']=='baseline' and passes(r,slo)]
        bn=min((r['design']['layout']['installed_gpus'] for r in base),default=None)
        for arch in ('shared_site','h3_cascaded'):
            pool=[r for r in rows if r['design']['layout']['architecture']==arch and passes(r,slo)]
            hn=min((r['design']['layout']['installed_gpus'] for r in pool),default=None)
            price=max(((bn-r['design']['layout']['installed_gpus'])/r['installed_hbf_TB'] for r in pool),default=None) if bn is not None else None
            output.append(dict(scenario=scenario,slo=slo,architecture=arch,baseline_n=bn,hbf_n=hn,
                premium_gpu_units=None if bn is None or hn is None else bn-hn,premium_per_TB=price,
                classification='no_reference' if bn is None else 'no_hbf' if hn is None else 'positive' if bn>hn else 'no_positive_headroom'))
    return output


def write_csv(path,rows):
    with path.open('w') as stream:
        w=csv.DictWriter(stream,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def corners(root,out,corrections,expansion):
    top=json.loads((root/'manifest.json').read_text())
    if top['status']!='completed' or len(top['jobs'])!=6:raise ValueError('six complete sensitivity cases required')
    results=[];integrity=[];case_order=[]
    for job in top['jobs']:
        if job['status']!='pass':raise ValueError('failed sensitivity worker')
        verify(job['receipt']);folder=root/job['id'];manifest=json.loads((folder/'manifest.json').read_text())
        if manifest['status']!='completed' or manifest['failures']:raise ValueError('failed sensitivity row')
        for record in manifest['source_inputs']:verify(record)
        for r in manifest['outputs']:verify(r)
        verify(manifest['source']);verify(manifest['expansion_source'])
        central_folder=(ROOT/manifest['source']['path']).parent
        central=corrections.load(central_folder)+expansion.load(central_folder.name)
        central=[r for r in central if r['seed']==17 and r['view']=='fixed_users' and r['workload']==manifest['workload']]
        rows=readlines(folder/'results.jsonl');case=(manifest['model'],manifest['gpu'],manifest['workload']);case_order.append(case)
        expected=len(central)
        for name in manifest['scenarios']:
            selected=[r for r in rows if r['scenario']==name]
            if len(selected)!=expected:raise ValueError('sensitivity candidate coverage mismatch')
            if {r['design']['layout']['installed_gpus'] for r in selected}!={1,2,4,8,16,32,64,72,100}:raise ValueError('missing count')
            if any(r['status'] not in ('simulated_projection','conditional_service_bound_excluded') for r in selected):raise ValueError('failure is not an exclusion')
            for row in frontiers(selected,name):results.append(dict(model=case[0],gpu=case[1],workload=case[2],**row))
        for row in frontiers(central,'central_seed17'):results.append(dict(model=case[0],gpu=case[1],workload=case[2],**row))
        for path in sorted((folder/'runs').glob('*.gz')):
            with gzip.open(path,'rt') as f:run=json.load(f)
            if run['completed_requests']!=len(run['requests']):raise ValueError('incomplete sensitivity trajectory')
            if any(len(r['token_times_s'])!=r['expected_output_tokens'] for r in run['requests']):raise ValueError('missing token')
            if any(r['utilization']>1.000001 for r in run['resources'].values()):raise ValueError('resource overbooked')
        integrity.append(dict(case=case,rows=len(rows),statuses=dict(Counter(r['status'] for r in rows)),
            reused_central=sum('reused_central_source' in r for r in rows),source=fingerprint(folder/'results.jsonl')))
    write_csv(out/'sensitivity-frontiers.csv',results)
    case_order=sorted(case_order,key=lambda c:(c[1],c[0],c[2]))
    scenarios=['central_seed17',*json.loads((root/top['jobs'][0]['id']/'manifest.json').read_text())['scenarios']]
    fig,axes=plt.subplots(2,1,figsize=(9,7.8),sharex=True)
    finite=[abs(r['premium_gpu_units']) for r in results if r['slo']=='balanced' and r['premium_gpu_units'] is not None]
    extent=max([1,*finite])
    short_labels=['Central','64 GB/s','256 GB/s','HBF slow','Conv. fast','Skew','Write slow','Write fast','10 W','Combined']
    for ax,arch in zip(axes,['shared_site','h3_cascaded']):
        matrix=[];labels=[]
        for case in case_order:
            matched=[r for r in results if (r['model'],r['gpu'],r['workload'])==case and r['architecture']==arch and r['slo']=='balanced']
            matrix.append([next(r for r in matched if r['scenario']==s)['premium_gpu_units'] for s in scenarios])
            labels.append(('R/' if case[1]=='rtx4090' else 'H/')+SHORT[case[0]]+' '+case[2].replace('_',' '))
        values=np.array([[np.nan if v is None else v for v in row] for row in matrix])
        plot=ax.imshow(values,cmap='RdBu',vmin=-extent,vmax=extent,aspect='auto')
        for i in range(len(matrix)):
            for j in range(len(scenarios)):ax.text(j,i,'--' if matrix[i][j] is None else str(matrix[i][j]),ha='center',va='center',fontsize=9)
        ax.set_xticks(range(len(scenarios)),short_labels,rotation=45,ha='right',fontsize=9)
        ax.set_yticks(range(len(case_order)),labels,fontsize=9);ax.set_title(NAMES[arch])
    fig.suptitle('Balanced four-user total premium ceiling (GPU units)\n-- lacks a feasible reference or HBF design',fontsize=12)
    fig.tight_layout();savefig(fig,out,'sensitivity-frontiers')
    all_groups=defaultdict(list)
    for r in results:all_groups[(r['model'],r['gpu'],r['workload'],r['slo'],r['architecture'])].append(r)
    stable=[key for key,rows in all_groups.items() if all(r['classification']=='positive' for r in rows)]
    return results,integrity,stable


def routing(root,central,out):
    manifest=json.loads((root/'manifest.json').read_text())
    if manifest['status']!='completed':raise ValueError('route intervention incomplete')
    for r in manifest['inputs']+manifest['outputs']:verify(r)
    exact=readlines(root/'results.jsonl');comparisons=[]
    for gpu in ('rtx4090','h200'):
        source=central/f'qwen3_30b_a3b-{gpu}'/'results.jsonl'
        mean={r['design_id']:r for r in readlines(source) if r['seed']==17 and r['view']=='fixed_users' and r['workload']=='short_chat'}
        for r in exact:
            if r['gpu']!=gpu:continue
            other=mean[r['design_id']]
            comparisons.append(dict(gpu=gpu,n=r['design']['layout']['installed_gpus'],architecture=r['design']['layout']['architecture'],
                design_id=r['design_id'],meanfield_status=other['status'],exact_status=r['status'],
                meanfield_weight_read_bytes=other.get('traffic',{}).get('weight_read_bytes'),exact_weight_read_bytes=r['traffic']['weight_read_bytes'],
                meanfield_balanced_pass=passes(other,'balanced'),exact_balanced_pass=passes(r,'balanced'),
                meanfield_elapsed_s=other.get('elapsed_s'),exact_elapsed_s=r['elapsed_s']))
    write_csv(out/'route-intervention.csv',comparisons)
    return comparisons


def lifetime(central,report,out,corrections,expansion):
    aggregate=json.loads((report/'aggregate.json').read_text());selected={}
    # All feasible minimum-GPU fixed-demand HBF witnesses, not only favorable cases.
    for r in aggregate['minimum_frontiers']:
        if r['hbf_design'] is not None:selected[(r['model'],r['gpu'],r['workload'],r['hbf_design'])]=r
    source_rows={};records=[]
    for model,gpu,workload,did in selected:
        key=(model,gpu)
        if key not in source_rows:
            source_rows[key]={r['design_id']+'-'+r['workload']:r for r in corrections.load(central/f'{model}-{gpu}')+expansion.load(f'{model}-{gpu}')
                if r['seed']==17 and r['view']=='fixed_users'}
        row=source_rows[key][did+'-'+workload];d=row['design'];l=d['layout']
        capacity=row['installed_hbf_stacks']*512*2**30
        retained=0
        for t in row['templates']:
            with gzip.open(Path(row['_run_directory'])/'runs'/t['file'],'rt') as f:run=json.load(f)
            local=run['design']['layout']
            retained+=sum(p['cold_weight_bytes'] for p in run['memory_plans'])*local['tp']*local['replicas']*t['copies']
        logical_rate=row['traffic']['prefix_write_logical_bytes']/row['elapsed_s']
        # Finite fresh-prefix-rate extrapolation; retained copies otherwise persist.
        media_j=row['traffic']['hbf_read_joules']+row['traffic']['hbf_program_joules']
        for pe in (1000,3000,10000):
            for hours in (6,24,168):
                for wa in (1.,1.3,4.):
                    refresh_rate=retained/(hours*3600)
                    physical_rate=wa*(logical_rate+refresh_rate)
                    years=pe*capacity/(physical_rate*365.25*86400) if physical_rate else None
                    records.append(dict(model=model,gpu=gpu,workload=workload,design_id=did,
                        architecture=l['architecture'],installed_gpus=l['installed_gpus'],installed_stacks=row['installed_hbf_stacks'],
                        installed_bytes=capacity,retained_weight_bytes=retained,logical_prefix_bytes_per_second=logical_rate,
                        refresh_hours=hours,pe_budget=pe,write_amplification=wa,ideal_uniform_wear_years=years,
                        central_replay_media_plus_assumed_static_watts=media_j/row['elapsed_s']+row['installed_hbf_stacks']*2))
    if not records:raise ValueError('no feasible HBF design for lifetime arithmetic')
    write_csv(out/'lifetime-envelopes.csv',records)
    return records


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('input','routes','central','report','out','corrections','expansion'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();out=a.out.resolve();out.mkdir(parents=True,exist_ok=False);shutil.copy2(__file__,out/'runner.py')
    corrections=Corrections(a.corrections);expansion=Expansion(a.expansion)
    results,integrity,stable=corners(a.input.resolve(),out,corrections,expansion)
    route=routing(a.routes.resolve(),a.central.resolve(),out)
    life=lifetime(a.central.resolve(),a.report.resolve(),out,corrections,expansion)
    (out/'analysis.json').write_text(json.dumps(dict(frontiers=results,integrity=integrity,stable_positive=stable,
        route_comparisons=len(route),lifetime_envelopes=len(life)),indent=2)+'\n')
    (out/'REPORT.md').write_text('# Conditional sensitivity results\n\n'+
        f'{len(results)} architecture/SLO/scenario records; {len(stable)} positive comparisons survive every tested corner. '+
        f'{len(route)} matched correlated-route interventions; {len(life)} lifetime arithmetic envelopes.\n\n'+
        'Finite scenario stability is not a confidence interval. All corner frontiers, route changes and lifetime assumptions are retained in CSV.\n\n'+
        '![Finite-scenario frontiers](sensitivity-frontiers.png)\n')
    (out/'output-manifest.json').write_text(json.dumps(dict(inputs=[fingerprint(a.input/'manifest.json'),fingerprint(a.routes/'manifest.json'),fingerprint(a.report/'aggregate.json')],
        outputs=[fingerprint(path) for path in sorted(out.iterdir()) if path.is_file()]),indent=2)+'\n')


if __name__=='__main__':main()
