"""Auditable overlay of corrected trajectories on immutable central results."""
import json
from pathlib import Path
from run_validation import ROOT,fingerprint


def key(row):return tuple(row[k] for k in ('model','gpu','workload','design_id','seed','view'))


class Corrections:
    def __init__(self,path):
        self.path=Path(path).resolve();self.manifest=json.loads((self.path/'manifest.json').read_text())
        if self.manifest['status']!='completed':raise ValueError('completed boundary correction audit required')
        for record in self.manifest['outputs']+self.manifest['sources']:
            if fingerprint(ROOT/record['path'])!=record:raise ValueError('changed correction lineage')
        self.rows={};self.used=set()
        for line in (self.path/'results.jsonl').read_text().splitlines():
            r=json.loads(line)
            if key(r) in self.rows:raise ValueError('duplicate correction')
            if r['status'] not in ('simulated_projection','conditional_service_bound_excluded'):raise ValueError('failed correction')
            self.rows[key(r)]=r

    def load(self,folder):
        folder=Path(folder).resolve();source=fingerprint(folder/'results.jsonl');output=[]
        for line in (folder/'results.jsonl').read_text().splitlines():
            r=json.loads(line);k=key(r)
            if k in self.rows:
                r=self.rows[k]
                if r['corrected_from']['source']!=source:raise ValueError('correction source differs')
                self.used.add(k);directory=self.path
            else:directory=folder
            output.append(dict(r,_run_directory=str(directory)))
        return output

    def complete(self):
        if self.used!=set(self.rows):raise ValueError('unmatched boundary correction')


class Expansion:
    def __init__(self,path):
        self.path=Path(path).resolve();self.manifest=json.loads((self.path/'manifest.json').read_text())
        if self.manifest['status']!='completed' or self.manifest['cells']!=720 or len(self.manifest['jobs'])!=16:
            raise ValueError('complete fixed-demand expansion required')
        self.jobs={};self.inputs=[fingerprint(self.path/'manifest.json')]
        for job in self.manifest['jobs']:
            if fingerprint(ROOT/job['manifest']['path'])!=job['manifest']:raise ValueError('changed expansion worker')
            folder=self.path/job['id'];record=json.loads((folder/'manifest.json').read_text())
            if record['status']!='completed' or record['cells']!=45:raise ValueError('incomplete expansion worker')
            for item in record['sources']+record['outputs']:
                if fingerprint(ROOT/item['path'])!=item:raise ValueError('changed expansion source/output')
            self.jobs[job['id']]=folder;self.inputs.extend(record['outputs'])

    def load(self,name):
        folder=self.jobs[name];rows=[]
        for line in (folder/'results.jsonl').read_text().splitlines():
            row=json.loads(line)
            if row['status'] not in ('simulated_projection','conditional_service_bound_excluded') or row['view']!='fixed_users':
                raise ValueError('failed or wrong-domain expansion')
            rows.append(dict(row,_run_directory=str(folder)))
        return rows
