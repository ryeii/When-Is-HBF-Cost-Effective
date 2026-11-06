"""Bound execution-host memory without changing simulated data or clocks."""
from collections import OrderedDict
from contextlib import contextmanager
import fcntl
import gc
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys
import uuid
from run_validation import ROOT,HERE


def compact(records):
    # Interning changes object sharing, never block values/publication state.
    for r in records:
        for field in ('input_blocks','completed_blocks'):
            r[field]=[sys.intern(value) for value in r[field]]
    return records


class DiskRuns:
    def __init__(self,out):self.out=Path(out);self.index={};self.memory=OrderedDict()
    def __contains__(self,key):return key in self.index
    def __setitem__(self,key,value):
        run,name=value;self.index[key]=name;self.memory[key]=value
        while len(self.memory)>1:self.memory.popitem(last=False)
    def __getitem__(self,key):
        if key in self.memory:return self.memory[key]
        name=self.index[key]
        with gzip.open(self.out/'runs'/name,'rt') as stream:run=json.load(stream)
        self.memory.clear();self.memory[key]=(run,name)
        return run,name
    def drop_memory(self):self.memory.clear()
    def clear(self):self.memory.clear();self.index.clear()


class DiskPrograms:
    def __init__(self,provider):
        self.provider=provider;self.memory={}
        self.root=ROOT/'.cache/hbf-adoption/immutable-workload-cache-v1';self.root.mkdir(parents=True,exist_ok=True)
        inputs=[HERE/'projection_workloads.py',HERE/'prefix_workloads.py']
        trace=HERE/'results/causal-workloads-v2-20261104/causal-shapes.jsonl.gz'
        if not trace.exists():trace=next(trace.parent.glob('*.gz'))
        inputs.append(trace)
        self.namespace=hashlib.sha256(b''.join(hashlib.sha256(p.read_bytes()).digest() for p in inputs)).hexdigest()
    def __contains__(self,key):return True  # A deterministic lazy data source.
    @contextmanager
    def lock(self):
        with (self.root/'materialization.lock').open('a+b') as stream:
            fcntl.flock(stream.fileno(),fcntl.LOCK_EX)
            try:yield
            finally:fcntl.flock(stream.fileno(),fcntl.LOCK_UN)
    def __getitem__(self,key):
        if key in self.memory:return self.memory[key]
        name=hashlib.sha256(json.dumps([self.namespace,*key],separators=(',',':')).encode()).hexdigest()
        path=self.root/(name+'.json.gz');digest=path.with_suffix('.sha256')
        with self.lock():
            valid=path.exists() and digest.exists()
            if valid:
                # A checksum precedes parsing, and gzip also checks its CRC.
                valid=hashlib.sha256(path.read_bytes()).hexdigest()==digest.read_text().strip()
            if valid:
                with gzip.open(path,'rt') as stream:records=compact(json.load(stream))
            else:
                records=compact(self.provider(*key))
                temp=self.root/(name+'.'+str(os.getpid())+'.'+uuid.uuid4().hex+'.tmp')
                payload=json.dumps(records,separators=(',',':')).encode()
                with gzip.GzipFile(filename=str(temp),mode='wb',compresslevel=1,mtime=0) as stream:stream.write(payload)
                del payload
                checksum=hashlib.sha256(temp.read_bytes()).hexdigest()
                temp.replace(path);digest.write_text(checksum+'\n')
        self.memory.clear();self.memory[key]=records
        return records
    def drop_memory(self):self.memory.clear()
    def clear(self):self.memory.clear()


def install_runner_cache():
    import projection_sweep as sweep
    old_init=sweep.Runner.__init__;old_execute=sweep.Runner.execute
    def init(self,out):
        old_init(self,out);self.cache=DiskRuns(out)
        self.workload_cache=DiskPrograms(lambda *key:sweep.make(*key))
    def execute(self,*args,**kwargs):
        try:return old_execute(self,*args,**kwargs)
        finally:
            self.cache.drop_memory();self.workload_cache.drop_memory();gc.collect()
    sweep.Runner.__init__=init;sweep.Runner.execute=execute
