"""Conditional multi-node event simulation with native resident GPU timings.

This is a new composite model, not a validation claim for any upstream
simulator. All clocks, per-node contention, finite KV/staging allocations and
causal transfers are explicit. Routing traffic is a declared mean-field model;
the separate released-route replay measures the error of that assumption.
"""
from __future__ import annotations
from collections import OrderedDict,deque
from dataclasses import dataclass,asdict
from functools import lru_cache
import math
import numpy as np

from model_catalog import kv_bytes
from projection_timing import Layout,DeviceService,layer_parameters,GiB
from projection_native import NativeTiming
from projection_resources import Events,Resource,Hop,chain,StreamedOperator
from metrics import assess_requests,SLOS
from evidence import quantile
from run_validation import HERE


@dataclass(frozen=True)
class Design:
    layout:Layout
    gpu:str
    comparator:str='gpu_staging'
    host_gib:int=1024
    drives:int=8
    batch_cap:int=4
    chunk:int=512
    prefix:bool=True
    slots:int=2
    gpu_scale:float=1.
    io_scale:float=1.
    hbf_link_Bps:float=2e12
    thermal_watts:float=20.
    write_amplification:float=1.3
    route_mode:str='default'
    cold_compute_scale:float=1.25
    host_Bps:float=0.
    ssd_Bps_per_drive:float=2.825e9


@lru_cache(maxsize=1)
def qwen_inclusion():
    path=HERE/'results/routing-20261104/general.npz'
    with np.load(path) as data:
        arrays=[data[k] for k in data if data[k].ndim==3]
        if len(arrays)!=1:raise ValueError('ambiguous released routing array')
        route=arrays[0]
    if route.shape!=(48,8000,8):raise ValueError('unexpected routing geometry')
    result=[]
    for layer in range(48):
        counts=np.bincount(route[layer,:2000].reshape(-1),minlength=128)/2000
        result.append(np.sort(counts)[::-1])
    return np.asarray(result)


@lru_cache(maxsize=10000)
def inclusion(experts,topk,mode,layer):
    if mode=='empirical_qwen':return qwen_inclusion()[layer]
    if mode=='uniform':return np.full(experts,topk/experts)
    if mode=='skewed':
        p=1/np.arange(1,experts+1,dtype=float);p/=p.sum()
        low,high=0.,float(experts*topk)
        for _ in range(60):
            middle=(low+high)/2
            if (1-(1-p)**middle).sum()<topk:low=middle
            else:high=middle
        return 1-(1-p)**((low+high)/2)
    raise ValueError('unknown routing scenario')


class Cache:
    def __init__(self,capacity):
        self.capacity=max(0,int(capacity));self.rows=OrderedDict();self.evictions=0

    def has(self,key):
        if key not in self.rows:return False
        self.rows.move_to_end(key);return True

    def put(self,key):
        if not self.capacity:return
        self.rows[key]=True;self.rows.move_to_end(key)
        while len(self.rows)>self.capacity:self.rows.popitem(last=False);self.evictions+=1


class CapacityError(ValueError):pass


class Simulator:
    def __init__(self,model,design,requests,*,capture=False):
        self.model=model;self.d=design;self.layout=design.layout
        self.events=Events();self.timings=NativeTiming(model,design.gpu,design.layout,gpu_scale=design.gpu_scale)
        self.device=DeviceService(stacks=self.layout.hbf_stacks,link_Bps=design.hbf_link_Bps,
                                  read_scale=1/design.io_scale,thermal_watts_per_stack=design.thermal_watts)
        self.requests={r['id']:{**r,'arrival_s':None,'token_times_s':[],
            'expected_output_tokens':r['output_tokens'],'processed':0,'replica':None} for r in requests}
        self.successors={r['predecessor']:r['id'] for r in requests if r['predecessor'] is not None}
        self.resources={};self.capture=capture
        self.stats=dict(weight_read_bytes=0.,prefix_read_bytes=0.,prefix_write_logical_bytes=0.,
                        flash_program_physical_bytes=0.,host_read_bytes=0.,ssd_read_bytes=0.,
                        gpu_staging_bytes=0.,network_bytes=0.,prefix_hit_tokens=0,offered_input_tokens=0,
                        hbf_read_joules=0.,hbf_program_joules=0.,max_outstanding_chunks=0)
        self.source_counts={}
        self.plans=[];self.kv_used=[0.]*self.layout.replicas
        self.waiting=[deque() for _ in range(self.layout.replicas)]
        self.ready=[deque() for _ in range(self.layout.replicas)]
        self.stage_queues=[[deque() for _ in range(self.layout.pp)] for _ in range(self.layout.replicas)]
        self.stage_busy=[[False]*self.layout.pp for _ in range(self.layout.replicas)]
        self.inflight=[0]*self.layout.replicas
        self.save_queues=[[deque() for _ in range(self.layout.pp)] for _ in range(self.layout.replicas)]
        self.save_busy=[[False]*self.layout.pp for _ in range(self.layout.replicas)]
        self.near_caches=[];self.far_caches=[];self.pending_prefix=[set() for _ in range(self.layout.replicas)]
        self._memory_plan(requests)
        self._host_plan()
        for r in self.requests.values():
            if r['predecessor'] is None:self.events.at(r['root_arrival_s'],self.arrive,r['id'])

    def resource(self,name):
        if name not in self.resources:self.resources[name]=Resource(self.events,name,capture=self.capture)
        return self.resources[name]

    def _memory_plan(self,requests):
        m=self.model;d=self.d;l=self.layout;t=self.timings
        largest=max(r['input_tokens']+r['output_tokens'] for r in requests)
        self.layer_plan={}
        for stage in range(l.pp):
            layers=list(l.layers(m,stage))
            # NVIDIA's released H200 system description reserves 3.5 GiB
            # non-KV runtime memory plus up to 392 MiB for NCCL. Reuse that
            # conservative reservation, with explicit activation queues.
            max_chunk=min(d.chunk,max(r['input_tokens'] for r in requests))
            workspace=3.5*GiB+(392*2**20 if l.tp>1 else 0)+(2+l.pp)*d.batch_cap*max_chunk*m['hidden']*2
            buffers=(d.slots*128+64)*2**20
            near=t.near_capacity;available=near-workspace-buffers
            if available<=0:raise CapacityError('workspace and bounded staging exceed near capacity')
            one=kv_bytes(m,largest,l.tp,l.pp)
            prefix_budget=.10*available if d.prefix else 0.
            kv_budget=min(available-prefix_budget,max(.20*available,one*d.batch_cap))
            if one>kv_budget:raise CapacityError('one request active KV cannot fit the reserved near tier')
            weight_budget=max(0,available-prefix_budget-kv_budget)
            weight_bytes=t.stage_weight_bytes(stage)
            # Boundary embeddings/logits are pinned before uniform per-layer
            # shared tensors; any residual shared bytes are streamed causally.
            layer_bytes=sum(2*sum(layer_parameters(m,i,l.tp)) for i in layers)
            boundary=max(0,weight_bytes-layer_bytes)
            pinned=min(boundary,weight_budget);remaining=weight_budget-pinned
            shared=sum(2*sum(layer_parameters(m,i,l.tp)[:2]) for i in layers)
            shared_fraction=min(1.,remaining/shared) if shared else 1.
            remaining=max(0,remaining-shared)
            moe_count=sum(i>=m['dense_layers'] for i in layers) if m['family']=='moe' else 0
            expert_bytes=2*m['expert_parameters']/l.tp
            cache_experts=min(m['experts'],int(remaining/(moe_count*expert_bytes))) if moe_count and expert_bytes else 0
            resident=pinned+shared*shared_fraction+cache_experts*expert_bytes*moe_count
            cold=max(0,weight_bytes-resident)
            if d.comparator=='resident' and cold>4096:raise CapacityError('resident design exceeds near-memory capacity')
            if l.hbf_stacks and cold>self.device.capacity_bytes-64*2**20:
                raise CapacityError('nonresident weights exceed installed HBF capacity')
            block=kv_bytes(m,16,l.tp,l.pp)
            far_capacity=max(0,self.device.capacity_bytes-cold-64*2**20) if l.hbf_stacks else d.drives*3.84e12
            # Bound retention to 64 GiB/rank in the main design. Flash capacity
            # outside this allocation remains installed and charged.
            far_prefix=min(64*GiB,far_capacity) if d.prefix else 0
            self.plans.append(dict(stage=stage,near_bytes=near,workspace_bytes=workspace,
                staging_bytes=buffers,active_kv_bytes=kv_budget,prefix_near_bytes=prefix_budget,
                weight_bytes=weight_bytes,resident_weight_bytes=resident,cold_weight_bytes=cold,
                shared_resident_fraction=shared_fraction,cached_experts=cache_experts,
                boundary_cold_bytes=boundary-pinned,far_prefix_bytes=far_prefix,
                near_blocks=int(prefix_budget//block),far_blocks=int(far_prefix//block)))
            for i in layers:
                attn,ffn,routed=layer_parameters(m,i,l.tp)
                self.layer_plan[i]=(2*attn*(1-shared_fraction),2*ffn*(1-shared_fraction),cache_experts,expert_bytes)
        self.kv_capacity=min(p['active_kv_bytes'] for p in self.plans)
        near_blocks=min(p['near_blocks'] for p in self.plans)
        far_blocks=min(p['far_blocks'] for p in self.plans)
        self.near_caches=[Cache(near_blocks) for _ in range(l.replicas)]
        self.far_caches=[Cache(far_blocks) for _ in range(l.replicas)]

    def _host_plan(self):
        l=self.layout
        stages=[set() for _ in range(l.nodes)]
        for replica in range(l.replicas):
            for stage in range(l.pp):stages[l.node(replica,stage)].add(stage)
        # Identical replicas can share immutable host copies. Distinct PP/TP
        # shards occupy bytes once per node, rather than once per replica.
        self.host_fraction=[];self.host_prefix_fraction=[]
        for node,rows in enumerate(stages):
            demand=sum(self.plans[s]['cold_weight_bytes']*l.tp for s in rows)
            self.host_fraction.append(min(1.,self.d.host_gib*GiB/demand) if demand else 1.)
            free=max(0,self.d.host_gib*GiB-demand)
            prefixes=0.
            for replica in range(l.replicas):
                count=sum(len(r['completed_blocks']) for r in self.requests.values()
                          if r['session']%l.replicas==replica)
                for stage in range(l.pp):
                    if l.node(replica,stage)==node:
                        prefixes+=min(kv_bytes(self.model,count*16,l.tp,l.pp),self.plans[stage]['far_prefix_bytes'])*l.tp
            self.host_prefix_fraction.append(min(1.,free/prefixes) if prefixes else 1.)

    def _route(self,layer,tokens):
        m=self.model;p=self.layer_plan[layer]
        if m['family']!='moe' or layer<m['dense_layers']:return 0.,0.
        mode=self.d.route_mode
        if mode=='default':mode='empirical_qwen' if m['id']=='qwen3_30b_a3b' else 'uniform'
        probabilities=inclusion(m['experts'],m['top_k'],mode,layer)
        cache=p[2]
        unique=float((1-(1-probabilities[cache:])**tokens).sum())
        fraction=float(probabilities[cache:].sum()/m['top_k'])
        return unique*p[3],fraction

    def _source_hops(self,replica,stage,nbytes,*,weight=True,write=False,cpu_activation_bytes=None):
        d=self.d;l=self.layout;node=l.node(replica,stage);tp=l.tp
        nbytes=math.ceil(nbytes/4096)*4096 if nbytes else 0
        if not nbytes:return []
        if l.hbf_stacks:
            if write:
                physical=nbytes*d.write_amplification
                self.stats['flash_program_physical_bytes']+=physical*tp
                self.stats['hbf_program_joules']+=physical*tp*self.device.write_energy_j_per_byte
            else:
                self.stats['hbf_read_joules']+=nbytes*tp*self.device.read_energy_j_per_byte
            return [Hop(self.resource(f'hbf:{replica}:{stage}'),
                        self.device.seconds(nbytes*(d.write_amplification if write else 1),write=write))]
        if d.comparator=='resident' and weight:raise AssertionError('resident configuration attempted an offload')
        aggregate=nbytes*tp
        host_fraction=self.host_fraction[node] if weight else self.host_prefix_fraction[node]
        missing=aggregate*(1-host_fraction)
        host_bw=d.host_Bps or (100e9 if d.gpu=='rtx4090' else 460.8e9)
        ingress=self.timings.gpu['ingress_Bps']*8
        hops=[]
        if write:
            # Prefix save goes through the node's shared ingress and host
            # memory before durable SSD programming; no free host KV pool.
            hops=[Hop(self.resource(f'pcie:{node}'),aggregate/ingress),
                  Hop(self.resource(f'dram:{node}'),aggregate/host_bw)]
            if missing:hops.append(Hop(self.resource(f'ssd:{node}'),missing/(d.drives*d.ssd_Bps_per_drive*.7)+16e-6))
        else:
            self.stats['host_read_bytes']+=aggregate-missing
            self.stats['ssd_read_bytes']+=missing
            if missing:hops.append(Hop(self.resource(f'ssd:{node}'),missing/(d.drives*d.ssd_Bps_per_drive)*d.io_scale+16e-6))
            hops.append(Hop(self.resource(f'dram:{node}'),(aggregate+missing)/host_bw*d.io_scale))
            transfer=(cpu_activation_bytes if d.comparator=='cpu_oracle' and weight
                      and cpu_activation_bytes is not None else aggregate)
            # Private GPU ingress and shared node ingress both constrain TP.
            hops.append(Hop(self.resource(f'pcie:{node}'),transfer/ingress*d.io_scale))
            private=transfer/tp/self.timings.gpu['ingress_Bps']
            hops.append(Hop(self.resource(f'ingress:{replica}:{stage}'),private*d.io_scale))
        return hops

    def arrive(self,rid):
        r=self.requests[rid]
        if r['arrival_s'] is not None:raise AssertionError('request released twice')
        r['arrival_s']=self.events.now
        # Stable session affinity; initial sessions spread across replicas.
        replica=r['session']%self.layout.replicas;r['replica']=replica
        self.waiting[replica].append(rid);self.stats['offered_input_tokens']+=r['input_tokens']
        self.admit(replica)

    def admit(self,replica):
        queue=self.waiting[replica]
        while queue:
            r=self.requests[queue[0]]
            needed=kv_bytes(self.model,r['input_tokens']+r['output_tokens'],self.layout.tp,self.layout.pp)
            if self.kv_used[replica]+needed>self.kv_capacity:break
            queue.popleft();self.kv_used[replica]+=needed;r['kv_reservation']=needed
            r['kv_admitted_s']=self.events.now
            hit=0;far=0
            if self.d.prefix:
                for key in r['input_blocks'][:max(0,(r['input_tokens']-1)//16)]:
                    if self.near_caches[replica].has(key):hit+=1
                    elif self.far_caches[replica].has(key):hit+=1;far+=1
                    else:break
            r['processed']=hit*16;self.stats['prefix_hit_tokens']+=hit*16
            if far:self.restore(replica,r['id'],far)
            else:self.enqueue_ready(replica,r['id'])

    def restore(self,replica,rid,blocks):
        remaining=[self.layout.pp]
        def done():
            remaining[0]-=1
            if not remaining[0]:self.enqueue_ready(replica,rid)
        for stage in range(self.layout.pp):
            nbytes=kv_bytes(self.model,blocks*16,self.layout.tp,self.layout.pp)
            self.stats['prefix_read_bytes']+=nbytes*self.layout.tp
            hops=self._source_hops(replica,stage,nbytes,weight=False)
            hops.append(Hop(self.resource(f'gpu:{replica}:{stage}'),nbytes/self.timings.near_Bps))
            self.stats['gpu_staging_bytes']+=nbytes*self.layout.tp
            chain(self.events,hops,done)

    def enqueue_ready(self,replica,rid):
        self.ready[replica].append(rid)
        # Zero-delay dispatch event coalesces arrivals at the same clock while
        # imposing no artificial batching wait.
        self.events.at(self.events.now,self.dispatch,replica)

    def dispatch(self,replica):
        if (self.stage_busy[replica][0] or not self.ready[replica]
                or self.inflight[replica]>=self.layout.pp+1):return
        first=self.requests[self.ready[replica][0]]
        prefill=not first['token_times_s']
        selected=[];retained=deque()
        while self.ready[replica]:
            rid=self.ready[replica].popleft();r=self.requests[rid]
            if len(selected)<self.d.batch_cap and (not r['token_times_s'])==prefill:selected.append(rid)
            else:retained.append(rid)
        self.ready[replica]=retained
        queries=[min(self.d.chunk,self.requests[rid]['input_tokens']-self.requests[rid]['processed']) if prefill else 1 for rid in selected]
        context=max(self.requests[rid]['processed']+q if prefill else
                    self.requests[rid]['input_tokens']+len(self.requests[rid]['token_times_s']) for rid,q in zip(selected,queries))
        query=max(queries)
        components,sources=self.timings.components(prefill,len(selected),query,context)
        if prefill:
            terminal=sum(self.requests[rid]['processed']+q>=self.requests[rid]['input_tokens']
                         for rid,q in zip(selected,queries))
            if not terminal:components['logits']=0.
            elif terminal<len(selected):
                final,_=self.timings.components(False,terminal,1,context)
                components['logits']=final['logits']
        for key,count in sources.items():self.source_counts[key]=self.source_counts.get(key,0)+count
        batch=dict(requests=selected,queries=queries,prefill=prefill,context=context,
                   query=query,components=components,tokens=sum(queries),replica=replica)
        self.stage_busy[replica][0]=True
        self.inflight[replica]+=1
        cpu=self.resource(f'cpu:{self.layout.node(replica,0)}')
        cpu.submit(self.timings.cpu_seconds(prefill),self.stage_begin,batch,0)

    def stage_begin(self,batch,stage):
        replica=batch['replica']
        if stage and self.stage_busy[replica][stage]:
            self.stage_queues[replica][stage].append(batch);return
        self.stage_busy[replica][stage]=True
        batch_stage=dict(batch,stage=stage,layer_list=list(self.layout.layers(self.model,stage)),layer_index=0)
        boundary=self.plans[stage]['boundary_cold_bytes']
        gpu=self.resource(f'gpu:{replica}:{stage}')
        # Embedding/logits shared tensors are loaded at their real boundary.
        embedding=batch['components']['embedding'] if stage==0 else 0.
        if boundary:
            self.stream(batch_stage,boundary,embedding,0.,lambda:self.next_layer(batch_stage),cpu_allowed=False)
        else:gpu.submit(embedding,self.next_layer,batch_stage)

    def next_layer(self,batch):
        index=batch['layer_index']
        if index==len(batch['layer_list']):self.stage_finish(batch);return
        layer=batch['layer_list'][index];batch['layer_index']+=1
        cold_attn,cold_ffn,cache,expert_bytes=self.layer_plan[layer]
        routed,fraction=self._route(layer,batch['tokens'])
        # Costs are accumulated model-wide by the native operator graph. PP
        # redistributes those totals uniformly across physical layers. This is
        # exact for uniform layers, an exposed approximation for dense-prefix
        # MoE families; their whole-model total remains unchanged.
        attention=batch['components']['attention']/self.model['layers']
        ffn=batch['components']['ffn']/self.model['layers']
        # A smaller near tier can increase the resident memory-bound fraction.
        # Preserve native compute cost; enforce fitted near bandwidth as a
        # separate lower bound instead of adding the normal HBM read again.
        attn_p,ffn_p,routed_p=layer_parameters(self.model,layer,self.layout.tp)
        local_kv=kv_bytes(self.model,batch['context'],self.layout.tp,self.layout.pp)/math.ceil(self.model['layers']/self.layout.pp)
        attention=max(attention,(2*attn_p+local_kv*len(batch['requests']))/self.timings.near_Bps)
        active_routed=routed if routed else 0.
        ffn=max(ffn,(2*ffn_p+active_routed)/self.timings.near_Bps)
        if batch['prefill']:
            floor=self.timings.eager_layer_floor(True)
            if attention+ffn<floor:attention+=floor-attention-ffn
        def after_attention():
            self.stream(batch,cold_ffn+routed,ffn,fraction if not cold_ffn else 1.,
                        lambda:self.next_layer(batch),cpu_allowed=routed>0)
        self.stream(batch,cold_attn,attention,0.,after_attention,cpu_allowed=False)

    def stream(self,batch,nbytes,seconds,cold_fraction,done,*,cpu_allowed):
        replica=batch['replica'];stage=batch['stage'];gpu=self.resource(f'gpu:{replica}:{stage}')
        if not nbytes:gpu.submit(seconds,done);return
        self.stats['weight_read_bytes']+=nbytes*self.layout.tp
        # Shared/dense streaming has no all-resident subset in this phase;
        # routed work uses the current expected cold token-assignment share.
        fraction=cold_fraction if cold_fraction else 1.
        oracle=self.d.comparator=='cpu_oracle' and cpu_allowed
        hot=seconds*(1-fraction);cold=0. if oracle else seconds*fraction*self.d.cold_compute_scale
        chunks=max(1,math.ceil(nbytes/(128*2**20)))
        sizes=[min(128*2**20,nbytes-i*128*2**20) for i in range(chunks)]
        activations=batch['tokens']*self.model['hidden']*2*2*self.model['top_k']*self.layout.tp
        transfers=[lambda size=size:self._source_hops(replica,stage,size,
                    cpu_activation_bytes=activations*size/nbytes if oracle else None) for size in sizes]
        staging=[]
        for size in sizes:
            copy=(activations/self.layout.tp*size/nbytes) if oracle else size
            staging.append(copy/self.timings.near_Bps)
            self.stats['gpu_staging_bytes']+=copy*self.layout.tp
        operator=StreamedOperator(self.events,gpu,hot_gpu_seconds=hot,transfers=transfers,
                    cold_gpu_seconds=[cold*size/nbytes for size in sizes],staging_seconds=staging,
                    slots=self.d.slots,callback=done)
        self.stats['max_outstanding_chunks']=max(self.stats['max_outstanding_chunks'],operator.peak_outstanding)

    def stage_finish(self,batch):
        replica=batch['replica'];stage=batch['stage']
        def release():
            self.stage_busy[replica][stage]=False
            if self.stage_queues[replica][stage]:
                next_batch=self.stage_queues[replica][stage].popleft()
                self.stage_begin(next_batch,stage)
            if stage==0:self.dispatch(replica)
        if stage==self.layout.pp-1:
            def finished():
                release();self.iteration_finished(batch)
            self.resource(f'gpu:{replica}:{stage}').submit(batch['components']['logits'],finished)
        else:
            next_stage=stage+1;nbytes=batch['tokens']*self.model['hidden']*2
            a=self.layout.node(replica,stage);b=self.layout.node(replica,next_stage)
            hops=[]
            if a==b:
                hops=[Hop(self.resource(f'pp-local:{a}'),nbytes/(self.timings.gpu['intra_Bps']*.7)+8e-6)]
            else:
                self.stats['network_bytes']+=nbytes
                bw=50e9*.7  # declared 400 Gb/s NIC at 70% payload efficiency
                hops=[Hop(self.resource(f'nic-out:{a}'),nbytes/bw+8e-6),
                      Hop(self.resource(f'nic-in:{b}'),nbytes/bw+8e-6)]
            # A finite stage handoff buffer holds activation output until the
            # physical hop completes. No next-stage reservation in advance.
            def transferred():release();self.stage_begin(batch,next_stage)
            chain(self.events,hops,transferred)

    def iteration_finished(self,batch):
        replica=batch['replica']
        self.inflight[replica]-=1
        for rid,q in zip(batch['requests'],batch['queries']):
            r=self.requests[rid]
            if batch['prefill']:
                r['processed']+=q
                if r['processed']<r['input_tokens']:
                    self.enqueue_ready(replica,rid);continue
            r['token_times_s'].append(self.events.now)
            if len(r['token_times_s'])<r['output_tokens']:
                self.enqueue_ready(replica,rid)
            else:
                if rid in self.successors:
                    successor=self.successors[rid]
                    self.events.at(self.events.now+r['successor_gap_s'],self.arrive,successor)
                self.persist(replica,rid)
        self.dispatch(replica)

    def persist(self,replica,rid):
        r=self.requests[rid];near=self.near_caches[replica];far=self.far_caches[replica]
        keys=r['completed_blocks'] if self.d.prefix else []
        new=[key for key in keys if key not in far.rows and key not in self.pending_prefix[replica]]
        for key in keys:near.put(key)
        # Keep immutable far copies on promotion. Program only absent copies,
        # never assume instantaneous persistence or unlimited write buffers.
        if not new or not far.capacity:
            self.free_request(replica,rid);return
        self.pending_prefix[replica].update(new)
        remaining=[self.layout.pp]
        def all_done():
            remaining[0]-=1
            if remaining[0]:return
            for key in new:far.put(key);self.pending_prefix[replica].remove(key)
            self.free_request(replica,rid)
        for stage in range(self.layout.pp):
            amount=kv_bytes(self.model,len(new)*16,self.layout.tp,self.layout.pp)
            self.stats['prefix_write_logical_bytes']+=amount*self.layout.tp
            # One 64 MiB foreground save buffer per GPU. It is additional to
            # the two bounded weight-stream buffers and charged in the plan.
            self.save_queues[replica][stage].append((amount,all_done))
            self.save_next(replica,stage)

    def save_next(self,replica,stage):
        if self.save_busy[replica][stage] or not self.save_queues[replica][stage]:return
        self.save_busy[replica][stage]=True
        amount,done=self.save_queues[replica][stage].popleft()
        def write_piece(left):
            if left<=0:
                self.save_busy[replica][stage]=False
                done();self.save_next(replica,stage);return
            size=min(left,64*2**20)
            hops=[Hop(self.resource(f'gpu:{replica}:{stage}'),size/self.timings.near_Bps)]
            hops+=self._source_hops(replica,stage,size,weight=False,write=True)
            chain(self.events,hops,write_piece,left-size)
        write_piece(amount)

    def free_request(self,replica,rid):
        self.kv_used[replica]-=self.requests[rid]['kv_reservation']
        if self.kv_used[replica]<-1e-4:raise AssertionError('negative KV allocation')
        self.admit(replica)

    def service_bounds(self):
        """Optimistic lower bounds, independent of queueing and token clocks.

        Current-batch cold-expert demand is monotone in tokens. A single-token
        route therefore bounds each decode interval from below. GPU consumption
        and source queues impose separate bounds; holding a bounded stream slot
        across every hop also bounds service by total chain work / slot count.
        For a zero-gap offline cohort, mandatory resource work bounds makespan.
        These are proofs inside this model, not measured device limits.
        """
        l=self.layout;d=self.d;m=self.model;t=self.timings
        def source(replica,stage,nbytes,cpu):
            if not nbytes:return (0.,0.,{})
            if l.hbf_stacks:
                seconds=nbytes/self.device.read_Bps
                return seconds,seconds,{f'hbf:{replica}:{stage}':seconds}
            node=l.node(replica,stage);amount=nbytes*l.tp
            missing=amount*(1-self.host_fraction[node])
            host=d.host_Bps or (100e9 if d.gpu=='rtx4090' else 460.8e9)
            ssd=missing/(d.drives*d.ssd_Bps_per_drive)*d.io_scale
            dram=(amount+missing)/host*d.io_scale
            ingress=0. if cpu else nbytes/t.gpu['ingress_Bps']*d.io_scale
            shared=0. if cpu else amount/(8*t.gpu['ingress_Bps'])*d.io_scale
            work={f'ssd:{node}':ssd,f'dram:{node}':dram}
            if not cpu:
                work[f'pcie:{node}']=shared
                work[f'ingress:{replica}:{stage}']=ingress
            return max(ssd,dram,ingress,shared),ssd+dram+ingress+shared,work
        per_replica=[];resource_work={};by_replica={};tokens_by_replica={}
        for replica in range(l.replicas):
            previous_work=resource_work.copy()
            latency=0.
            tokens=sum(max(0,r['output_tokens']-1) for r in self.requests.values()
                       if r['session']%l.replicas==replica)
            for stage in range(l.pp):
                for layer in l.layers(m,stage):
                    a,f,_,_=self.layer_plan[layer]
                    attn_p,ffn_p,_=layer_parameters(m,layer,l.tp)
                    route1,fraction=self._route(layer,1)
                    oracle=d.comparator=='cpu_oracle' and route1>0
                    fraction=1. if f else fraction
                    attn_gpu=2*attn_p/t.near_Bps
                    ffn_gpu=(2*ffn_p+route1)/t.near_Bps
                    if oracle:ffn_gpu*=1-fraction
                    attn_max,attn_chain,_=source(replica,stage,a,False)
                    ffn_max,ffn_chain,_=source(replica,stage,f+route1,oracle)
                    attn_copy=a/t.near_Bps
                    ffn_copy=0. if oracle else (f+route1)/t.near_Bps
                    cold_share=fraction if route1 else (1. if f else 0.)
                    attn_cold=attn_gpu*d.cold_compute_scale if a else 0.
                    ffn_cold=0. if oracle else ffn_gpu*cold_share*d.cold_compute_scale
                    attn_total=attn_gpu*(d.cold_compute_scale if a else 1.)
                    ffn_total=ffn_gpu if oracle else ffn_gpu*(1+cold_share*(d.cold_compute_scale-1))
                    latency+=max(attn_total,attn_max,
                                 (attn_chain+attn_copy+attn_cold)/d.slots)
                    latency+=max(ffn_total,ffn_max,
                                 (ffn_chain+ffn_copy+ffn_cold)/d.slots)
                    # Concavity of distinct-expert use makes traffic/token
                    # smallest at the largest legal decode batch. KV admission
                    # can reduce that batch further for homogeneous offline jobs.
                    members=[r for r in self.requests.values() if r['session']%l.replicas==replica]
                    needed=min((kv_bytes(m,r['input_tokens']+r['output_tokens'],l.tp,l.pp) for r in members),default=1)
                    sessions=len({r['session'] for r in members})
                    max_batch=min(d.batch_cap,max(1,sessions),max(1,int(self.kv_capacity//needed)))
                    cycles=tokens/max_batch
                    route_b,frac_b=self._route(layer,max_batch)
                    oracle_b=d.comparator=='cpu_oracle' and route_b>0
                    frac_b=1. if f else frac_b
                    gpu_attn=2*attn_p/t.near_Bps
                    gpu_ffn=(2*ffn_p+route_b)/t.near_Bps
                    share_b=frac_b if route_b else (1. if f else 0.)
                    gpu_attn*=d.cold_compute_scale if a else 1.
                    gpu_ffn*=1-frac_b if oracle_b else (1+share_b*(d.cold_compute_scale-1))
                    copies=a+(0. if oracle_b else f+route_b)
                    gpu_work=gpu_attn+gpu_ffn+copies/t.near_Bps
                    key=f'gpu:{replica}:{stage}'
                    resource_work[key]=resource_work.get(key,0)+cycles*gpu_work
                    for payload,cpu in ((a,False),(f+route_b,oracle_b)):
                        _,_,work=source(replica,stage,payload,cpu)
                        for key,value in work.items():resource_work[key]=resource_work.get(key,0)+cycles*value
            per_replica.append(latency)
            tokens_by_replica[replica]=tokens
            by_replica[replica]={key:value-previous_work.get(key,0.) for key,value in resource_work.items()
                                if value-previous_work.get(key,0.)>1e-12}
        roots=[r for r in self.requests.values() if r['predecessor'] is None]
        root_upper=len(roots);deadline=max((10+.2*(r['output_tokens']-1) for r in roots),default=10.)
        if roots and any(value>0 for value in resource_work.values()) and all(r['root_arrival_s']==0 for r in roots):
            # Every passing root must finish by TTFT_max + ITL_max*(O-1).
            # Permit fractional selection of whichever roots are cheapest,
            # perfect packing and no prefill/queue/network work. This LP is
            # optimistic: its integer floor bounds roots that could pass.
            from scipy.optimize import linprog
            keys=[key for key,value in resource_work.items() if value>0]
            matrix=[]
            for key in keys:
                matrix.append([by_replica[r['session']%l.replicas].get(key,0.)*
                               max(0,r['output_tokens']-1)/max(1,tokens_by_replica[r['session']%l.replicas])
                               for r in roots])
            solution=linprog(-np.ones(len(roots)),A_ub=np.asarray(matrix),
                b_ub=np.full(len(keys),deadline),bounds=(0,1),method='highs')
            if solution.success:root_upper=min(len(roots),max(0,math.floor(-solution.fun+1e-5)))
        return dict(decode_interval_s=min(per_replica),per_replica_decode_s=per_replica,
                    offline_makespan_s=max(resource_work.values(),default=0),mandatory_resource_work_s=resource_work,
                    mandatory_work_by_replica=by_replica,decode_tokens_by_replica=tokens_by_replica,
                    root_requests=len(roots),passing_roots_upper_bound=root_upper,root_response_deadline_s=deadline,
                    scope='Lower bounds under the declared traffic/capacity/resource model; queues, prefills, writes, launch cost and network are omitted.')

    def run(self,*,retain_tokens=False):
        self.events.run()
        if any(self.waiting) or any(self.ready) or any(any(q) for q in self.stage_queues):
            raise RuntimeError('deadlock/incomplete queue; cannot classify as zero throughput')
        rows=list(self.requests.values())
        if any(r['arrival_s'] is None or len(r['token_times_s'])!=r['output_tokens'] for r in rows):
            raise RuntimeError('incomplete causal workload')
        start=min(r['arrival_s'] for r in rows)
        end=max(self.events.now,max(r['token_times_s'][-1] for r in rows))
        # Drain includes foreground retention writes and their resource cost.
        duration=max(1e-9,end-start)
        scored={name:assess_requests(rows,slo,start,start+duration) for name,slo in SLOS.items()}
        summaries={}
        for name,result in scored.items():
            detail=result.pop('requests')
            for field in ('ttft_s','mean_itl_s','p99_itl_s'):
                values=[r[field] for r in detail if r[field] is not None]
                result[field+'_p50']=quantile(values,.5) if values else None
                result[field+'_p99']=quantile(values,.99) if values else None
            summaries[name]=result
        result=dict(status='simulated_projection',validated=False,design=asdict(self.d),
            summaries=summaries,traffic=self.stats,memory_plans=self.plans,
            events=self.events.processed,elapsed_s=duration,source_counts=self.source_counts,
            resources={name:dict(work_s=r.work,queue_wait_s=r.queue_wait,count=r.count,
                                 utilization=r.work/duration) for name,r in self.resources.items()},
            retained_prefix_metadata_lower_bound_bytes=32*sum(len(c.rows) for c in self.near_caches+self.far_caches),
            completed_requests=len(rows),output_tokens=sum(r['output_tokens'] for r in rows),
            offline_deadline_pass=all(r['token_times_s'][-1]-r['arrival_s']<=3600 for r in rows))
        if retain_tokens:
            result['requests']=[{key:value for key,value in row.items()
                                if key not in ('input_blocks','completed_blocks')} for row in rows]
        return result
