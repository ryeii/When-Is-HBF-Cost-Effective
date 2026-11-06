"""Data-fitted timing primitives and physical accounting for projection runs.

Fitted rooflines remain operator surrogates. Shared-site near-memory changes,
cross-model attention/host behavior, and other platforms are explicit model
extrapolations. No published simulator's E2E accuracy is inherited.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import json
import math

from run_validation import ROOT,HERE
from model_catalog import kv_bytes

GB=1e9
GiB=2**30


GPUS={
    'rtx4090':dict(memory_bytes=24*GiB,memory_Bps=1008e9,peak_flops=165.2e12,
                   ingress_Bps=31.5e9,intra_Bps=12e9,tdp_watts=450,
                   spec_source='https://images.nvidia.com/aem-dam/Solutions/geforce/ada/nvidia-ada-gpu-architecture.pdf'),
    'h200':dict(memory_bytes=141*GiB,memory_Bps=4800e9,peak_flops=989e12,
                ingress_Bps=64e9,intra_Bps=450e9,tdp_watts=700,
                spec_source='https://www.nvidia.com/en-us/data-center/h200/')}


@dataclass(frozen=True)
class Layout:
    installed_gpus:int
    tp:int
    pp:int
    replicas:int
    architecture:str
    near_sites:int=8
    hbf_stacks:int=0

    def __post_init__(self):
        if self.tp not in (1,2,4,8) or self.pp<1 or self.replicas<1:
            raise ValueError('invalid parallel layout')
        if self.tp*self.pp*self.replicas>self.installed_gpus:
            raise ValueError('layout exceeds installed compute')
        if self.architecture not in ('baseline','shared_site','h3_cascaded'):
            raise ValueError('unknown architecture')
        if self.architecture=='baseline' and (self.near_sites!=8 or self.hbf_stacks):
            raise ValueError('baseline cannot contain HBF')
        if self.architecture=='shared_site' and self.near_sites+self.hbf_stacks!=8:
            raise ValueError('shared-site package overprovisioned')
        if self.architecture=='h3_cascaded' and self.near_sites!=8:
            raise ValueError('cascaded design retains its near tier')
        if not 0<self.near_sites<=8 or self.hbf_stacks<0:
            raise ValueError('invalid memory sites')

    @property
    def width(self):return self.tp*self.pp

    @property
    def used_gpus(self):return self.width*self.replicas

    @property
    def nodes(self):return math.ceil(self.installed_gpus/8)

    def node(self,replica,stage):
        return (replica*self.width+stage*self.tp)//8

    def layers(self,model,stage):
        # Balanced contiguous integer partition; no invented fractional layers.
        start=model['layers']*stage//self.pp
        finish=model['layers']*(stage+1)//self.pp
        if start==finish:raise ValueError('empty pipeline stage')
        return range(start,finish)


def layer_parameters(model,layer,tp):
    """Logical per-TP-rank core parameters with explicit KV head replication."""
    if model['attention_kind']=='gqa':
        h=model['hidden'];d=model['head_dim']
        attention=(2*h*model['heads']*d/tp + 2*h*math.ceil(model['kv_heads']/tp)*d)
        # Conservative replicated Q/K norm vectors where present; negligible
        # payload but retain them rather than changing logical parameter count.
        attention+=max(0,model['attention_parameters_per_layer']-
                       (2*h*model['heads']*d+2*h*model['kv_heads']*d))
    else:
        replicated=model['mla_replicated_attention_parameters']
        attention=replicated+(model['attention_parameters_per_layer']-replicated)/tp
    dense=(model['family']=='dense' or layer<model['dense_layers'])
    if dense:
        ffn=3*model['hidden']*model['dense_intermediate']/tp
        routed=0.
    else:
        ffn=(model['shared_experts']*model['expert_parameters']/tp +
             model['hidden']*model['experts'])
        routed=model['experts']*model['expert_parameters']/tp
    ffn+=2*model['hidden']  # replicated layer normalization vectors
    return attention,ffn,routed


class Timing:
    def __init__(self,model,gpu,layout,*,gpu_scale=1.,host_scale=1.):
        self.model=model;self.gpu_id=gpu;self.gpu=GPUS[gpu];self.layout=layout
        self.gpu_scale=gpu_scale;self.host_scale=host_scale
        profiles=json.loads((HERE/'results/projection-profiles-20261104/profiles.json').read_text())
        self.gemm=profiles['gemm'][gpu]
        self.attention=profiles['attention'][gpu]
        self.cpu=profiles['cpu'][model['family']]
        fraction=layout.near_sites/8
        self.near_capacity=int(self.gpu['memory_bytes']*fraction)
        self.near_Bps=self.gpu['memory_Bps']*self.gemm['memory_fraction']*fraction
        self.peak=self.gpu['peak_flops']*self.gemm['compute_fraction']

    def gemm_seconds(self,flops,near_bytes,kernels=1):
        if min(flops,near_bytes)<0:raise ValueError('negative work')
        return self.gpu_scale*(max(flops/self.peak,near_bytes/self.near_Bps)+kernels*self.gemm['launch_s'])

    @lru_cache(maxsize=16384)
    def attention_seconds(self,queries,context_pairs,kv_read_tokens):
        model=self.model;tp=self.layout.tp;fit=self.attention
        width=model['attention_width']/tp
        # MLA uses its separately counted compressed payload; applying this
        # GQA-fitted attention primitive to MLA is an exposed extrapolation.
        local_kv=kv_bytes(model,1,tp,self.layout.pp)/math.ceil(model['layers']/self.layout.pp)
        flops=4*width*context_pairs
        nbytes=kv_read_tokens*local_kv+queries*width*2*2
        peak=self.gpu['peak_flops']*fit['compute_fraction']
        bw=self.gpu['memory_Bps']*fit['memory_fraction']*(self.layout.near_sites/8)
        return self.gpu_scale*(max(flops/peak,nbytes/bw)+fit['launch_s'])

    def cpu_seconds(self,prefill):
        key='prefill_exclusive_cpu_s' if prefill else 'decode_exclusive_cpu_s'
        return self.host_scale*self.cpu[key]

    def eager_layer_floor(self,prefill):
        return self.host_scale*self.cpu['eager_launch_span_s_per_layer'] if prefill else 0.

    def collective_seconds(self,query_tokens,*,kind='allreduce'):
        if self.layout.tp==1:return 0.
        n=self.layout.tp
        payload=query_tokens*self.model['hidden']*2
        factor=2*(n-1)/n if kind=='allreduce' else (n-1)/n
        # H200 one-direction NVLink convention is explicit; consumer GPUs use
        # a host-staged scenario, whose fabric is not validated P2P hardware.
        return 8e-6+factor*payload/(self.gpu['intra_Bps']*.7)

    def stage_weight_bytes(self,stage):
        model=self.model;tp=self.layout.tp
        layers=list(self.layout.layers(model,stage))
        layerwise=sum(sum(layer_parameters(model,layer,tp)) for layer in layers)
        # Allocate all unassigned shared tensors conservatively across stages;
        # first/last embedding imbalance is charged to the bottleneck stage.
        accounted=sum(sum(layer_parameters(model,layer,1)) for layer in range(model['layers']))
        remainder=max(0,model['parameters']-accounted)
        boundary_share=(.5 if stage in (0,self.layout.pp-1) and self.layout.pp>1 else
                        (1 if self.layout.pp==1 else 0))
        return 2*(layerwise+remainder*boundary_share/tp)


class DeviceService:
    """Read-service line fit to native QD1 range latency, with physical caps.

    Native reduced-geometry timing supplies an anchor. Installed capacity and
    arbitrary stack scaling are explicit projection inputs, not measurements.
    Sustainable energy limits are scenario calculations from HBFSim's assumed
    8 pJ/read-bit, 150 pJ/program-bit, 2 W/stack static and thermal envelope.
    """
    def __init__(self,*,stacks,link_Bps=2e12,read_scale=1.,thermal_watts_per_stack=20.,
                 write_Bps_per_stack=8e9):
        data=json.loads((HERE/'results/device-service-20261104/manifest.json').read_text())
        points=[row for row in data['points'] if row['tier']=='hbf_assumed' and
                row['caller_queue_depth']==1 and row['range_bytes']>=1048576]
        small,large=sorted(points,key=lambda row:row['range_bytes'])
        slope=(large['p50_range_latency_ns']-small['p50_range_latency_ns'])/(large['range_bytes']-small['range_bytes'])*1e-9
        self.startup=max(0,small['p50_range_latency_ns']*1e-9-slope*small['range_bytes'])
        self.read_Bps=min((1/slope)*stacks/4*read_scale,link_Bps)
        power=max(0,thermal_watts_per_stack-2.)*stacks
        if stacks and power<=0:raise ValueError('no dynamic HBF power budget')
        self.read_Bps=min(self.read_Bps,power/(8*8e-12)) if stacks else 0.
        self.write_Bps=min(write_Bps_per_stack*stacks,power/(8*150e-12),link_Bps) if stacks else 0.
        self.capacity_bytes=stacks*512*GiB
        self.read_energy_j_per_byte=8*8e-12
        self.write_energy_j_per_byte=8*150e-12
        self.static_watts=stacks*2.

    def seconds(self,nbytes,write=False):
        if not nbytes:return 0.
        bw=self.write_Bps if write else self.read_Bps
        if bw<=0:raise ValueError('disabled device cannot serve bytes')
        return nbytes/bw+(75e-6 if write else self.startup)
