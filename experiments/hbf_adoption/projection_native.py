"""Native resident-operation predictions reused by the deployment event model.

The historical wheel's immutable compiled specs and bundled data are pinned by
native_profiles.py. No regression fallback, correction learning or replacement
of measured MLA by GQA is performed. PP redistributes accumulated layer costs;
that redistribution and the 4090 cross-platform estimates are projections.
"""
from functools import lru_cache
import json
from pathlib import Path

from run_validation import HERE
from projection_timing import GPUS,Timing

BASE=HERE/'results/native-profiles-v4-20261104'


@lru_cache(maxsize=32)
def engine(model_id,tp):
    from aiconfigurator_core.sdk.engine import EngineHandle
    record=json.loads((BASE/f'{model_id}-tp{tp}.json').read_text())
    if record['status']!='executed' or record['diagnostics']['source']!='aic':
        raise ValueError('native reference is unsupported')
    return EngineHandle((BASE/f'{model_id}-tp{tp}.bin').read_bytes())


@lru_cache(maxsize=100000)
def reference(model_id,tp,prefill,batch,query,context):
    h=engine(model_id,tp)
    if prefill:
        prefix=max(0,context-query)
        rows,_=h.run_static_per_op(batch_size=batch,isl=prefix+query,osl=1,
                                   prefix=prefix,mode='static_ctx')
        # The native context phase does not include first-token logits.
        # Charge a B-token logits operation before the first sampled token.
        _,gen=h.run_static_per_op(batch_size=batch,isl=max(0,context-1),osl=2,mode='static_gen')
        rows=rows+[row for row in gen if 'logits' in row[0]]
    else:
        _,rows=h.run_static_per_op(batch_size=batch,isl=max(0,context-1),osl=2,mode='static_gen')
    result=dict(attention=0.,ffn=0.,embedding=0.,logits=0.)
    sources={}
    for name,latency,energy,source in rows:
        if 'p2p' in name:continue  # PP1 reference; actual PP hops live in DES.
        if 'logits' in name:component='logits'
        elif 'embedding' in name:component='embedding'
        elif any(word in name for word in ('moe','ffn','act_gate','router','add_norm_2','ar_2','tp_allreduce')):component='ffn'
        else:component='attention'
        result[component]+=latency/1000
        sources[source]=sources.get(source,0)+1
    if sum(result.values())<=0:raise ValueError('empty native work')
    return result,sources


class NativeTiming(Timing):
    def components(self,prefill,batch,query,context):
        raw,sources=reference(self.model['id'],self.layout.tp,prefill,batch,query,context)
        result=raw.copy()
        if self.gpu_id=='rtx4090':
            # Cross-platform component transfer is explicitly extrapolated.
            # The two platform GEMM fits anchor memory/compute throughput;
            # a memory-vs-compute interpolation supplies the transfer factor.
            tokens=batch*query
            memory_ratio=(GPUS['h200']['memory_Bps']*.8)/(self.gpu['memory_Bps']*.9)
            compute_ratio=GPUS['h200']['peak_flops']/self.gpu['peak_flops']
            mixing=min(1.,tokens/512.)
            factor=memory_ratio*(1-mixing)+compute_ratio*mixing
            for component in result:result[component]*=factor
        for component in result:result[component]*=self.gpu_scale
        return result,sources
