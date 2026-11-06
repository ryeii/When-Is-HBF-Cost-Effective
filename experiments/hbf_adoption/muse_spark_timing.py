"""Text-only Muse Spark timing using measured NVIDIA operator tables.

The graph is constructed explicitly, independent of AISimulate's model registry.
Missing kernel shapes have named transfers, exposed in the output provenance.
"""
from functools import lru_cache
import math

from muse_spark_supplement import model_profile
from projection_timing import GPUS, Timing
from run_validation import HERE

SYSTEMS = HERE / 'results/nvidia-native-systems-v2-20261104'
BACKEND_VERSION = '0.30.0'
EXPERT_DONOR_WIDTH = 2048
WINDOW_TILE = 512


@lru_cache(maxsize=1)
def deployment_model():
    profile = model_profile()
    c = profile['text_config']
    h, w = c['hidden_size'], c['moe_hidden_size']
    return dict(id=profile['id'], family='moe', hidden=h,
                layers=c['num_hidden_layers'], dense_layers=0,
                experts=c['num_local_experts'], top_k=c['num_experts_per_tok'],
                expert_parameters=3*w*c['moe_intermediate_size'],
                parameters=profile['core_text_parameters'], text_config=c,
                full_layers=tuple(i for i, kind in enumerate(c['layer_types'])
                                  if kind == 'full_attention'))


def parameters_for_rank(model, tp):
    c = model['text_config']
    h, w, d = c['hidden_size'], c['moe_hidden_size'], c['head_dim']
    q = c['num_attention_heads'] // tp
    kv = c['num_key_value_heads'] // tp
    attention = h*(3*q+2*kv)*d + 2*h
    # Router, norms and residual gates are replicated. Both compressed-space
    # projections and all three routed matrices are rank-sharded.
    ffn = 2*h*w//tp + h*model['experts'] + 2*w + 3*h
    return attention, ffn, model['expert_parameters']*model['experts']//tp


def layer_kv_bytes(model, layer, tokens, tp):
    c = model['text_config']
    visible = tokens if layer in model['full_layers'] else min(tokens, c['sliding_window'])
    return 4*(c['num_key_value_heads']//tp)*c['head_dim']*visible


def reservation_bytes(model, tokens, tp, pp):
    c = model['text_config']
    bytes_per_token = 4*(c['num_key_value_heads']//tp)*c['head_dim']
    if model.get('archive_prefix_kv'):
        # Arbitrary future prefix boundaries require every materialized block.
        # Reserve its complete bytes until request completion and account for
        # normal staged copying/programming. The attention mask still slides.
        return math.ceil(model['layers']/pp)*tokens*bytes_per_token
    local_tokens = min(tokens, c['sliding_window'])
    return bytes_per_token*max(full*tokens+local*local_tokens for full, local in
                               stage_layer_counts(model['layers'], model['full_layers'], pp))


@lru_cache(maxsize=16)
def stage_layer_counts(layers, full_layers, pp):
    counts = []
    for stage in range(pp):
        start, end = layers*stage//pp, layers*(stage+1)//pp
        full = sum(start <= layer < end for layer in full_layers)
        counts.append((full, end-start-full))
    return tuple(counts)


@lru_cache(maxsize=1)
def database_engine():
    from aiconfigurator_core.sdk.engine import EngineHandle
    from aiconfigurator_core.sdk.perf_database import get_database_view

    path = str(SYSTEMS)
    db = get_database_view('h200_sxm', 'vllm', BACKEND_VERSION,
                           systems_paths=path, database_mode='SILICON')
    return EngineHandle.for_database(db, systems_path=path)


@lru_cache(maxsize=16)
def operation_specs(tp, prefill):
    from aiconfigurator_core.sdk import common as c, operations as o
    from aiconfigurator_core.sdk.engine import build_ops_json

    model = deployment_model()
    cfg = model['text_config']
    h, w, d = cfg['hidden_size'], cfg['moe_hidden_size'], cfg['head_dim']
    q, kv = cfg['num_attention_heads']//tp, cfg['num_key_value_heads']//tp
    vocab, topk = cfg['vocab_size'], model['top_k']
    bf16 = c.GEMMQuantMode.bfloat16
    kv16 = c.KVCacheQuantMode.bfloat16
    attention = (o.ContextAttention('attention', 1, q, kv, kv16,
                                   c.FMHAQuantMode.bfloat16, head_size=d, use_qk_norm=True)
                 if prefill else
                 o.GenerationAttention('attention', 1, q, kv, kv16,
                                       head_size=d, use_qk_norm=True))
    # Row-parallel pooler produces replicated compressed inputs. All-reduce
    # each W2 output before its nonlinear expert norm and the top-k combine.
    # Charge unpooler and attention output reductions separately.
    ops = {
        'attention': [attention],
        'attention_shared': [
            o.GEMM('qkv', 1, (q+2*kv)*d, h, bf16),
            o.GEMM('attention_gate', 1, q*d, h, bf16),
            o.GEMM('output', 1, h, q*d, bf16),
            o.CustomAllReduce('attention_reduce', 1, h, tp),
        ],
        'ffn_shared': [
            o.GEMM('router_tf32_proxy', 1, model['experts'], h, bf16),
            o.GEMM('compress', 1, w, h//tp, bf16),
            o.CustomAllReduce('compress_reduce', 1, w, tp),
            o.CustomAllReduce('expert_output_reduce', 1, w*topk, tp),
            o.GEMM('expand', 1, h, w//tp, bf16),
            o.CustomAllReduce('expand_reduce', 1, h, tp),
        ],
        'expert_donor': [
            o.MoE('expert_donor', 1, w, EXPERT_DONOR_WIDTH, topk, model['experts'],
                  tp, 1, c.MoEQuantMode.bfloat16, 'balanced', 1, prefill),
        ],
        'embedding': [
            o.Embedding('embedding', 1, math.ceil(vocab/tp), h),
            o.CustomAllReduce('embedding_reduce', 1, h, tp),
        ],
        'logits': [o.GEMM('logits', 1, math.ceil(vocab/tp), h, bf16)],
    }
    return {name: build_ops_json(items) for name, items in ops.items()}


@lru_cache(maxsize=65536)
def native_ops(tp, prefill, family, batch, query, context):
    rows = database_engine().evaluate_ops_json(
        operation_specs(tp, prefill)[family], is_context=prefill, batch_size=batch,
        s=query if prefill else context, prefix=context-query if prefill else 0,
        x=batch*query)
    sources = {}
    for _, latency, _, source in rows:
        if not math.isfinite(latency) or latency < 0 or source not in ('silicon', 'mixed', 'empirical', 'estimated'):
            raise ValueError(f'Unexpected native operator result for {family}: {latency}, {source}')
        sources[f'{family}/{source}'] = sources.get(f'{family}/{source}', 0) + 1
    return sum(row[1] for row in rows)*1e-3, sources


def local_attention(tp, prefill, batch, query, context, window):
    if context <= window or not prefill:
        return native_ops(tp, prefill, 'attention', batch, query, min(context, window))[0]
    # Causal full-attention tiles with up to W-1 prior tokens conservatively
    # approximate the unmeasured W=2048 sliding kernel. Cap each tile's context
    # at its real causal endpoint and price a whole tile in the native table.
    prefix = context-query
    return sum(native_ops(tp, True, 'attention', batch, q, min(prefix+start, window-1)+q)[0]
               for start in range(0, query, WINDOW_TILE)
               for q in (min(WINDOW_TILE, query-start),))


@lru_cache(maxsize=65536)
def component_estimate(gpu, tp, prefill, batch, query, context):
    model = deployment_model()
    cfg = model['text_config']
    h, w, topk = cfg['hidden_size'], cfg['moe_hidden_size'], model['top_k']
    tokens = batch*query
    sources = {}

    def op(family, b=batch, q=query, c=context):
        value, records = native_ops(tp, prefill, family, b, q, c)
        sources.update(records)
        return value

    # All published table lookups use the original H200 database. Apply the
    # existing study's explicit RTX4090 transfer to these same operations.
    transfer = 1.
    if gpu == 'rtx4090':
        h200, ada = GPUS['h200'], GPUS[gpu]
        mixing = min(1., tokens/512.)
        transfer = ((h200['memory_Bps']*.8)/(ada['memory_Bps']*.9)*(1-mixing)
                    + h200['peak_flops']/ada['peak_flops']*mixing)
    shared = op('attention_shared')
    full = shared+op('attention')
    local = shared+local_attention(tp, prefill, batch, query, context, cfg['sliding_window'])
    sources['sliding_attention/capped_full_attention_transfer'] = 1
    ffn = op('ffn_shared')+op('expert_donor')*cfg['moe_intermediate_size']/EXPERT_DONOR_WIDTH
    sources['expert_geometry/donor_intermediate_width_scaled'] = 1
    sources['router/bf16_table_proxy_for_tf32'] = 1

    # Fused norm, residual gate, and expert finalization transfers use a
    # bandwidth model. FP32 residual loads/stores are four-byte tensors.
    bandwidth = GPUS[gpu]['memory_Bps']*.8
    norm_and_residual = tokens*(12*h+6*h//tp)/bandwidth + 4e-6
    expert_norm = tokens*topk*4*w/bandwidth + 4e-6
    sources['norm_and_residual/bandwidth_model'] = 1
    full = full*transfer+norm_and_residual
    local = local*transfer+norm_and_residual+tokens*4*h//tp/bandwidth
    ffn = ffn*transfer+norm_and_residual+expert_norm
    embedding = op('embedding')*transfer
    # Compute only one output position per request. Terminal-prefill selection
    # still belongs to the shared event engine.
    logits = op('logits', batch, 1, 1)*transfer+tokens*6/bandwidth+4e-6
    return (full, local, ffn, embedding, logits), sources


class MuseTiming(Timing):
    def stage_weight_bytes(self, stage):
        c = self.model['text_config']
        n = len(self.layout.layers(self.model, stage))
        tp, pp = self.layout.tp, self.layout.pp
        layer_bytes = 2*n*sum(parameters_for_rank(self.model, tp))
        boundary = int(stage == 0)+int(stage == pp-1)
        final_norm = 2*c['hidden_size'] if stage == pp-1 else 0
        return layer_bytes + boundary*2*c['hidden_size']*math.ceil(c['vocab_size']/tp) + final_norm

    def components(self, prefill, batch, query, context):
        values, sources = component_estimate(self.gpu_id, self.layout.tp, prefill, batch, query, context)
        full, local, ffn, embedding, logits = (self.gpu_scale*v for v in values)
        global_layers = len(self.model['full_layers'])
        layers = self.model['layers']
        return dict(attention=full*global_layers+local*(layers-global_layers), ffn=ffn*layers,
                    full_attention_layer=full, sliding_attention_layer=local,
                    ffn_layer=ffn, embedding=embedding, logits=logits), sources
