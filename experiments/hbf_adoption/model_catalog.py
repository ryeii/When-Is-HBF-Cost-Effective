"""Literature/configuration-derived text inference payloads for projections.

No checkpoint is executed. BF16 payloads of FP8-released architectures are
explicit counterfactual representations; tensor-index storage is kept separate.
Optional multi-token prediction and training-only parameters are omitted.
"""
from __future__ import annotations

import json
import math

from run_validation import HERE, fingerprint


def catalog():
    base=HERE/'results/model-metadata-20261104'
    metadata=json.loads((base/'manifest.json').read_text())
    models=[]
    for source in metadata['models']:
        path=base/source['id']/'config.json'
        if path.exists():
            config=json.loads(path.read_text())
            provenance=dict(kind='official_immutable_config',file=fingerprint(path))
        else:
            snapshot=base/'llama-architecture.json'
            record=json.loads(snapshot.read_text())[source['repository']]
            config=record['config']
            provenance=dict(kind='public_meta_architecture',file=fingerprint(snapshot),
                            original=record['original_provenance'])
        h=config['hidden_size'];layers=config['num_hidden_layers']
        heads=config['num_attention_heads'];head_dim=config.get('head_dim',h//heads)
        kvheads=config.get('num_key_value_heads',heads)
        experts=config.get('num_experts',config.get('n_routed_experts',0)) or 0
        topk=config.get('num_experts_per_tok',0) or 0
        if experts:
            dense_layers=config.get('first_k_dense_replace',0)
            if config.get('mlp_only_layers'):
                dense_layers=len(config['mlp_only_layers'])
            moe_layers=layers-dense_layers
        else:
            dense_layers=layers;moe_layers=0
        expert_params=3*h*config.get('moe_intermediate_size',0)
        routed=expert_params*experts*moe_layers
        shared_expert=expert_params*(config.get('n_shared_experts') or 0)*moe_layers
        if 'kv_lora_rank' in config:
            no_rope=config['qk_nope_head_dim'];rope=config['qk_rope_head_dim']
            vdim=config['v_head_dim'];rank=config['kv_lora_rank']
            qrank=config.get('q_lora_rank')
            qparams=(h*qrank + qrank*heads*(no_rope+rope)+qrank) if qrank else h*heads*(no_rope+rope)
            attn_per_layer=qparams+h*(rank+rope)+rank*heads*(no_rope+vdim)+heads*vdim*h+rank
            kv_values_per_layer_token=rank+rope
            attention_width=heads*vdim
            replicated_attention=(h*qrank+qrank if qrank else 0)+h*(rank+rope)+rank
            attention_kind='mla'
        else:
            attn_per_layer=h*(heads+2*kvheads)*head_dim+heads*head_dim*h
            kv_values_per_layer_token=2*kvheads*head_dim
            attention_width=heads*head_dim
            replicated_attention=0
            attention_kind='gqa'
            # Qwen3 q/k normalization parameters.
            if config.get('model_type') in ('qwen3','qwen3_moe'):
                attn_per_layer+=2*head_dim
        embedding=h*config['vocab_size']*(1 if config.get('tie_word_embeddings',False) else 2)
        dense_ffn=3*h*config.get('intermediate_size',0)*dense_layers
        router=h*experts*moe_layers
        normalization=2*h*layers+h
        total=embedding+layers*attn_per_layer+dense_ffn+shared_expert+routed+router+normalization
        shared=total-routed
        active=shared+(routed*topk/experts if experts else 0)
        model=dict(id=source['id'],repository=source['repository'],revision=source['revision'],
                   family=source['family'],layers=layers,hidden=h,heads=heads,kv_heads=kvheads,
                   head_dim=head_dim,attention_width=attention_width,attention_kind=attention_kind,
                   kv_values_per_layer_token=kv_values_per_layer_token,
                   kv_bytes_per_token_bf16=2*layers*kv_values_per_layer_token,
                   parameters=total,shared_parameters=shared,routed_parameters=routed,
                   expert_parameters=expert_params,experts=experts,top_k=topk,
                   moe_layers=moe_layers,dense_layers=dense_layers,
                   dense_intermediate=config.get('intermediate_size'),
                   expert_intermediate=config.get('moe_intermediate_size'),
                   attention_parameters_per_layer=attn_per_layer,
                   mla_replicated_attention_parameters=replicated_attention,
                   shared_experts=config.get('n_shared_experts') or 0,
                   embedding_parameters=embedding,
                   active_parameters=active,bf16_weight_bytes=2*total,
                   serialized_quantization=config.get('quantization_config'),
                   provenance=provenance,
                   representation='dimension-counted BF16 core text inference; not loaded checkpoint storage')
        if model['attention_kind']=='mla':
            model['kv_sharding']='replicated across TP; sharded by PP layers'
        else:
            model['kv_sharding']='KV heads sharded across TP, replicated when TP exceeds KV heads; sharded by PP'
        models.append(model)
    if len(models)!=8 or any(model['parameters']<=0 or model['active_parameters']>model['parameters'] for model in models):
        raise ValueError('invalid model catalog')
    return models


def kv_bytes(model,tokens,tp=1,pp=1):
    layer_count=math.ceil(model['layers']/pp)
    if model['attention_kind']=='mla':
        values=model['kv_values_per_layer_token']
    else:
        values=2*math.ceil(model['kv_heads']/tp)*model['head_dim']
    return int(tokens*layer_count*values*2)


if __name__=='__main__':
    for model in catalog():
        print(model['id'],f"{model['parameters']/1e9:.3f} B total",
              f"{model['active_parameters']/1e9:.3f} B active",model['kv_bytes_per_token_bf16'])
