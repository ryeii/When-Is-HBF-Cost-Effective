"""Export CPU-executed NVIDIA resident-operation references, without GPUs.

Uses the published AISimulate 0.12.0.dev2 wheel, not the newer checked-out
source. Native fallback regression is rejected. The immutable local model
configs and bundled operator tables determine the reference. This exports
model predictions, not additional hardware measurements.
"""
from __future__ import annotations
import argparse
import importlib.metadata
import json
from pathlib import Path
import shutil
import time

from model_catalog import catalog,meta_config
from run_validation import ROOT,HERE,fingerprint

BATCHES=[1,4,16,64]
CONTEXTS=[1,128,512,2048,8192,32768]
QUERIES=[1,64,128,512,2048,8192,16384]
PREFIXES=[0,512,4096,16384]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--systems',type=Path)
    args=parser.parse_args();out=args.out.resolve();out.mkdir(parents=True,exist_ok=False)
    shutil.copy2(__file__,out/'runner.py')
    from aiconfigurator_core.sdk.rust_engine_step import RustForwardPassPerfModel
    from aiconfigurator_core.sdk.engine import EngineHandle
    import aiconfigurator_core
    package=Path(aiconfigurator_core.__file__).parent
    # Capture the entire installed implementation/data, rather than pretending
    # a newer Git checkout built this historical binary wheel.
    inputs=[fingerprint(p) for p in sorted(package.rglob('*'))
            if p.is_file() and p.suffix in ('.py','.yaml','.parquet','.csv')]
    inputs += [fingerprint(p) for p in sorted((package.parent/'aisimulate').glob('*.so'))]
    if args.systems:inputs.append(fingerprint(args.systems.resolve()/'manifest.json'))
    wheel=next((ROOT/'.cache/hbf-adoption/wheels').glob('*.whl'))
    manifest=dict(schema='hbf-adoption.native-reference.v1',wheel=fingerprint(wheel),
                  distribution_version=importlib.metadata.version('aisimulate'),
                  inputs=inputs,records=[],started_unix=time.time(),
                  scope='CPU-executed resident operator predictions; not new measurements or E2E/HBF validation.',
                  grids=dict(batch=BATCHES,decode_context=CONTEXTS,prefill_query=QUERIES,prefill_prefix=PREFIXES))
    for model in catalog():
        local=out/'configs'/model['id'];local.mkdir(parents=True)
        source=HERE/'results/model-metadata-20261104'/model['id']/'config.json'
        if source.exists():config=json.loads(source.read_text())
        else:
            config,_=meta_config(model['repository']);config['architectures']=['LlamaForCausalLM']
            config['model_type']='llama';config['rms_norm_eps']=1e-5
            config['max_position_embeddings']=131072
            config['rope_theta']=500000.0
        (local/'config.json').write_text(json.dumps(config,indent=2)+'\n')
        for tp in (1,2,4,8):
            record=dict(model=model['id'],tp=tp,model_config=fingerprint(local/'config.json'))
            try:
                systems=str(args.systems.resolve()) if args.systems else None
                version='0.30.0' if systems else 'current'
                identity=dict(schema_version=1,model_name=str(local),system_name='h200_sxm',
                              systems_path=systems,systems_root=systems,
                              backend='vllm',backend_version=version,tp_size=tp,pp_size=1,
                              moe_tp_size=tp,moe_ep_size=1,attention_dp_size=1,
                              weight_dtype='bfloat16',moe_dtype='bfloat16',
                              activation_dtype='bfloat16',kv_cache_dtype='bfloat16',
                              forward_model='op_level',database_mode='SILICON')
                canonical=RustForwardPassPerfModel.best_available(identity)
                diagnostics=canonical.diagnostics()
                if diagnostics['source']!='aic' or diagnostics['readiness']!='ready':
                    raise ValueError(f'native unavailable: {diagnostics}')
                handle=EngineHandle.compile(str(local),'h200_sxm','vllm',backend_version=version,systems_path=systems,
                        tp_size=tp,pp_size=1,moe_tp_size=tp,moe_ep_size=1,
                        gemm_quant_mode='bfloat16',moe_quant_mode='bfloat16',
                        fmha_quant_mode='bfloat16',kvcache_quant_mode='bfloat16',forward_model='op_level')
                record.update(identity=identity,diagnostics=diagnostics,status='executed',points=[],misses=[])
                for batch in BATCHES:
                    for context in CONTEXTS:
                        shape=dict(phase='decode',batch=batch,context=context,query=1,prefix=0)
                        try:
                            _,ops=handle.run_static_per_op(batch_size=batch,isl=max(0,context-1),osl=2,mode='static_gen')
                            if not ops or sum(row[1] for row in ops)<=0:raise ValueError('empty/zero prediction')
                            record['points'].append(dict(**shape,operations=ops))
                        except Exception as e:record['misses'].append(dict(**shape,error=f'{type(e).__name__}: {e}'))
                    for query in QUERIES:
                        for prefix in PREFIXES:
                            shape=dict(phase='prefill',batch=batch,query=query,prefix=prefix,context=query+prefix)
                            try:
                                ops,_=handle.run_static_per_op(batch_size=batch,isl=query+prefix,osl=1,prefix=prefix,mode='static_ctx')
                                if not ops or sum(row[1] for row in ops)<=0:raise ValueError('empty/zero prediction')
                                record['points'].append(dict(**shape,operations=ops))
                            except Exception as e:record['misses'].append(dict(**shape,error=f'{type(e).__name__}: {e}'))
                spec=out/f"{model['id']}-tp{tp}.bin";spec.write_bytes(handle.spec_bytes)
                record['compiled_spec']=fingerprint(spec)
                canonical.close()
            except Exception as e:record.update(status='unsupported',error=f'{type(e).__name__}: {e}')
            path=out/f"{model['id']}-tp{tp}.json";path.write_text(json.dumps(record,indent=2)+'\n')
            manifest['records'].append(dict(model=model['id'],tp=tp,file=fingerprint(path),status=record['status'],
                                           points=len(record.get('points',[])),misses=len(record.get('misses',[]))))
            (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
            print(model['id'],tp,record['status'],len(record.get('points',[])),len(record.get('misses',[])),flush=True)
    manifest['elapsed_seconds']=time.time()-manifest['started_unix']
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')


if __name__=='__main__':main()
