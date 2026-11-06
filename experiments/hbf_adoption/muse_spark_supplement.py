"""Frozen Muse text architecture derived from the public Transformers config."""
import json

from run_validation import HERE

MODEL = 'muse_spark_open_816b_a42b'


def model_profile():
    path = HERE/'results/model-metadata-20261104/muse-architecture.json'
    return json.loads(path.read_text())


def kv_bytes(config, tokens, tp=1):
    if not 0 <= tokens <= config['max_position_embeddings'] or tp < 1:
        raise ValueError('Invalid context or tensor parallel width')
    full = config['layer_types'].count('full_attention')
    local = len(config['layer_types'])-full
    return 4*math.ceil(config['num_key_value_heads']/tp)*config['head_dim']*(
        full*tokens + local*min(tokens,config['sliding_window']))
