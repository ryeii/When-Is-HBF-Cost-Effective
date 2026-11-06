"""Build ancestor-sensitive prefix block keys."""
from __future__ import annotations

import hashlib
import struct


def block_keys(tokens, namespace, block_size=16):
    if block_size < 1 or any(type(value) is not int or not 0 <= value < 2**32 for value in tokens):
        raise ValueError('invalid token sequence or block size')
    parent = hashlib.sha256(namespace.encode()).digest()
    keys = []
    for start in range(0, len(tokens)-block_size+1, block_size):
        parent = hashlib.sha256(parent + struct.pack('<'+'I'*block_size,
                                *tokens[start:start+block_size])).digest()
        keys.append(parent.hex())
    return keys
