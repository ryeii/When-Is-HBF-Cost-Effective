"""Receipted local validation; never launches paid hardware or changes upstreams."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def fingerprint(path: Path) -> dict:
    return {'path': str(path.relative_to(ROOT)), 'bytes': path.stat().st_size,
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def execute(argv: list[str], cwd: Path, destination: Path, env: dict | None = None,
            timeout: int = 1800, inputs: list[Path] | None = None) -> dict:
    destination.mkdir(parents=True, exist_ok=False)
    receipt = {'argv': argv, 'cwd': str(cwd),
               'environment_overrides': env or {},
               'started_utc': datetime.now(timezone.utc).isoformat(),
               'inputs': [fingerprint(p) for p in inputs or []],
               'host': {'system': platform.system(), 'machine': platform.machine(),
                        'python_runner': sys.version},
               'timeout_seconds': timeout, 'status': 'running'}
    (destination / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    start = time.monotonic()
    with (destination / 'stdout.log').open('w') as log:
        process = subprocess.Popen(argv, cwd=cwd, env={**os.environ, **(env or {})},
                                   stdout=log, stderr=subprocess.STDOUT)
        try:
            receipt['exit_code'] = process.wait(timeout=timeout)
            receipt['status'] = 'pass' if process.returncode == 0 else 'fail'
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            receipt.update(status='timeout', exit_code=process.returncode)
    receipt.update(elapsed_seconds=time.monotonic() - start,
                   finished_utc=datetime.now(timezone.utc).isoformat(),
                   log=fingerprint(destination / 'stdout.log'))
    receipt['input_hashes_unchanged'] = receipt['inputs'] == [fingerprint(p) for p in inputs or []]
    if not receipt['input_hashes_unchanged']:
        receipt['status'] = 'inputs_changed'
    (destination / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(f"{destination.name}: {receipt['status']} ({receipt['elapsed_seconds']:.1f}s)", flush=True)
    return receipt
