"""Run or resume all 90 Muse platform/workload/GPU-count cells."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import fcntl
from itertools import product
import json
from pathlib import Path
import subprocess
import sys

from projection_sweep import COUNTS, WORKLOADS
from projection_timing import GPUS
from run_validation import HERE, ROOT, fingerprint


def completed(folder):
    try:
        receipt = json.loads((folder/'manifest.json').read_text())
        return (receipt['status'] == 'completed' and receipt['cells_completed'] == 1
                and all(fingerprint(ROOT/record['path']) == record
                        for record in receipt['inputs']+receipt['outputs']))
    except (OSError, ValueError, KeyError):
        return False


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--workers', type=int, default=2)
    a = p.parse_args()
    if a.workers < 1:
        p.error('--workers must be positive')
    out = a.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    with (out/'matrix.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(out, a.workers)


def run(out, workers):
    grid = list(product(GPUS, WORKLOADS, COUNTS))
    # Spread the smaller counts across all workload/platform strata first.
    grid.sort(key=lambda case: case[2])
    cells = {f'{gpu}-{workload}-{count}': (gpu, workload, count) for gpu, workload, count in grid}
    records = {}

    def save(status):
        temp = out/'manifest.tmp'
        temp.write_text(json.dumps(dict(status=status, cells_expected=len(cells),
              cells_completed=sum(v['status']=='completed' for v in records.values()),
              cells_failed=sum(v['status']=='failed' for v in records.values()),
              cells=records), indent=2)+'\n')
        temp.replace(out/'manifest.json')

    def run_cell(key):
        gpu, workload, count = cells[key]
        folder = out/key
        folder.mkdir(exist_ok=True)
        attempts = sorted(folder.glob('attempt-*'), reverse=True)
        for attempt in attempts:
            if completed(attempt/'result'):
                return dict(status='completed', receipt=fingerprint(attempt/'result/manifest.json'), reused=True)
        attempt = folder/f'attempt-{len(attempts)+1:04d}'
        attempt.mkdir()
        argv = [sys.executable, str(HERE/'muse_spark_sweep.py'),
                '--gpu', gpu, '--workload', workload, '--counts', str(count),
                '--out', str(attempt/'result')]
        with (attempt/'worker.log').open('w') as log:
            exit_code = subprocess.run(argv, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=False).returncode
        if not exit_code and completed(attempt/'result'):
            return dict(status='completed', receipt=fingerprint(attempt/'result/manifest.json'), reused=False)
        return dict(status='failed', exit_code=exit_code, log=str(attempt/'worker.log'))

    save('running')
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(run_cell, key): key for key in cells}
        for future in as_completed(futures):
            key = futures[future]
            records[key] = future.result()
            save('running')
            print(key, records[key]['status'], flush=True)
    status = 'failed' if any(v['status'] == 'failed' for v in records.values()) else 'completed'
    save(status)
    if status == 'failed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
