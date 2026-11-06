"""Optional faster execution; identical source and checked complete clocks."""
import math
from check_compiled_runtime import install,BUILD
from run_validation import HERE,ROOT,fingerprint
import json

PROOF=BUILD.parent/'compiled-check/manifest.json'


def runtime_inputs():
    return [HERE/'compiled_entry.py',HERE/'check_compiled_runtime.py',
        PROOF,
        *sorted(BUILD.glob('*.so')),BUILD/'setup.py']


def check():
    proof=json.loads(PROOF.read_text())
    if proof['status']!='completed' or proof['tests']['errors'] or proof['tests']['failures']:raise ValueError('compiled checks did not pass')
    for r in proof['inputs']:
        if fingerprint(ROOT/r['path'])!=r:raise ValueError('changed compiled binary/source')


def main():
    check()
    install()
    from projection_resources import Events
    original=Events.run
    # Cython default storage need not reflect mutations to __kwdefaults__.
    # Preserve the source runner's adjustable computational event guard with
    # an ordinary Python forwarding method; the compiled event loop is intact.
    def run(self,*,until=math.inf,max_events=20_000_000):
        return original(self,until=until,max_events=max_events)
    Events.run=run
    import runtime_runner
    runtime_runner.main()


if __name__=='__main__':main()
