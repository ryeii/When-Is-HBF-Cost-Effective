"""Memory-bounded execution of unchanged event-model sources."""
import importlib
import math
from pathlib import Path
import sys
import weakref
from functools import lru_cache
from check_compiled_runtime import install, BUILD
from run_validation import HERE


def runtime_inputs():
    from compiled_entry import runtime_inputs as base
    return [*(base() if BUILD.exists() else []),HERE/'memory_entry.py',HERE/'memory_cache.py',
        HERE/'runtime_runner.py']


def setup():
    if BUILD.exists():
        from compiled_entry import check
        check()
        install()
    from projection_resources import Events
    original_run=Events.run
    def run(self,*,until=math.inf,max_events=20_000_000):return original_run(self,until=until,max_events=max_events)
    Events.run=run
    from projection_engine import Simulator
    original_init=Simulator.__init__;original_route=Simulator._route
    def init(self,*args,**kwargs):
        original_init(self,*args,**kwargs)
        # Stateful held-out-route subclasses deliberately remain uncached.
        if type(self)._route is original_route:
            owner=weakref.ref(self)
            @lru_cache(maxsize=8192)
            def route(layer,tokens):return original_route(owner(),layer,tokens)
            self._route=route
    Simulator.__init__=init
    import projection_workloads as workloads
    from memory_cache import compact,install_runner_cache
    original_make=workloads.make
    def make(*args,**kwargs):return compact(original_make(*args,**kwargs))
    workloads.make=make
    install_runner_cache()


def main():
    target=Path(sys.argv[1]).resolve();sys.argv=[str(target),*sys.argv[2:]]
    if str(target.parent) not in sys.path:sys.path.insert(0,str(target.parent))
    setup()
    module=importlib.import_module(target.stem)
    module.main()


if __name__=='__main__':main()
