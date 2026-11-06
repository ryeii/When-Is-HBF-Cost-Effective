"""Exact per-instance memoization of pure marginal-route calculations.

The event model, resources, float expressions and ordering are unchanged.
This wrapper is inappropriate for RouteSimulator's stateful trace intervention.
"""
import runpy
import sys
from functools import lru_cache
from pathlib import Path
from projection_engine import Simulator

original_init=Simulator.__init__


def cached_init(self,*args,**kwargs):
    original_init(self,*args,**kwargs)
    # Each instance has immutable model/design/layer plans. No request/token
    # state enters the marginal route method; bounded caching changes no value.
    self._route=lru_cache(maxsize=8192)(self._route)


def main():
    target=Path(sys.argv[1]).resolve();sys.argv=[str(target),*sys.argv[2:]]
    Simulator.__init__=cached_init
    runpy.run_path(str(target),run_name='__main__')


if __name__=='__main__':main()
