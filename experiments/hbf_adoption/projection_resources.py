"""Causal event/resource primitives for conditional deployment projections.

Every I/O hop is submitted only when its producer completes. Requests already
revealed to a busy resource join its FCFS queue. Dependencies never reserve an
unrevealed successor's resource. Copy writes and GPU work can share one server
to enforce conservative near-memory contention without counting service twice.
"""
from __future__ import annotations

from dataclasses import dataclass
import heapq
import itertools
import math


class Events:
    def __init__(self):
        self.now=0.
        self.queue=[]
        self.serial=itertools.count()
        self.processed=0

    def at(self,time,callback,*args):
        if not math.isfinite(time) or time<self.now-1e-9:
            raise ValueError('event cannot precede its producer')
        heapq.heappush(self.queue,(max(time,self.now),next(self.serial),callback,args))

    def run(self,*,until=math.inf,max_events=20_000_000):
        while self.queue:
            time,serial,callback,args=heapq.heappop(self.queue)
            if time>until:
                heapq.heappush(self.queue,(time,serial,callback,args))
                break
            self.now=time
            self.processed+=1
            if self.processed>max_events:
                raise RuntimeError('event budget exceeded; result is incomplete')
            callback(*args)


class Resource:
    def __init__(self,events,name,*,capture=False):
        self.events=events;self.name=name
        self.free=0.;self.work=0.;self.queue_wait=0.;self.count=0
        self.capture=capture;self.intervals=[]

    def submit(self,duration,callback,*args):
        if not math.isfinite(duration) or duration<0:
            raise ValueError('resource service must be finite and nonnegative')
        ready=self.events.now
        start=max(ready,self.free);finish=start+duration
        self.free=finish;self.work+=duration;self.queue_wait+=start-ready;self.count+=1
        if self.capture:
            self.intervals.append(dict(ready=ready,start=start,finish=finish))
        self.events.at(finish,callback,*args)
        return finish


@dataclass(frozen=True)
class Hop:
    resource: Resource
    duration: float


def chain(events,hops,callback,*args):
    """A dependent transfer chain; later hops are not reserved in advance."""
    hops=tuple(hops)
    def next_hop(index):
        if index==len(hops):
            callback(*args)
        else:
            hop=hops[index]
            hop.resource.submit(hop.duration,next_hop,index+1)
    next_hop(0)


class StreamedOperator:
    """Bounded source/copy/GPU pipeline with observable current-operator inputs.

    `transfers` contains functions that reveal one chunk's hop list on issue.
    At most `slots` chunks are outstanding until their GPU consumption ends.
    HBM copy writes and GPU consumption use the same Resource. Source reads may
    overlap GPU work. There is no next-layer expert oracle or hidden buffer.
    """
    def __init__(self,events,gpu,*,hot_gpu_seconds,transfers,cold_gpu_seconds,
                 staging_seconds,slots,callback):
        if slots<1 or len(transfers)!=len(cold_gpu_seconds) or len(transfers)!=len(staging_seconds):
            raise ValueError('invalid streamed operator')
        if any(value<0 for value in [hot_gpu_seconds,*cold_gpu_seconds,*staging_seconds]):
            raise ValueError('negative service')
        self.events=events;self.gpu=gpu;self.transfers=transfers
        self.cold_gpu_seconds=cold_gpu_seconds;self.staging_seconds=staging_seconds
        self.slots=slots;self.callback=callback
        self.next_chunk=0;self.outstanding=0;self.peak_outstanding=0
        self.remaining=len(transfers)+1
        # Hot work and current cold sources start at the same causal boundary.
        gpu.submit(hot_gpu_seconds,self.finished,-1)
        for _ in range(min(slots,len(transfers))):self.issue_next()

    def issue_next(self):
        index=self.next_chunk
        if index==len(self.transfers):return
        self.next_chunk+=1;self.outstanding+=1
        self.peak_outstanding=max(self.peak_outstanding,self.outstanding)
        if self.outstanding>self.slots:raise AssertionError('staging buffer overflow')
        chain(self.events,self.transfers[index](),self.data_ready,index)

    def data_ready(self,index):
        # A copy cannot consume near-memory bandwidth while opaque GPU work
        # already occupies it. Serializing them is a stated conservative policy.
        self.gpu.submit(self.staging_seconds[index],self.copy_ready,index)

    def copy_ready(self,index):
        self.gpu.submit(self.cold_gpu_seconds[index],self.finished,index)

    def finished(self,index):
        self.remaining-=1
        if index>=0:
            self.outstanding-=1
            self.issue_next()
        if self.remaining==0:
            if self.outstanding or self.next_chunk!=len(self.transfers):
                raise AssertionError('incomplete stream')
            self.callback()
