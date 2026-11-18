"""A single compute owner, separate from the ROS ingress executor.

The node owns the sole latest input; this loop never copies or queues maps.
Processing still includes encoding, source-age admission and publication.
"""
import threading
import time
import math
from collections import deque
from statistics import median


class OneShotComputeProfile:
    """Opt-in stage diagnosis; owner writes the profile only after shutdown."""
    def __init__(self):
        self.profile=None

    def run(self,ready,compute,*args):
        if not ready or self.profile is not None:return compute(*args)
        import cProfile
        self.profile=cProfile.Profile()
        return self.profile.runcall(compute,*args)

    def save(self,path):
        if self.profile is None:return False
        self.profile.dump_stats(path)
        return True


class SourceBudget:
    """Avoid spending a compute slot on an input unlikely to meet delivery.

    This is scheduling only. The original hard input/output age checks remain
    authoritative. Occasional probes avoid starvation after load decreases;
    a probe never grants publication or changes a source timestamp.
    """
    def __init__(self,probe_period_sec=.5):
        if not math.isfinite(probe_period_sec) or not .05<=probe_period_sec<=1.:
            raise ValueError('source-budget probe period must be within .05..1s')
        self.probe_period_sec=probe_period_sec
        self.samples=deque(maxlen=16);self.last_start=-math.inf
        self.lock=threading.Lock()

    def start(self,age,admission_age,now):
        if not math.isfinite(age) or age<0:return False
        with self.lock:samples=tuple(self.samples)
        ready=(admission_age<=0 or len(samples)<4 or
               age+median(samples)+.005<=admission_age)
        if ready or now-self.last_start>=self.probe_period_sec:
            self.last_start=now
            return True
        return False

    def observe(self,wall_seconds,source_seconds,*,waiting_wall_seconds=0.,waiting_source_seconds=0.):
        if not math.isfinite(wall_seconds) or wall_seconds<0:return
        if (not math.isfinite(waiting_wall_seconds) or waiting_wall_seconds<0 or
                not math.isfinite(waiting_source_seconds) or waiting_source_seconds<0):return
        source=source_seconds if math.isfinite(source_seconds) and source_seconds>=0 else 0.
        # Past egress congestion is not the service time of a later frame.
        # Each CURRENT job still faces its unchanged queue/encoded age gates;
        # removing old waiting from the scheduling predictor grants no map
        # publication permission and changes no source timestamp.
        with self.lock:self.samples.append(max(0.,wall_seconds-waiting_wall_seconds,
                                              source-waiting_source_seconds))


class ProcessingLoop:
    def __init__(self,process,ready_at,on_error):
        self.process=process;self.ready_at=ready_at;self.on_error=on_error
        self.changed=threading.Event();self.stopping=threading.Event()
        self.thread=threading.Thread(target=self._run,name='terrain_compute')

    def start(self):self.thread.start()

    def notify(self):self.changed.set()

    def close(self,timeout=10.):
        self.stopping.set();self.changed.set()
        self.thread.join(timeout)
        return not self.thread.is_alive()

    def _run(self):
        while not self.stopping.is_set():
            # Clear BEFORE inspecting node state. An arrival after the read
            # sets the event and cannot be lost between inspecting and wait.
            self.changed.clear()
            try:
                ready=self.ready_at()
                if ready is None:
                    self.changed.wait(.1)
                    continue
                delay=max(0.,ready-time.monotonic())
                if delay:
                    self.stopping.wait(min(delay,.05))
                    continue
                self.process()
            except Exception as exc:
                self.on_error(exc)
                # Broken metadata must not turn into an unbounded busy loop.
                self.stopping.wait(.01)
