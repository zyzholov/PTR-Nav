"""One in-flight prediction with source-time and rolling-window checks."""
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass,replace
import multiprocessing
import os
import time
import numpy as np


def shift_prediction(value,old_key,new_key):
    """Reindex cached world-aligned predictions; new window strips stay unknown."""
    shape,ox,oy,res=old_key
    if shape!=new_key[0] or abs(res-new_key[3])>1e-8: return None
    dx=(new_key[1]-ox)/res;dy=(new_key[2]-oy)/res
    if abs(dx-round(dx))>1e-4 or abs(dy-round(dy))>1e-4: return None
    dx,dy=round(dx),round(dy);h,w=shape
    if abs(dx)>=w or abs(dy)>=h: return None
    src=(slice(max(0,dy),min(h,h+dy)),slice(max(0,dx),min(w,w+dx)))
    dst=(slice(max(0,-dy),min(h,h-dy)),slice(max(0,-dx),min(w,w-dx)))
    updates={}
    for name in ("mean","latent_var","observation_var","nearest_distance","supported","accepted","local_conditioned","gradient_xy"):
        array=getattr(value,name,None)
        if array is None: continue
        array=np.asarray(array);trailing=array.shape[1:]
        fill=False if array.dtype==bool else np.inf if name=="nearest_distance" else np.nan
        target=np.full(shape+trailing,fill,dtype=array.dtype)
        target[dst]=array.reshape(shape+trailing)[src]
        updates[name]=target.reshape(array.shape)
    return replace(value,**updates)


@dataclass(frozen=True)
class PredictionSnapshot:
    key: tuple
    source_stamp: float
    value: object
    fallbacks: int = 0
    worker_pid: int | None = None


class PredictionWorker:
    def __init__(self, pool=None):
        self.pool=pool if pool is not None else ThreadPoolExecutor(max_workers=1,thread_name_prefix="terrain_prediction")
        self.future=None
        self.latest=None
        self.submitted_wall=-float("inf")

    def _submit(self, key, source_stamp, work):
        return self.pool.submit(lambda: PredictionSnapshot(key,source_stamp,work()))

    def update(self,key,source_stamp,work,*,period,max_age):
        if self.future is not None and self.future.done():
            future=self.future; self.future=None
            self.latest=future.result()
        if self.latest is not None and source_stamp<self.latest.source_stamp:
            self.latest=None
        now=time.monotonic()
        if self.future is None and now-self.submitted_wall>=period:
            self.submitted_wall=now
            self.future=self._submit(key,source_stamp,work)
        result=self.latest
        return result if result is not None and result.key==key and 0<=source_stamp-result.source_stamp<=max_age else None

    def close(self):
        self.pool.shutdown(wait=True,cancel_futures=True)


def _initialize_feature_predictor(parameters):
    import torch
    from .feature_sgp import FeatureSgpParams, FeatureSgpPredictor
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    global _process_predictor
    _process_predictor = FeatureSgpPredictor(FeatureSgpParams(**parameters))


def _feature_prediction(key, source_stamp, arrays):
    training, query = arrays
    _process_predictor.fit(training)
    value = _process_predictor.predict(query)
    return PredictionSnapshot(key, source_stamp, value, _process_predictor.prediction_fallbacks, os.getpid())


class FeaturePredictionWorker(PredictionWorker):
    """One persistent spawned model, isolated from the ROS process's Python GIL.

    Spawn avoids inheriting CUDA or ROS state. Only host arrays cross the
    boundary; the high-rate current-scan conditioner stays in the ROS process.
    """
    def __init__(self, parameters):
        pool = ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context('spawn'),
                                   initializer=_initialize_feature_predictor, initargs=(parameters,))
        super().__init__(pool)

    def _submit(self, key, source_stamp, work):
        return self.pool.submit(_feature_prediction, key, source_stamp, work)
