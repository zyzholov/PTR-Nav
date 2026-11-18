import threading
import time

from fsgp_bgk_se2.prediction_worker import PredictionWorker,shift_prediction


def test_worker_is_nonblocking_bounded_and_rejects_wrong_window_or_old_source():
    worker=PredictionWorker(); release=threading.Event(); started=threading.Event(); calls=[]
    def work():
        calls.append(1); started.set(); release.wait(timeout=2); return 42
    try:
        assert worker.update((1,),10.,work,period=1,max_age=2) is None
        assert started.wait(timeout=1)
        for _ in range(20):
            assert worker.update((2,),10.1,work,period=1,max_age=2) is None
        assert len(calls)==1
        release.set(); worker.future.result(timeout=1)
        result=worker.update((1,),10.2,work,period=10,max_age=2)
        assert result.value==42 and result.source_stamp==10.
        assert worker.update((2,),10.2,work,period=10,max_age=2) is None
        assert worker.update((1,),13.,work,period=10,max_age=2) is None
        assert worker.update((1,),1.,work,period=10,max_age=2) is None
    finally:
        release.set(); worker.close()


def test_prediction_window_shift_preserves_world_cells_and_leaves_new_strip_unknown():
    import numpy as np
    from fsgp_bgk_se2.feature_sgp import TerrainPrediction
    mean=np.arange(12,dtype=np.float32)
    p=TerrainPrediction(mean,mean*.01,mean*.02,mean+1,np.ones(12,bool),np.ones(12,bool),"cpu",1,1,12)
    shifted=shift_prediction(p,((3,4),0.,0.,.2),((3,4),.2,0.,.2))
    np.testing.assert_equal(shifted.mean.reshape(3,4)[:,:3],mean.reshape(3,4)[:,1:])
    assert np.isnan(shifted.mean.reshape(3,4)[:,-1]).all()
    assert not shifted.accepted.reshape(3,4)[:,-1].any()
    assert shift_prediction(p,((3,4),0.,0.,.2),((3,4),.1,0.,.2)) is None


def test_feature_worker_uses_spawned_process_and_rejects_expired_or_wrong_window():
    from dataclasses import asdict
    import os
    import numpy as np
    from fsgp_bgk_se2.feature_sgp import FeatureSgpParams
    from fsgp_bgk_se2.prediction_worker import FeaturePredictionWorker
    x, y = np.meshgrid(np.arange(5)*.2, np.arange(5)*.2)
    training = np.c_[x.ravel(), y.ravel(), (.02*x-.01*y).ravel()]
    query = np.array([[.3, .3], [.4, .4]])
    worker = FeaturePredictionWorker(asdict(FeatureSgpParams(device='cpu', training_steps=0)))
    try:
        assert worker.update(('map',),10.,(training,query),period=100,max_age=2) is None
        future = worker.future
        assert worker.update(('next',),10.1,(training,query),period=100,max_age=2) is None
        assert worker.future is future
        future.result(timeout=20.)
        snapshot = worker.update(('map',),10.2,(training,query),period=100,max_age=2)
        assert snapshot.worker_pid != os.getpid()
        assert snapshot.value.device == 'cpu'
        np.testing.assert_allclose(snapshot.value.mean, [.003, .004], atol=2e-5)
        assert worker.update(('next',),10.2,(training,query),period=100,max_age=2) is None
        assert worker.update(('map',),13.,(training,query),period=100,max_age=2) is None
    finally:
        worker.close()
