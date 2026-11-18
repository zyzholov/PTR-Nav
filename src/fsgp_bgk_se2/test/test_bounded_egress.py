"""Bounded publication, exact source ownership, and final deadline enforcement."""
import threading
import time
from types import SimpleNamespace
import numpy as np
from fsgp_bgk_se2.bounded_egress import BoundedEgress


def test_one_inflight_one_latest_pending_and_no_publication_after_close():
    entered=threading.Event();release=threading.Event();finished=threading.Event();calls=[]
    def publish(job):
        calls.append(job)
        if job[0]==1:entered.set();assert release.wait(3.)
        else:finished.set()
    worker=BoundedEgress(publish,lambda exc:None);worker.start()
    try:
        first=(1,bytearray(b'first'));last=(501,bytearray(b'last'))
        assert worker.submit(first) and entered.wait(2.)
        for i in range(2,501):assert worker.submit((i,bytearray(b'replaced')))
        assert worker.submit(last);assert worker.replaced==499
        release.set();assert finished.wait(2.)
        assert calls[0] is first and calls[1] is last and len(calls)==2
    finally:release.set();assert worker.close(2.)
    assert not worker.submit((502,b'closed'))


def test_shutdown_discards_pending_but_never_destroys_running_publication_resources():
    entered=threading.Event();release=threading.Event();calls=[]
    def publish(job):calls.append(job);entered.set();assert release.wait(3.)
    worker=BoundedEgress(publish,lambda exc:None);worker.start()
    try:
        worker.submit(1);assert entered.wait(2.);worker.submit(2)
        assert not worker.close(.01);assert worker.pending is None
    finally:release.set();assert worker.close(2.)
    assert calls==[1]


def test_publisher_exception_is_reported_and_next_source_can_be_processed():
    errored=threading.Event();done=threading.Event();errors=[]
    def publish(job):
        if job==1:raise ValueError('bad wire payload')
        done.set()
    def report(exc):errors.append(exc);errored.set()
    worker=BoundedEgress(publish,report);worker.start()
    try:
        worker.submit(1);assert errored.wait(2.);worker.submit(2);assert done.wait(2.)
        assert len(errors)==1 and isinstance(errors[0],ValueError)
    finally:assert worker.close(2.)


def test_actual_node_egress_rechecks_age_after_encoding_and_keeps_source_stage_ownership(monkeypatch):
    from se2_grid_msgs.msg import SE2Grid
    from rclpy.serialization import deserialize_message
    import fsgp_bgk_se2.fsgp_bgk_se2_node as module
    from fsgp_bgk_se2.processing_loop import SourceBudget
    wire=[];reports=[];age=[.14]
    logger=SimpleNamespace(info=lambda *a,**k:None,warn=lambda *a,**k:None)
    owner=SimpleNamespace(_source_budget=SourceBudget(),output_admission_age_sec=.285,
        output_delivery_reserve_sec=.015,_deadline_output_drops=0,_deadline_input_drops=0,
        _source_age=lambda stamp:age[0],_age_expired=module.FsgpBgkSe2Node._age_expired,
        map_pub=SimpleNamespace(publish=wire.append),last_backend_warn='',publish_debug_clouds=False,
        _publish_runtime=lambda *a,**k:reports.append(k),get_logger=lambda:logger,
        log_period_sec=1.,_runtime_gc_warmup=None)
    source=SE2Grid();source.info.header.stamp.sec=123;source.info.header.frame_id='map'
    out=SE2Grid();out.info=source.info
    result=SimpleNamespace(backend='cupy',backend_reason='test',bgk_mean=np.zeros((2,2)),final_risk_slices=[None])
    stages={'support':23.};start=time.perf_counter()
    job=(source,result,out,.125,start,start,stages,.13)
    original=module.serialize_grid
    def expired_during_encoding(msg):
        payload=original(msg);age[0]=.29;return payload
    monkeypatch.setattr(module,'serialize_grid',expired_during_encoding)
    module.FsgpBgkSe2Node._publish_prepared_map(owner,job)
    assert wire==[] and owner._deadline_output_drops==1
    assert reports[-1]['published'] is False and reports[-1]['stages']['support']==23.
    age[0]=.14;monkeypatch.setattr(module,'serialize_grid',original)
    module.FsgpBgkSe2Node._publish_prepared_map(owner,job)
    assert len(wire)==1
    received=deserialize_message(wire[0],SE2Grid)
    assert received.info.header==source.info.header and reports[-1]['stages']['support']==23.
    # A source expiring WHILE waiting behind publication may not be encoded
    # or relabelled with the current source's time to make it pass.
    age[0]=.4
    def must_not_encode(msg):raise AssertionError('expired queued map encoded')
    monkeypatch.setattr(module,'serialize_grid',must_not_encode)
    module.FsgpBgkSe2Node._publish_prepared_map(owner,job)
    assert len(wire)==1 and owner._deadline_output_drops==2
    assert reports[-1]['drop_reason']=='egress queue deadline'
