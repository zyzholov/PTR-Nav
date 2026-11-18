import threading
import time
import pytest

from fsgp_bgk_se2.processing_loop import ProcessingLoop


def test_busy_worker_retains_only_the_nodes_latest_input_and_cooperative_shutdown():
    lock=threading.Lock();pending=[1];entered=threading.Event();release=threading.Event();done=threading.Event()
    calls=[]
    def ready():
        with lock:return 0. if pending[0] is not None else None
    def process():
        with lock:value=pending[0];pending[0]=None
        calls.append(value)
        if value==1:entered.set();assert release.wait(3.)
        else:done.set()
    loop=ProcessingLoop(process,ready,lambda exc:None);loop.start()
    try:
        assert entered.wait(2.)
        for i in range(2,1002):
            with lock:pending[0]=i
            loop.notify()
        assert not loop.close(timeout=.01)  # Never destroy live worker resources.
        # Use a fresh loop to inspect replacement after the old owner stops.
        release.set();assert loop.close(2.)
        assert calls==[1] and pending[0]==1001
        loop=ProcessingLoop(process,ready,lambda exc:None);loop.start()
        assert done.wait(2.) and calls==[1,1001]
    finally:release.set();assert loop.close(2.)


def test_arrival_between_pending_check_and_wait_is_not_lost():
    checked=threading.Event();proceed=threading.Event();done=threading.Event();first=[True]
    def ready():
        if first[0]:
            first[0]=False;checked.set();assert proceed.wait(2.);return None
        return 0.
    loop=ProcessingLoop(done.set,ready,lambda exc:None);loop.start()
    try:
        assert checked.wait(2.)
        loop.notify();proceed.set()
        assert done.wait(2.)
    finally:proceed.set();assert loop.close(2.)


def test_rate_deadline_does_not_run_early_and_shutdown_wakes_a_wait():
    called=threading.Event();deadline=time.monotonic()+.10
    loop=ProcessingLoop(called.set,lambda:deadline,lambda exc:None);loop.start()
    try:
        for _ in range(10):loop.notify()
        assert not called.wait(.03)
        assert called.wait(2.)
    finally:assert loop.close(2.)
    loop=ProcessingLoop(lambda:None,lambda:time.monotonic()+100.,lambda exc:None);loop.start()
    assert loop.close(.5)


def test_failed_computation_reports_and_can_process_a_later_source():
    pending=[True];errors=[];error_seen=threading.Event();done=threading.Event()
    def ready():return 0. if pending[0] else None
    def process():
        pending[0]=False
        if not errors:raise ValueError('bad source')
        done.set()
    def error(exc):errors.append(exc);error_seen.set()
    loop=ProcessingLoop(process,ready,error);loop.start()
    try:
        assert error_seen.wait(2.) and isinstance(errors[0],ValueError)
        pending[0]=True;loop.notify();assert done.wait(2.)
    finally:assert loop.close(2.)


def test_source_budget_skips_doomed_work_but_retains_fresh_slots_and_eventual_load_probes():
    from fsgp_bgk_se2.processing_loop import SourceBudget
    b=SourceBudget()
    for t in [0.,.2,.4,.6]:
        assert b.start(.13,.285,t);b.observe(.14,.14)
    assert not b.start(.15,.285,.8)  # .15 + .14 + .005 cannot fit.
    assert b.start(.13,.285,.81)
    assert not b.start(.15,.285,1.)
    assert b.start(.15,.285,1.32)  # Probe must still face the original final age gate.
    assert not b.start(float('nan'),.285,2.)
    assert not b.start(-.1,.285,2.)


def test_bounded_service_retry_escapes_periodic_clock_bursts_without_granting_publication():
    from fsgp_bgk_se2.processing_loop import SourceBudget
    legacy=SourceBudget();cadenced=SourceBudget(.2)
    for budget in (legacy,cadenced):
        for _ in range(16):budget.observe(.14,.16)
    starts=[]
    for k in range(11):
        now=.1*k
        # The measured simulation-clock advance can alternate while the
        # wall service cost remains .14s. Publication uses ACTUAL final age.
        advance=.16 if k%5==0 else .13
        for name,budget in (('legacy',legacy),('cadenced',cadenced)):
            if budget.start(.13,.285,now):
                accepted=.13+advance<=.285
                starts.append((name,accepted))
    assert not any(accepted for name,accepted in starts if name=='legacy')
    assert any(accepted for name,accepted in starts if name=='cadenced')
    assert all(not accepted for name,accepted in starts[:2])
    for invalid in (0.,-.1,float('nan'),2.):
        with pytest.raises(ValueError):SourceBudget(invalid)


def test_source_budget_accounts_for_simulation_clock_growth_and_recovers_from_cold_outliers():
    from fsgp_bgk_se2.processing_loop import SourceBudget
    b=SourceBudget();b.observe(.8,.8)
    for _ in range(3):b.observe(.08,.10)
    assert b.start(.17,.285,0.)
    assert not b.start(.19,.285,.1)
    assert b.start(.0,0.,.11)  # No output age contract: no predictive limit.
    old=list(b.samples);b.observe(float('nan'),.1);assert list(b.samples)==old
    for _ in range(20):b.observe(.05,.05)
    assert len(b.samples)==16 and b.start(.20,.285,.12)


def test_egress_wait_does_not_poison_future_service_predictions_but_clock_work_is_retained():
    from fsgp_bgk_se2.processing_loop import SourceBudget
    b=SourceBudget()
    for _ in range(4):
        b.observe(.19,.21,waiting_wall_seconds=.10,waiting_source_seconds=.10)
    assert all(abs(s-.11)<1e-12 for s in b.samples)
    assert b.start(.15,.285,0.)
    assert not b.start(.18,.285,.1)
    previous=list(b.samples)
    b.observe(.1,.1,waiting_wall_seconds=float('nan'))
    b.observe(.1,.1,waiting_source_seconds=-1.)
    assert list(b.samples)==previous


def test_profile_waits_for_positive_readiness_runs_compute_once_and_defers_disk_write(tmp_path):
    from fsgp_bgk_se2.processing_loop import OneShotComputeProfile
    profile=OneShotComputeProfile();path=tmp_path/'compute.pstats';calls=[]
    def compute(value):calls.append(value);return value+1
    assert profile.run(False,compute,1)==2
    assert profile.profile is None and not profile.save(str(path))
    assert profile.run(True,compute,2)==3
    owner=profile.profile
    assert profile.run(True,compute,3)==4
    assert calls==[1,2,3] and profile.profile is owner
    assert not path.exists()  # Owner hasn't stopped or requested persistence.
    assert profile.save(str(path)) and path.stat().st_size>0


def test_profiled_compute_exception_is_not_swallowed_or_retried():
    from fsgp_bgk_se2.processing_loop import OneShotComputeProfile
    profile=OneShotComputeProfile();calls=[]
    def fail():calls.append(1);raise RuntimeError('invalid terrain')
    with pytest.raises(RuntimeError,match='invalid terrain'):profile.run(True,fail)
    assert calls==[1] and profile.profile is not None


