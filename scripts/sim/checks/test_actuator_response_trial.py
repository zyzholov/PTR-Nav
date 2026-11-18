import numpy as np
from actuator_response_trial import analyze,response_schedule,SCHEDULE
import pytest

def test_response_speed_domain_preserves_steering_timing_and_explicit_stops():
    assert response_schedule()==SCHEDULE
    high=response_schedule(1.8)
    assert [(t,s)for t,v,s in high]==[(t,s)for t,v,s in SCHEDULE]
    assert [v for t,v,s in high]==[1.8 if v else 0. for t,v,s in SCHEDULE]
    for value in (0.,-1.,float('nan'),float('inf'),6.):
        with pytest.raises(ValueError):response_schedule(value)
    small=response_schedule(1.8,.1)
    assert max(abs(s)for t,v,s in small)==.1
    assert [t for t,v,s in small]==[t for t,v,s in SCHEDULE]
    for value in (0.,-.1,float('nan'),1.):
        with pytest.raises(ValueError):response_schedule(1.8,value)

def synthetic(respond=True):
    t=np.arange(0.,13.,.01);speed=np.ones(len(t))
    if respond:speed=np.clip(1-2.5*np.maximum(0,t-9.65),0,1)
    rate=np.zeros(len(t))
    if respond:
        rate[(t>=4.6)&(t<5.5)]=.4;rate[(t>=7.1)&(t<8.)]=-.4
    yaw=np.cumsum(rate)*.01
    x=np.cumsum(speed*np.cos(yaw))*.01;y=np.cumsum(speed*np.sin(yaw))*.01
    return np.column_stack((t,x,y,np.full(len(t),.26),yaw)).tolist()

def test_known_response_latency_and_no_response_are_distinguished():
    result=analyze(synthetic(),[],0.)
    assert .07<=result['events'][0]['first_sustained_response_s']<=.13
    assert .07<=result['events'][1]['first_sustained_response_s']<=.13
    assert .16<=result['events'][2]['first_sustained_response_s']<=.25
    assert result['terminal_speed_max_mps']<1e-6
    missing=analyze(synthetic(False),[],0.)
    assert all(row['first_sustained_response_s']is None for row in missing['events'])

def test_world_translation_cannot_change_actuator_response():
    truth=np.asarray(synthetic());reference=analyze(truth.tolist(),[],0.)
    truth[:,1]+=21.;truth[:,2]-=13.
    shifted=analyze(truth.tolist(),[],0.)
    assert [r['first_sustained_response_s']for r in shifted['events']]==[r['first_sustained_response_s']for r in reference['events']]
