import pytest
from boundary_trial import single_goal_contract,unexpected_goal,mission_inputs_ready,localization_interlock_engaged,current_dynamic_clearance_failure


def test_clearance_failure_stop_uses_bracketed_truth_and_keeps_the_original_threshold():
    # A 0.5m front extent and 0.2m obstacle radius at x=.79 leave .09m.
    geometry=dict(front=.5,back=.5,half_width=.3)
    truth=[[0.,0.,0.,0.,0.,0.,10.,0.]]
    bodies=[dict(name='crossing',radius=.2)]
    moving={'crossing':[[9.95,.79,0.],[10.05,.79,0.]]}
    event=current_dynamic_clearance_failure(truth,moving,bodies,geometry)
    assert event['body']=='crossing' and event['stamp_s']==10.
    assert event['clearance_m']==pytest.approx(.09) and event['required_m']==.10
    assert event['bracketing_gap_s']==pytest.approx(.1)
    safe={'crossing':[[9.95,.81,0.],[10.05,.81,0.]]}
    assert current_dynamic_clearance_failure(truth,safe,bodies,geometry) is None
    assert current_dynamic_clearance_failure([],moving,bodies,geometry) is None
    assert current_dynamic_clearance_failure(truth,{},bodies,geometry) is None
    assert current_dynamic_clearance_failure(truth,{'crossing':[[9.95,.79,0.]]},bodies,geometry) is None
    assert current_dynamic_clearance_failure(truth,{'crossing':[[9.8,.79,0.],[10.2,.79,0.]]},bodies,geometry) is None
    with pytest.raises(ValueError):current_dynamic_clearance_failure(truth,moving,bodies,geometry,float('nan'))


def test_clearance_failure_stop_chooses_the_first_source_incident_without_mutating_inputs():
    import copy
    truth=[[0.,0.,0.,0.,0.,0.,t,0.]for t in (10.,10.025)]
    moving={'later':[[10.025,.79,0.]],'earlier':[[10.,.79,0.]]};original=copy.deepcopy(moving)
    bodies=[dict(name=n,radius=.2)for n in moving]
    event=current_dynamic_clearance_failure(truth,moving,bodies,dict(front=.5,back=.5,half_width=.3))
    assert event['body']=='earlier' and event['stamp_s']==10.
    assert moving==original


def test_a_latched_navigation_localization_fault_is_a_failure_abort_not_a_timeout_or_pass():
    assert not localization_interlock_engaged([])
    clear=dict(name='terrain_nav/chance_constraints',values={'localization_fault':'0'})
    assert not localization_interlock_engaged([{'status':[clear]}])
    unrelated=dict(name='another_node',values={'localization_fault':'1'})
    assert not localization_interlock_engaged([{'status':[unrelated]}])
    fault=dict(name='terrain_nav/chance_constraints',values={'localization_fault':'1'})
    assert localization_interlock_engaged([{'status':[clear]},{'status':[fault]}])


def test_observer_is_prepared_before_monitor_gate_and_waits_for_fresh_inputs(tmp_path,monkeypatch):
    import json
    import boundary_trial as module
    from types import SimpleNamespace
    node=SimpleNamespace(clock_ns=40_000_000_000);prepared=tmp_path/'prepared.json';start=tmp_path/'start'
    calls=[]
    def spin(_node,timeout_sec):
        assert prepared.exists() and json.loads(prepared.read_text())['observer_initialized']
        calls.append(1)
        if len(calls)==2:start.write_text('armed')
    monkeypatch.setattr(module.rclpy,'spin_once',spin)
    monkeypatch.setattr(module,'mission_inputs_ready',lambda _:len(calls)>=3)
    module.await_mission_gate(node,prepared,start)
    assert len(calls)==3
    assert json.loads(prepared.read_text())['scope']=='observer ready; no mission goal sent'
    with pytest.raises(ValueError):module.await_mission_gate(node,prepared,start)
    with pytest.raises(ValueError):module.await_mission_gate(node,None,start)


def test_observer_gate_timeout_does_not_authorize_a_goal(tmp_path,monkeypatch):
    import boundary_trial as module
    from types import SimpleNamespace
    monkeypatch.setattr(module.rclpy,'spin_once',lambda *_a,**_kw:None)
    monkeypatch.setattr(module,'mission_inputs_ready',lambda _:True)
    with pytest.raises(RuntimeError):
        module.await_mission_gate(SimpleNamespace(clock_ns=1),tmp_path/'prepared',tmp_path/'start',timeout=.001)
    assert not (tmp_path/'start').exists()


def test_one_initial_external_goal_is_distinct_from_internal_rolling_targets():
    goal=dict(frame='map',source_stamp_s=40.1,x=2.5,y=4.2,yaw=0.)
    assert single_goal_contract([dict(goal,topic='/goal_pose')],[dict(goal)],goal)
    assert single_goal_contract([dict(goal,topic='/move_base_simple/goal')],[dict(goal)],goal)
    assert not single_goal_contract([goal,dict(goal,x=12.)],[goal],goal)
    assert not single_goal_contract([goal],[goal,dict(goal,x=12.)],goal)
    assert not single_goal_contract([],[],goal)
    assert not unexpected_goal([],[],goal)
    assert not unexpected_goal([goal],[],goal)
    assert unexpected_goal([goal,dict(goal,x=12.)],[goal],goal)


@pytest.mark.parametrize('change',[dict(x=3.),dict(yaw=.1),dict(frame='odom'),
                                   dict(source_stamp_s=40.2),dict(x=float('nan'))])
def test_goal_replacement_or_unmatched_acknowledgement_cannot_qualify(change):
    goal=dict(frame='map',source_stamp_s=40.1,x=2.5,y=4.2,yaw=0.)
    assert not single_goal_contract([goal],[dict(goal,**change)],goal)
    assert unexpected_goal([goal],[dict(goal,**change)],goal)


def test_goal_auditor_self_subscription_and_zero_clock_cannot_trigger_publication():
    from types import SimpleNamespace as S
    header=S(stamp=S(sec=40,nanosec=0))
    n=S(truth=S(header=header),odom=S(header=header),last_map=S(info=S(header=header)),clock_ns=0,
        goal_pub=S(get_subscription_count=lambda:2),
        get_subscriptions_info_by_topic=lambda _: [S(node_name='hunter_navigation_trial')])
    assert not mission_inputs_ready(n)
    n.clock_ns=40_000_000_000
    assert not mission_inputs_ready(n)
    n.get_subscriptions_info_by_topic=lambda _: [S(node_name='hunter_navigation_trial'),S(node_name='terrain_nav_node')]
    assert mission_inputs_ready(n)
    n.odom=S(header=S(stamp=S(sec=37,nanosec=900_000_000)))
    assert not mission_inputs_ready(n)
    n.odom=S(header=header);n.last_map=S(info=S(header=S(stamp=S(sec=39,nanosec=600_000_000))))
    assert not mission_inputs_ready(n)
    n.last_map=S(info=S(header=header));n.truth=S(header=S(stamp=S(sec=40,nanosec=1)))
    assert not mission_inputs_ready(n)
