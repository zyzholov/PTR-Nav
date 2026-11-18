import json
from types import SimpleNamespace
import pytest
from boundary_cloud_evidence import ClearingEvidence


def test_bounded_snapshots_keep_real_source_times_and_serialize_only_after_stop(tmp_path):
    evidence=ClearingEvidence(True);evidence.actor(10,4.3,3)
    assert evidence.vacated is None
    evidence.actor(11,4.3,0);evidence.actor(12,4.3,-1.6)
    assert evidence.vacated==12
    messages=[]
    for offset in (0,1,2,4):
        message=SimpleNamespace(data=f'frame-{offset}'.encode());messages.append(message)
        evidence.retain('laser_map',message,12.1+offset,12.15+offset)
    assert len(evidence.samples)==4 and not list(tmp_path.iterdir())
    assert evidence.samples['laser_map',0][0] is messages[0]
    evidence.retain('laser_map',SimpleNamespace(data=b'late-old'),13,17)
    manifest=json.loads(open(evidence.archive(tmp_path/'capture',lambda m:m.data)).read())
    assert evidence.out_of_order==1
    assert len(manifest['snapshots'])==5
    assert all(row['source_age_s']==pytest.approx(.05) for row in manifest['snapshots'])
    assert manifest['snapshots'][-1]['source_stamp_s']==16.1
    assert (tmp_path/'capture/laser_map_terminal.cdr').read_bytes()==b'frame-4'


def test_missing_intervals_cannot_be_filled_with_a_late_frame(tmp_path):
    evidence=ClearingEvidence(True);evidence.actor(10,4.3,0);evidence.actor(11,4.3,-2)
    evidence.retain('registered',b'late',17,17.1)
    path=evidence.archive(tmp_path/'capture',lambda m:m)
    rows=json.loads(open(path).read())['snapshots']
    assert len([r for r in rows if r['status']=='missed_source_time_window'])==4
    assert len([r for r in rows if r['status']=='captured'])==1


def test_disabled_evidence_never_retains_or_writes(tmp_path):
    evidence=ClearingEvidence(False);evidence.actor(10,4.3,0);evidence.actor(11,4.3,-2)
    evidence.retain('risk',object(),12,12)
    assert evidence.archive(tmp_path/'capture',lambda _:b'') is None
    assert not list(tmp_path.iterdir())


def test_risk_snapshot_retains_original_arrays_and_grid_geometry(monkeypatch):
    import math
    import array
    from se2_grid_msgs.msg import SE2Grid
    from std_msgs.msg import Float32MultiArray
    import boundary_trial as module
    monkeypatch.setattr(module.Trial,'on_map',lambda *_:None)
    node=object.__new__(module.BoundaryTrial);node.clock_ns=100_000_000_000
    node.clearing_evidence=ClearingEvidence(True)
    msg=SE2Grid();msg.info.header.stamp.sec=100;msg.info.header.frame_id='map'
    msg.info.length_yaw=2*math.pi;msg.info.yaw_resolution=math.pi
    msg.start_index_row=2;msg.start_index_col=1
    msg.layers=['ignored','collision_risk','probability_valid'];msg.has_so2=[False,False,True]
    msg.data=[]
    for i in range(4):
        a=Float32MultiArray();a.data=array.array('f',[i]);msg.data.append(a)
    node.on_map(msg)
    selected=node.clearing_evidence.latest['risk'][0]
    assert selected.layers==['collision_risk','probability_valid']
    assert selected.has_so2==[False,True]
    assert selected.info is msg.info and selected.start_index_row==2 and selected.start_index_col==1
    assert len(selected.data)==3 and all(a is b for a,b in zip(selected.data,msg.data[1:]))
    assert msg.layers==['ignored','collision_risk','probability_valid'] and len(msg.data)==4
