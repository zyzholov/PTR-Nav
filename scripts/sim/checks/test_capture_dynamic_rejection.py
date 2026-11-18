from capture_dynamic_rejection import RejectionBuffers,StatusSelection


def test_latest_capture_keeps_its_last_complete_source_until_a_new_source_matches():
    b=RejectionBuffers(latest=True)
    def status(t):
        return dict(control_state_x_m=str(t),control_state_y_m='0',control_state_yaw_rad='0',
            measured_speed_mps='.3',reported_live_source_stamp_s=str(t))
    b.retain('map',1_000_000_000,b'map1');b.retain('front',1_000_000_000,b'front1')
    b.retain('cloud',1_000_000_000,b'cloud1');b.status(1_000_000_000,status(1))
    assert b.captured['map'][1]==b'map1' and b.pending is None
    b.status(2_000_000_000,status(2))
    b.retain('map',2_000_000_000,b'map2');b.retain('front',2_000_000_000,b'front2')
    assert b.captured['map'][1]==b'map1'
    b.retain('cloud',2_000_000_000,b'cloud2')
    assert b.captured['map'][1]==b'map2' and b.pending is None
    b.status(3_000_000_000,status(3));b.retain('cloud',3_000_000_000,b'unmatched3')
    assert b.captured['map'][1]==b'map2'
    assert b.captured['event']['original_status']['control_state_x_m']=='2'


def test_latest_capture_rolls_back_post_stop_status_and_never_relabels_its_clock():
    b=RejectionBuffers(latest=True)
    for sec in (1,2,3):
        for name in ('map','front','cloud'):b.retain(name,sec*1_000_000_000,bytes([sec]))
        b.status(sec*1_000_000_000,dict(control_state_x_m='0',control_state_y_m='0',
            control_state_yaw_rad='0',measured_speed_mps='0',control_state_stamp_s=str(sec)))
    assert b.captured['event']['control_root_stamp_s']=='3'
    prior=b.before_stop(2.1)
    assert prior['event']['control_root_stamp_s']=='2' and prior['map'][0]==2_000_000_000
    assert b.before_stop(.9) is None
    assert len(b.recent_captures)<=16


def test_revocation_selection_uses_accepted_motion_before_refusal_and_never_truth():
    policy=StatusSelection(after_action=True)
    refused=dict(dynamic_yield_state='refused',command_accepted='0',measured_speed_mps='0')
    assert not policy.accept(refused)
    for accepted,speed in (('0','.6'),('1','0'),('1','nan')):
        assert not policy.accept(dict(dynamic_yield_state='advance',command_accepted=accepted,measured_speed_mps=speed))
        assert not policy.accept(refused)
    assert not policy.accept(dict(dynamic_yield_state='advance',command_accepted='1',measured_speed_mps='.6'))
    assert policy.accept(refused)
    assert StatusSelection().accept(refused)


def test_refreshed_root_keeps_its_clock_instead_of_the_older_selection_callback():
    values=dict(control_state_x_m='3.3',control_state_y_m='-.6',control_state_yaw_rad='0',
                measured_speed_mps='.4',dynamic_control_callback_origin_s='73.074998366',
                odometry_source_stamp_s='73.014998367',control_prediction_sec='.154999997')
    b=RejectionBuffers();b.status(100,values)
    assert b.pending['control_root_stamp_s']=='73.169998364'
    values['control_state_stamp_s']='73.170000001'
    b=RejectionBuffers();b.status(100,values)
    assert b.pending['control_root_stamp_s']=='73.170000001'
    assert b.pending['original_status']['dynamic_control_callback_origin_s']=='73.074998366'


def test_capture_uses_control_time_speed_and_preserves_original_source_measurement():
    b=RejectionBuffers()
    values=dict(control_state_x_m='1',control_state_y_m='2',control_state_yaw_rad='0',
                measured_speed_mps='0',control_state_speed_mps='.24',dynamic_yield_reason='refused')
    b.status(100,values)
    assert b.pending['speed_mps']==.24
    assert b.pending['original_status']['measured_speed_mps']=='0'


def test_reason_selection_never_captures_an_earlier_unrelated_refusal():
    b=RejectionBuffers('terrain/occupancy')
    v=dict(control_state_x_m='1',control_state_y_m='2',control_state_yaw_rad='0',measured_speed_mps='0',
           dynamic_yield_reason='invalid/stale inputs or non-stopping rollout')
    b.status(10,v);assert b.pending is None
    b.event('unrelated refusal');assert b.pending is None
    v['dynamic_yield_reason']='terrain/occupancy footprint or state probability'
    b.status(11,v);assert b.pending['map_source_stamp_ns']==11
    assert b.pending['original_status']['dynamic_yield_reason']==v['dynamic_yield_reason']


LINE=('dynamic yield: no certified escape (tracks=1): advance: braking probability budget '
      '[source_s=59.114998679 sample=(6.70,-0.6,-0.082) t=2.2 speed=0.778 '
      'path_upper=0.083 brake_upper=0.083]; brake: not certified')


def test_capture_needs_exact_source_and_preserves_original_bytes():
    b=RejectionBuffers();b.event(LINE)
    b.retain('map',59_114_998_679,b'map-original')
    b.retain('front',59_214_998_679,b'next-front')
    assert b.captured is None
    b.retain('cloud',59_114_998_679,b'cloud-original')
    b.retain('front',59_114_998_679,b'front-original')
    assert b.captured['map']==(59_114_998_679,b'map-original')
    assert b.captured['front']==(59_114_998_679,b'front-original')
    assert b.captured['event']['rejected_pose']==[6.7,-.6,-.082]
    b.retain('map',60_000_000_000,b'new-map');b.event(LINE.replace('59.114998679','60'))
    assert b.captured['map'][1]==b'map-original'


def test_buffers_are_bounded_and_absent_evidence_is_not_fabricated():
    b=RejectionBuffers()
    for n in range(100):
        for name in b.queues:b.retain(name,n,b'actual')
    assert len(b.queues['map'])==len(b.queues['front'])==6
    assert len(b.queues['cloud'])==len(b.queues['odom'])==40
    b.event(LINE)
    assert b.captured is None


def test_conditional_dynamic_risk_extension_keeps_source_match_and_separate_risk():
    b=RejectionBuffers();b.event(LINE.replace('brake_upper=0.083]',
        'brake_upper=0.083 conditional_dynamic_upper=0.012]'))
    assert b.pending['map_source_stamp_ns']==59_114_998_679
    assert b.pending['conditional_dynamic_upper_at_rejection']==.012
    assert b.pending['path_upper_at_rejection']==.083


def test_explicit_status_source_is_matched_without_inferring_stamp_from_age():
    b=RejectionBuffers();stamp=59_114_998_679
    b.retain('map',stamp,b'map');b.retain('front',stamp,b'front')
    values=dict(control_state_x_m='4.1',control_state_y_m='-.01',control_state_yaw_rad='0',measured_speed_mps='1.2')
    b.status(stamp,values);values['control_state_x_m']='999'
    assert b.captured['event']['control_root']==[4.1,-.01,0]
    assert b.captured['event']['original_status']['control_state_x_m']=='4.1'


def test_distinct_live_and_map_sources_require_original_live_bytes_before_freezing():
    b=RejectionBuffers();stamp=59_114_998_679;live=59_314_998_679
    b.retain('map',stamp,b'map');b.retain('front',stamp,b'front')
    values=dict(control_state_x_m='4.1',control_state_y_m='-.01',control_state_yaw_rad='0',
                measured_speed_mps='1.2',reported_live_source_stamp_s='59.314998679')
    b.status(stamp,values);b.retain('cloud',stamp,b'conditioning')
    assert b.captured is None
    b.retain('cloud',live+1000,b'wrong-source');assert b.captured is None
    b.retain('cloud',live,b'actual-live')
    assert (live,b'actual-live') in b.captured['clouds']
    assert b.captured['event']['map_source_stamp_ns']==stamp


def test_a_missed_early_source_cannot_lock_out_all_later_original_refusals():
    b=RejectionBuffers();b.event(LINE)
    for i in range(6):b.retain('map',60_000_000_000+i,b'later-map')
    assert b.pending is None and b.captured is None and b.unmatched_events==1
    # No later bytes are relabelled with the missed event's source.
    b.event(LINE.replace('59.114998679','60.314998679'))
    b.retain('map',60_314_998_679,b'actual-map');b.retain('front',60_314_998_679,b'actual-front')
    assert b.captured['map']==(60_314_998_679,b'actual-map')
