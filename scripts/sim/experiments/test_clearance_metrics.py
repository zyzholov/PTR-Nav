import sys
from pathlib import Path

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"checks"))
from clearance_metrics import (sustained_clearance, rectangle_circle_clearance,
    sampled_body_clearances, held_velocity_encounter, event_stop_response)


def test_first_low_sample_is_not_sustained_clearing():
    result=sustained_clearance([(0,1),(.1,0),(.2,1)],.58)
    assert result["first_sustained_clear_s"] is None
    assert not result["clear_at_end"]


def test_reappearance_after_clearing_is_counted_and_not_hidden():
    rows=[(i*.1,0) for i in range(24)]+[(2.4,1),(2.5,1)]
    result=sustained_clearance(rows,.58)
    assert result["first_sustained_clear_s"]==0
    assert result["reappearance_count"]==1
    assert not result["clear_at_end"]


def test_missing_samples_cannot_prove_continuity():
    result=sustained_clearance([(0,0),(5,0),(5.1,0)],.58)
    assert result["first_sustained_clear_s"] is None
    assert sustained_clearance([],.58)["clear_at_end"] is None


def test_unordered_or_duplicate_times_are_rejected():
    with pytest.raises(ValueError,match="increasing"):
        sustained_clearance([(0,0),(0,0)],.58)


def test_explicit_unknown_breaks_a_low_risk_run():
    rows=[(i*.1,None if i==12 else 0) for i in range(24)]
    result=sustained_clearance(rows,.58)
    assert result["first_sustained_clear_s"] is None
    assert not result["clear_at_end"]


def test_center_distance_can_hide_bumper_contact_and_yaw_changes_clearance():
    import math
    geometry=dict(front=.58,back=.48,half_width=.40,radius=.65)
    # The old center-only >1.05m check would pass despite front contact.
    assert rectangle_circle_clearance(0,0,0,1.2,0,**geometry)==pytest.approx(-.03)
    assert rectangle_circle_clearance(0,0,math.pi/2,1.2,0,**geometry)==pytest.approx(.15)
    assert rectangle_circle_clearance(0,0,0,-1.2,0,**geometry)==pytest.approx(.07)


def test_clearance_alignment_interpolates_and_refuses_extrapolation_or_large_gaps():
    geometry=dict(front=.58,back=.48,half_width=.40,radius=.65)
    truth=[(0,0,0,0,0,0,t,0) for t in (-.01,.05,.15,.25)]
    values=sampled_body_clearances(truth,[(0,2,0),(.1,4,0),(.3,6,0)],**geometry)
    assert len(values)==1
    assert values[0]==pytest.approx((.05,3-.58-.65,.1))
    assert sampled_body_clearances(truth,[],**geometry)==[]
    with pytest.raises(ValueError):
        rectangle_circle_clearance(0,0,0,1,0,**dict(geometry,radius=-1))


def test_unresponsive_encounter_is_distinct_from_distant_noninteraction():
    geometry=dict(front=.58,back=.48,half_width=.40,radius=.65)
    truth=[(0,2.4+t,0,0,0,0,t,0) for t in (0,.1,.2)]
    obstacles=[(.2,5,3.2),(1.4,5,0),(2.6,5,-3.2)]
    near=held_velocity_encounter(truth,obstacles,**geometry)
    assert near["minimum_clearance_m"]==pytest.approx(-.03)
    far=held_velocity_encounter([(0,s[1]-2,*s[2:]) for s in truth],obstacles,**geometry)
    assert far["minimum_clearance_m"]>1
    assert held_velocity_encounter(truth,[(0,5,3.2),(.1,5,3.2)],**geometry) is None


def test_event_stop_is_not_limited_to_an_arbitrary_world_position():
    truth=[(0,3.5,0,0,0,0,i*.025,0) for i in range(80)]
    response=event_stop_response(truth,[(.1,1,0),(.3,0,0)],.2)
    assert response["first_zero_command_delay_s"]==pytest.approx(.1)
    assert response["first_sustained_truth_stop_delay_s"] is not None
    moving=[(0,3.5+i*.025,0,0,0,0,i*.025,0) for i in range(80)]
    assert event_stop_response(moving,[(.3,0,0)],.2)["first_sustained_truth_stop_delay_s"] is None
