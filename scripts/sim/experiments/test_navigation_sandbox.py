import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parent))
from navigation_sandbox import SCENARIOS, scenario, elevation, make_message


def test_seeded_cases_reproduce_and_holdout_changes_geometry_or_goal():
    for name in SCENARIOS:
        assert scenario(name,100)==scenario(name,100)
        assert scenario(name,100)!=scenario(name,101)


def test_expected_stop_is_not_arrival_success():
    assert {n for n in SCENARIOS if scenario(n,100)["expected_stop"]}=={"step","unknown","stale_map","steep_ramp"}
    assert scenario("reverse",100)["goal"][0]<0


def test_step_boundary_does_not_shift_when_grid_center_rounds_down():
    terrain={"kind":"step","scale":.4,"edge":3.5}
    values=elevation(terrain,np.array([[3.5-4e-16,0],[3.5-.01,0],[3.5+.01,0]]))
    np.testing.assert_array_equal(values,[.4,0,.4])


def test_message_preserves_canonical_yaw_and_unknown_cells():
    pytest.importorskip("se2_grid_msgs.msg")
    from fsgp_bgk_se2.se2_grid_tools import multiarray_to_matrix
    field=np.zeros((104,104),np.float32);field[3,8]=np.nan
    msg=make_message({"elevation":[field],"risk":[np.full_like(field,i/36) for i in range(36)]})
    assert len(msg.data)==37 and msg.info.length_yaw==pytest.approx(2*np.pi)
    assert msg.info.yaw_resolution==pytest.approx(2*np.pi/36)
    assert np.isnan(multiarray_to_matrix(msg.data[0])).sum()==1


def test_final_statistics_reject_incomplete_or_duplicate_navigation():
    from plot_expanded_campaign import validate_navigation
    plan={"settings":{"seed":7},"cases":[{"id":"a"},{"id":"b"}]}
    with pytest.raises(ValueError,match="incomplete"):
        validate_navigation(plan,{"settings":plan["settings"],"runs":[{"id":"a"}]*2})
    validate_navigation(plan,{"settings":plan["settings"],"runs":[{"id":"a"},{"id":"b"}]})


def test_expanded_terrain_statistics_require_all_eight_conditions():
    from plot_expanded_campaign import CONDITIONS,validate_terrain
    catalog={"datasets":[{"id":"a"}]}
    record={"settings":{"selected_ids":["a"],"conditions":list(CONDITIONS)},"planned_cases":8,
        "runs":[{"id":"a__"+c} for c in CONDITIONS]}
    validate_terrain(record,catalog)
    record["runs"].pop()
    with pytest.raises(ValueError,match="incomplete"):
        validate_terrain(record,catalog)
