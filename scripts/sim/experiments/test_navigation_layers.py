import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"checks"))
from navigation_trial import Trial


def test_baseline_and_enhanced_risk_are_captured_separately():
    node=SimpleNamespace(map_wall=0.,map_intervals=[],map_samples=0,last_map=None,
        obstacle_x=5.,vacated_stamp=1.,ghost_map_samples=[],ghost_enhanced_samples=[],
        ghost_map_before=[],ghost_probability_samples=[],clock_ns=2_000_000_000,map_ages=[])
    stamp=SimpleNamespace(sec=2,nanosec=0)
    message=SimpleNamespace(info=SimpleNamespace(header=SimpleNamespace(stamp=stamp)))
    with patch("navigation_trial.map_probe",return_value={"valid":1.,
               "baseline_risk_local_max":.1,"risk_local_max":.9}):
        Trial.on_map(node,message)
    assert node.ghost_map_samples==[(1.,1.,.1)]
    assert node.ghost_enhanced_samples==[(1.,1.,.9)]
    assert node.ghost_probability_samples==[(1.,None,None)]
