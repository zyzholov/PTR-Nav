import sys
from pathlib import Path
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"checks"))
from navigation_trial import truth_speed


def sample(t,x):
    return [t,x,0,0,0,0,t,0]


def test_truth_speed_uses_physical_displacement_and_simulation_stamp():
    assert truth_speed([sample(10,0),sample(10.1,.12),sample(10.2,.24)])==pytest.approx(1.2)
    assert truth_speed([sample(10,.5),sample(10.1,.5)])==0


def test_missing_or_duplicate_truth_stamps_cannot_claim_vehicle_stopped():
    assert truth_speed([]) is None
    assert truth_speed([sample(10,.5)]) is None
    assert truth_speed([sample(10,.5),sample(10,.5)]) is None
