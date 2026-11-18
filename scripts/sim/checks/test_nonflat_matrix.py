import json
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[3]


def matrix(*args):
    return subprocess.run([sys.executable, str(ROOT/'scripts/sim/experiments/run_matrix.py'),
                           '--dry-run', *args], cwd=ROOT, text=True, capture_output=True)


def test_downhill_plan_retains_full_courses_and_the_requested_grade():
    result = matrix('--case', 'ramp_detour:2', '--case', 'ramp_crossing:2',
                    '--ramp-direction', 'downhill', '--ramp-rise', '.9', '--vary-goals')
    assert result.returncode == 0, result.stderr
    plan = json.loads(result.stdout)
    assert len(plan['jobs']) == 4
    assert plan['settings']['ramp_direction'] == 'downhill'
    assert plan['settings']['ramp_rise'] == .9
    assert all(job['goal_distance_m'] == 7. and job['goal_lateral_m'] == 0.
               for job in plan['jobs'])


@pytest.mark.parametrize('args', [
    ('--case', 'crossing:1', '--ramp-direction', 'downhill'),
    ('--case', 'ramp_crossing:1', '--reverse-goal'),
    ('--case', 'ramp:1', '--ramp-rise', 'nan'),
    ('--case', 'ramp:1', '--ramp-rise', '1.3'),
])
def test_invalid_grade_or_direction_is_rejected_before_any_simulation(args):
    assert matrix(*args).returncode == 2
