import importlib.util
from pathlib import Path
import zipfile

import pytest

spec = importlib.util.spec_from_file_location('course_summary',
    Path(__file__).with_name('summarize_navigation_courses.py'))
summary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(summary)


@pytest.mark.parametrize('macro,scope', [
    ('RCLCPP_INFO', 'all_completed_planner_jobs'),
    ('RCLCPP_INFO_THROTTLE', 'throttled_log_samples_only')])
def test_planning_statistics_records_sampling_scope_and_failed_jobs(tmp_path, macro, scope):
    (tmp_path / 'planning.log').write_text(
        'terrain_nav planner: result=success elapsed=2.0ms map_seq=1\n'
        'terrain_nav planner: result=failed elapsed=100.0ms map_seq=2\n')
    with zipfile.ZipFile(tmp_path / 'source_snapshot.zip', 'w') as archive:
        archive.writestr('src/planner/terrain_nav/src/terrain_nav_node.cpp',
            f'{macro}(get_logger(), "terrain_nav planner: result=%s", result);')
    result = summary.planning_statistics(tmp_path)
    assert result['scope'] == scope
    assert result['count'] == 2
    assert result['p50_ms'] == 51.
    assert result['max_ms'] == 100.


def test_boundary_figures_reject_changed_fixture_generator(tmp_path):
    with zipfile.ZipFile(tmp_path/'source_snapshot.zip','w') as archive:
        archive.writestr('src/isaac_sim_hunter_livo2/scripts/boundary_courses.py',b'old generator')
    with pytest.raises(ValueError,match='fixture generator changed'):
        summary.boundary_geometry(tmp_path,{'spec':{'kind':'rubble'}})


def test_boundary_figures_keep_actual_frozen_height_and_evaluation_geometry(tmp_path):
    with zipfile.ZipFile(tmp_path/'source_snapshot.zip','w') as archive:
        archive.write(summary.boundary_courses.__file__,'src/isaac_sim_hunter_livo2/scripts/boundary_courses.py')
    data={'spec':{'kind':'rubble','height_m':.03,'seed':41}}
    objects=summary.boundary_geometry(tmp_path,data)
    assert len([o for o in objects if o['terrain']])==32
    assert max(o['height'] for o in objects if o['terrain'])<=.03
    assert '3 厘米' in summary.case_name({'scenario':'boundary','kind':'rubble'},data)
