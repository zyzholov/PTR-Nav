from types import SimpleNamespace
import json

import pytest

from navigation_trial import Trial


@pytest.mark.parametrize('level',[1,b'\x01'])
def test_controller_safety_evidence_is_bounded_and_keeps_decision_reasons(level):
    node = SimpleNamespace(recording=False, safety_status_samples=[], safety_status_overflow=False)
    msg = SimpleNamespace(header=SimpleNamespace(stamp=SimpleNamespace(sec=12, nanosec=500000000)),
                          status=[SimpleNamespace(name='terrain_nav', level=level, message='dynamic yield retreat',
                                                  values=[SimpleNamespace(key='state_support_known', value='1')])])
    Trial.on_safety_status(node, msg)
    assert not node.safety_status_samples
    node.recording = True
    Trial.on_safety_status(node, msg)
    sample = node.safety_status_samples[0]
    assert sample['stamp_s'] == 12.5
    assert sample['status'][0]['message'] == 'dynamic yield retreat'
    assert sample['status'][0]['values']['state_support_known'] == '1'
    assert json.loads(json.dumps(sample))['status'][0]['level'] == 1
    node.safety_status_samples = [sample]*10000
    Trial.on_safety_status(node, msg)
    assert len(node.safety_status_samples) == 10000
    assert node.safety_status_overflow
