from types import SimpleNamespace as S
import json
from capture_terrain_runtime import diagnostic_record


def test_clock_nanoseconds_is_an_integer_property_and_source_is_not_relabelled():
    msg=S(header=S(stamp=S(sec=7,nanosec=15)),status=[])
    row=diagnostic_record(msg,12.5,S(now=lambda:S(nanoseconds=8_000_000_123)))
    assert row['source_ns']==7_000_000_015
    assert row['receipt_sim_ns']==8_000_000_123
    assert row['receipt_wall_s']==12.5
    assert json.loads(json.dumps(row))==row


def test_original_missing_or_invalid_diagnostic_values_are_preserved():
    msg=S(header=S(stamp=S(sec=7,nanosec=15)),status=[S(name='runtime',level=1,
        message='output deadline',values=[S(key='fit_ms',value='nan'),S(key='result_published',value='0')])])
    row=diagnostic_record(msg,12.5,S(now=lambda:S(nanoseconds=8_000_000_123)))
    assert row['status'][0]['values']==dict(fit_ms='nan',result_published='0')


def test_real_humble_serialized_diagnostic_roundtrip_is_json_serializable():
    from diagnostic_msgs.msg import DiagnosticArray,DiagnosticStatus,KeyValue
    from rclpy.serialization import serialize_message,deserialize_message
    message=DiagnosticArray();message.header.stamp.sec=7;message.header.stamp.nanosec=15
    message.status=[DiagnosticStatus(name='runtime',level=DiagnosticStatus.WARN,
        message='output deadline',values=[KeyValue(key='fit_ms',value='nan')])]
    message=deserialize_message(serialize_message(message),DiagnosticArray)
    result=diagnostic_record(message,12.5,S(now=lambda:S(nanoseconds=8_000_000_123)))
    assert json.loads(json.dumps(result))['status'][0]['level']==1
