import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
from se2_grid_msgs.msg import SE2Grid
from std_srvs.srv import SetBool


def test_dropout_keeps_original_timestamp_and_resumes_only_new_input():
    spec=importlib.util.spec_from_file_location("map_fault_relay",Path(__file__).with_name("map_fault_relay.py"))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    relay=SimpleNamespace(paused=False,pub=Mock())
    msg=SE2Grid();msg.info.header.stamp.sec=123
    module.MapFaultRelay.forward(relay,msg)
    assert relay.pub.publish.call_args.args[0] is msg
    request=SetBool.Request();request.data=True
    response=module.MapFaultRelay.pause(relay,request,SetBool.Response())
    assert response.success and relay.paused
    module.MapFaultRelay.forward(relay,msg)
    assert relay.pub.publish.call_count==1
    request.data=False
    module.MapFaultRelay.pause(relay,request,SetBool.Response())
    assert relay.pub.publish.call_count==1
    msg.info.header.stamp.sec=125
    module.MapFaultRelay.forward(relay,msg)
    assert relay.pub.publish.call_count==2
    assert relay.pub.publish.call_args.args[0].info.header.stamp.sec==125
