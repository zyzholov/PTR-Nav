from pathlib import Path
import importlib.util
import subprocess
import ast
import pytest
from types import SimpleNamespace as State

spec=importlib.util.spec_from_file_location("goal_probe",Path(__file__).resolve().parents[2]/"tools/hunter_goal_probe.py")
probe=importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def test_capture_helper_is_defined_before_cli_entrypoint():
    tree=ast.parse(Path(spec.origin).read_text())
    helper=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=="capture_while_spinning")
    entry=next(node for node in tree.body if isinstance(node,ast.If) and "__name__" in ast.unparse(node.test))
    assert helper.lineno<entry.lineno


class Process:
    def __init__(self,code=0):
        self.polls=0;self.returncode=None;self.code=code;self.terminated=False

    def poll(self):
        self.polls+=1
        if self.polls>3:self.returncode=self.code
        return self.returncode

    def terminate(self):self.terminated=True;self.returncode=-15
    def wait(self,timeout=None):return self.returncode


@pytest.mark.parametrize("code",[0,2])
def test_capture_keeps_callbacks_running_and_propagates_errors(monkeypatch,code):
    process=Process(code);calls=[]
    monkeypatch.setattr(probe.subprocess,"Popen",lambda command:process)
    monkeypatch.setattr(probe.rclpy,"spin_once",lambda node,timeout_sec:calls.append(timeout_sec))
    if code:
        with pytest.raises(subprocess.CalledProcessError):probe.capture_while_spinning(None,["capture"])
    else:probe.capture_while_spinning(None,["capture"])
    assert calls==[.05]*3
    assert not process.terminated


def test_timeout_cleans_up_owned_capture(monkeypatch):
    process=Process();times=iter([0.,16.])
    monkeypatch.setattr(probe.subprocess,"Popen",lambda command:process)
    monkeypatch.setattr(probe.time,"monotonic",lambda:next(times))
    with pytest.raises(subprocess.TimeoutExpired):probe.capture_while_spinning(None,["capture"])
    assert process.terminated


@pytest.mark.parametrize('deadline,complete_at,stale,truth_moving,expected',[
    (10.,6.,False,False,True),  # beyond the old unrelated four-second limit
    (4.,6.,False,False,False), # cannot extend the goal's real time budget
    (10.,.2,True,False,False), # cleared markers do not excuse stale state
    (3.,.2,False,True,False),  # estimated stop is not physical stop
])
def test_arrival_requires_fresh_settled_completion_within_original_deadline(
        monkeypatch,deadline,complete_at,stale,truth_moving,expected):
    clock=[0.]
    def odom(speed):return State(twist=State(twist=State(linear=State(x=speed))))
    node=State(path_seen=True,goal_seen=True,path_cleared=False,goal_cleared=False,
        visualization_error=None,odom_received=0.,truth_received=0.,
        last_speed_command=.2,last_odom=odom(.2),last_truth=odom(.2))
    def spin(_node,timeout_sec):
        clock[0]+=timeout_sec
        if not stale:node.odom_received=node.truth_received=clock[0]
        if clock[0]>=complete_at:
            node.path_cleared=node.goal_cleared=True;node.last_speed_command=0.
            node.last_odom=odom(0.);node.last_truth=odom(.2 if truth_moving else 0.)
    monkeypatch.setattr(probe.time,'monotonic',lambda:clock[0])
    monkeypatch.setattr(probe.rclpy,'spin_once',spin)
    assert probe.await_arrival_cleanup(node,deadline)==expected
    assert clock[0]<=deadline+.11
