import json
from pathlib import Path
import os
import signal
import subprocess
import sys
import time
from process_ownership import domain_conflicts


def test_only_known_participants_in_the_same_domain_conflict_and_no_process_is_stopped(tmp_path):
    for pid,domain,command in [(1,88,'/project/install/fast_lio/lib/fast_lio/fastlio_mapping'),
            (2,89,'/project/install/terrain_nav/lib/terrain_nav/terrain_nav_node'),
            (3,88,'/bin/bash'),(4,88,'/sim/python3\0/project/hunter_livo2_isaac_scene.py')]:
        p=tmp_path/str(pid);p.mkdir()
        (p/'environ').write_bytes(f'ROS_DOMAIN_ID={domain}\0SECRET=not printed\0'.encode())
        (p/'cmdline').write_bytes(command.encode()+b'\0')
    (tmp_path/'5').mkdir() # Process exits between directory scan and read.
    assert domain_conflicts(88,tmp_path)==[dict(pid=1,roles=['fastlio_mapping']),
        dict(pid=4,roles=['hunter_livo2_isaac_scene.py'])]


def test_repeated_driver_interrupt_waits_for_pipeline_cleanup_and_cannot_start_next_case(tmp_path):
    ready,cleaned,finished=[tmp_path/n for n in ('ready','cleaned','finished')]
    child="""import signal,time,sys
from pathlib import Path
def stop(*_):
 signal.signal(signal.SIGINT,signal.SIG_IGN)
 time.sleep(.2)
 Path(sys.argv[2]).write_text('cleaned')
 raise SystemExit(1)
signal.signal(signal.SIGINT,stop)
Path(sys.argv[1]).write_text(str(__import__('os').getpid()))
while True:time.sleep(.02)
"""
    wrapper="""import json,sys
from pathlib import Path
from process_ownership import run_cancellable
code,cancelled=run_cancellable([sys.executable,'-c',sys.argv[1],sys.argv[2],sys.argv[3]],None)
Path(sys.argv[4]).write_text(json.dumps([code,cancelled]))
"""
    environment=dict(os.environ,PYTHONPATH=str(Path(__file__).parent))
    p=subprocess.Popen([sys.executable,'-c',wrapper,child,str(ready),str(cleaned),str(finished)],
        env=environment,start_new_session=True)
    try:
        until=time.monotonic()+5
        while not ready.exists() and time.monotonic()<until:time.sleep(.01)
        assert ready.exists();child_pid=int(ready.read_text())
        p.send_signal(signal.SIGINT);time.sleep(.05);p.send_signal(signal.SIGINT)
        assert p.wait(timeout=5)==0 and cleaned.exists()
        assert json.loads(finished.read_text())==[1,True]
        try:os.kill(child_pid,0)
        except ProcessLookupError:pass
        else:raise AssertionError('owned pipeline survived driver exit')
    finally:
        if p.poll() is None:os.killpg(p.pid,signal.SIGKILL);p.wait()
