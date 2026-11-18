"""Qualification process isolation; never stops an unrelated ROS participant."""
import os
from pathlib import Path
import signal
import subprocess
import time


ROLES=('fastlio_mapping','terrain_nav_node','terrain_analyzer_gpu','fsgp_bgk_se2_node',
       'hunter_livo2_isaac_scene.py','navigation_display_node','lidar_display_node',
       'se2_grid_to_pointcloud_node')


def domain_conflicts(domain,proc_root=Path('/proc')):
    conflicts=[]
    for entry in proc_root.iterdir():
        if not entry.name.isdecimal():continue
        try:
            environment=entry.joinpath('environ').read_bytes().split(b'\0')
            if f'ROS_DOMAIN_ID={domain}'.encode() not in environment:continue
            words=entry.joinpath('cmdline').read_bytes().split(b'\0')
            roles=set(ROLES)&{os.path.basename(w.decode(errors='replace'))for w in words}
            if roles:conflicts.append(dict(pid=int(entry.name),roles=sorted(roles)))
        except (PermissionError,FileNotFoundError,ProcessLookupError):continue
    return sorted(conflicts,key=lambda r:r['pid'])


def run_cancellable(command,stdout,cancel_timeout=20.,cwd=None):
    """Keep the driver alive until its owned pipeline has cleaned up.

    A separate session prevents a terminal interrupt reaching the child both
    directly and through this handler. A second interrupt does not abandon
    the child's bounded finally block. Returns (exit_code, cancelled).
    """
    child=None;cancelled=False;deadline=None
    def cancel(_number,_frame):
        nonlocal cancelled,deadline
        if cancelled:return
        cancelled=True;deadline=time.monotonic()+cancel_timeout
        if child is not None and child.poll() is None:child.send_signal(signal.SIGINT)
    old={s:signal.signal(s,cancel)for s in (signal.SIGINT,signal.SIGTERM)}
    try:
        if cancelled:return 130,True
        child=subprocess.Popen(command,stdout=stdout,stderr=subprocess.STDOUT,start_new_session=True,cwd=cwd)
        if cancelled:child.send_signal(signal.SIGINT)
        while True:
            try:return child.wait(timeout=.1),cancelled
            except subprocess.TimeoutExpired:
                if cancelled and time.monotonic()>=deadline:
                    os.killpg(child.pid,signal.SIGKILL);child.wait()
                    return 130,True
    finally:
        for number,handler in old.items():signal.signal(number,handler)
