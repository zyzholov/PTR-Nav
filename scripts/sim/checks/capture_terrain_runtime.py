#!/usr/bin/env python3
"""Passive stage diagnostics; write only after the owning trial stops.

Instrumented debugging, not an uninstrumented real-time qualification. Original
source timestamps and diagnostics are retained; no commands, goals or truth.
"""
import argparse
import json
from pathlib import Path
import time


def diagnostic_record(msg,wall,clock):
    def level(value):
        # ROS uint8 fields may be represented as one byte by the generated
        # Humble binding; convert the scalar value without losing its bits.
        if isinstance(value,bytes):
            if len(value)!=1:raise ValueError('invalid diagnostic level byte count')
            return value[0]
        return int(value)
    return dict(source_ns=msg.header.stamp.sec*10**9+msg.header.stamp.nanosec,
        receipt_wall_s=wall,receipt_sim_ns=clock.now().nanoseconds,
        status=[dict(name=s.name,level=level(s.level),message=s.message,
                     values={v.key:v.value for v in s.values})for s in msg.status])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('run',type=Path);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--timeout',type=float,default=300.)
    a=p.parse_args()
    if a.output.exists() or not 0<a.timeout<=600:p.error('new output and timeout 0..600s required')
    import rclpy
    from rclpy.parameter import Parameter
    from diagnostic_msgs.msg import DiagnosticArray
    rclpy.init();node=rclpy.create_node('terrain_runtime_observer',
        parameter_overrides=[Parameter('use_sim_time',value=True)]);rows=[];overflow=False
    def receive(msg):
        nonlocal overflow
        if len(rows)>=6000:overflow=True;return
        rows.append(diagnostic_record(msg,time.monotonic(),node.get_clock()))
    node.create_subscription(DiagnosticArray,'/car_0_fsgp_bgk/runtime',receive,10)
    deadline=time.monotonic()+a.timeout
    try:
        while time.monotonic()<deadline:
            rclpy.spin_once(node,timeout_sec=.02)
            if (a.run/'navigation_data.json').exists() or (a.run/'result.json').exists():break
        else:raise RuntimeError('owner did not stop; refusing disk write during motion')
        a.output.parent.mkdir(parents=True,exist_ok=True)
        a.output.write_text(json.dumps(dict(scope=__doc__,source_run=str(a.run.resolve()),
            overflow=overflow,rows=rows),ensure_ascii=False,indent=2)+'\n')
    finally:node.destroy_node();rclpy.shutdown()
    return int(overflow or not rows)


if __name__=='__main__':raise SystemExit(main())
