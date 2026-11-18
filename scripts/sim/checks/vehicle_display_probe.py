#!/usr/bin/env python3
"""Small isolated marker transport/reset test, not an occlusion or navigation test."""
import argparse
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time

import rclpy
from nav_msgs.msg import Odometry
from rosgraph_msgs.msg import Clock
from rclpy.qos import QoSProfile,DurabilityPolicy
from visualization_msgs.msg import Marker,MarkerArray


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--domain",type=int,default=92)
    args=parser.parse_args()
    if not 1<=args.domain<=100:parser.error("use isolated domain 1..100")
    os.environ["ROS_DOMAIN_ID"]=str(args.domain)
    args.output.mkdir(parents=True,exist_ok=False)
    rclpy.init();node=rclpy.create_node("vehicle_display_probe")
    clock=node.create_publisher(Clock,"/clock",10)
    odom=node.create_publisher(Odometry,"/probe/odom",10)
    samples=[]
    node.create_subscription(MarkerArray,"/probe/model",lambda msg:samples.append(msg),
        QoSProfile(depth=1,durability=DurabilityPolicy.TRANSIENT_LOCAL))
    file=(args.output/"vehicle.log").open("w")
    process=subprocess.Popen(["ros2","run","vehicle_visualizer","vehicle_visualizer_node","--ros-args",
        "-p","use_sim_time:=true","-p","mesh.resource:=package://fake_ugv/meshes/car.dae",
        "-r","odom:=/probe/odom","-r","vehicle_model:=/probe/model"],
        stdout=file,stderr=subprocess.STDOUT,start_new_session=True)
    def phase(start,x,invalid=False,duration=1.):
        begin=time.monotonic()
        while time.monotonic()-begin<duration:
            t=start+time.monotonic()-begin
            tick=Clock();tick.clock.sec=int(t);tick.clock.nanosec=int(t%1*1e9)
            clock.publish(tick);rclpy.spin_once(node,timeout_sec=.015)
            pose=Odometry();pose.header.stamp=tick.clock;pose.header.frame_id="probe_world"
            pose.pose.pose.position.x=math.nan if invalid else float(x)
            pose.pose.pose.orientation.w=1.;odom.publish(pose)
            rclpy.spin_once(node,timeout_sec=.04)
            if process.poll() is not None:raise RuntimeError("visualizer exited early")
    try:
        phase(100,0,duration=3.)
        before=len(samples);assert before>=5,"marker transport missing"
        phase(2,2);after=len(samples)
        assert after-before>=5,"markers stopped after simulation clock reset"
        mesh=next(m for m in samples[-1].markers if m.type==Marker.MESH_RESOURCE)
        assert mesh.header.frame_id=="probe_world"
        assert abs(mesh.pose.position.x-2)<1e-6
        assert mesh.color.a==1. and mesh.lifetime.sec==0 and mesh.lifetime.nanosec==0
        phase(4,0,invalid=True)
        assert all(math.isfinite(m.pose.position.x) for arr in samples for m in arr.markers)
        late=[]
        node.create_subscription(MarkerArray,"/probe/model",lambda msg:late.append(msg),
            QoSProfile(depth=1,durability=DurabilityPolicy.TRANSIENT_LOCAL))
        deadline=time.monotonic()+2
        while not late and time.monotonic()<deadline:rclpy.spin_once(node,timeout_sec=.05)
        assert late,"late subscriber did not get retained vehicle"
        latest=next(m for m in late[-1].markers if m.type==Marker.MESH_RESOURCE)
        assert latest.pose.position.x==2.,"invalid odometry overwrote last valid marker"
        report=dict(scope="marker_transport_reset_only_not_pixel_or_navigation_evidence",passed=True,
            markers_before_reset=before,markers_after_reset=after-before,
            opaque_mesh=True,persistent_marker=True,late_join=True,invalid_pose_rejected=True)
        (args.output/"qualification.json").write_text(json.dumps(report,indent=2)+"\n")
        print(json.dumps(report))
    finally:
        if process.poll() is None:os.killpg(process.pid,signal.SIGINT)
        try:process.wait(timeout=4)
        except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
        file.close();node.destroy_node();rclpy.shutdown()


if __name__=="__main__":main()
