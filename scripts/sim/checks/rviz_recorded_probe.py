#!/usr/bin/env python3
"""Bounded UI fixture from ONE recorded map; not a sensor/nav replay result.

Re-timestamps a stationary map/pose for presentation only, on an isolated DDS
domain. No command, goal, path or synthetic raw LiDAR publisher exists here.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time
import yaml
import numpy as np

import rclpy
from diagnostic_msgs.msg import DiagnosticArray
from geometry_msgs.msg import PointStamped, TransformStamped
from nav_msgs.msg import Odometry
from rclpy.serialization import deserialize_message
from rosgraph_msgs.msg import Clock
from se2_grid_msgs.msg import SE2Grid
from tf2_ros import TransformBroadcaster
from visualization_msgs.msg import Marker
from PIL import Image


ROOT=Path(__file__).resolve().parents[3]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--domain",type=int,default=90)
    parser.add_argument("--view",choices=("risk","height","cloud","source"),default="risk")
    parser.add_argument("--predict",action="store_true",help="Recompute background GP on the recorded measurements")
    parser.add_argument("--probe-distance",type=float,default=2.,help="Read-only probe distance in front of the recorded vehicle")
    parser.add_argument("--probe-candidate",action="store_true",help="Probe an existing GP candidate; readout distinguishes qualified support from candidate-only prediction; no navigation input")
    args=parser.parse_args()
    if not 1<=args.domain<=100: parser.error("use an isolated domain 1..100")
    if not math.isfinite(args.probe_distance) or not 0<=args.probe_distance<=20: parser.error("probe-distance must be in [0,20] metres")
    os.environ["ROS_DOMAIN_ID"]=str(args.domain)
    args.output.mkdir(parents=True,exist_ok=False)
    archive=(args.input/"map.cdr").read_bytes()
    info=json.loads((args.input/"diagnostic.json").read_text())
    if hashlib.sha256(archive).hexdigest()!=info["map_sha256"]: raise ValueError("recording digest mismatch")
    msg=deserialize_message(archive,SE2Grid)
    if args.predict:
        from fsgp_bgk_se2.se2_grid_tools import layer_refs
        refs=layer_refs(msg);raw=SE2Grid();raw.info=msg.info
        raw.start_index_row=msg.start_index_row;raw.start_index_col=msg.start_index_col
        for name in ("elevation","var","valid","ground_observed","risk"):
            if name in refs:
                raw.layers.append(name);raw.has_so2.append(refs[name].has_so2)
                raw.data.extend(msg.data[i] for i in refs[name].data_indices)
        msg=raw
    rclpy.init(); node=rclpy.create_node("recorded_display_fixture")
    grid=node.create_publisher(SE2Grid,"/terrain_fixture/input_map" if args.predict else "/car_0_fsgp_bgk/fused_map",1)
    clock=node.create_publisher(Clock,"/clock",1)
    odom=node.create_publisher(Odometry,"/Odometry",1)
    point=node.create_publisher(PointStamped,"/clicked_point",1)
    transform=TransformBroadcaster(node)
    readouts=[]
    height_meshes=[]
    view_states=[]
    def view_state(report):
        view_states.extend({p.key:p.value for p in s.values} for s in report.status if s.name=='ptr_nav/rviz_view_state')
        del view_states[:-8]
    from rclpy.qos import QoSProfile,DurabilityPolicy
    node.create_subscription(DiagnosticArray,'/terrain_nav/rviz_view_state',view_state,
        QoSProfile(depth=1,durability=DurabilityPolicy.TRANSIENT_LOCAL))
    def received(report):
        readouts.append({p.key:p.value for s in report.status for p in s.values})
        if len(readouts)>3: readouts.pop(0)
    node.create_subscription(DiagnosticArray,"/terrain_nav/display_status",received,1)
    if args.view=="height":
        node.create_subscription(Marker,"/terrain_display/elevation_surface",lambda msg:height_meshes.append(len(msg.points)//3),1)
    if args.probe_candidate:
        from fsgp_bgk_se2.se2_grid_tools import get_layer_2d,grid_buffer_to_world,matrix_shape,layer_refs
        def candidate_probe(current):
            if "gp_candidate" not in layer_refs(current): return
            mask,_=get_layer_2d(current,"gp_candidate",0,matrix_shape(current.data[0]))
            rows,cols=np.where(grid_buffer_to_world(mask,current)>.5)
            if not len(rows): return
            x=current.info.pose.position.x-current.info.length_x/2+(cols+.5)*current.info.pos_resolution
            y=current.info.pose.position.y-current.info.length_y/2+(rows+.5)*current.info.pos_resolution
            index=np.argmin((x-pose.pose.pose.position.x)**2+(y-pose.pose.pose.position.y)**2)
            clicked.point.x=float(x[index]);clicked.point.y=float(y[index])
        node.create_subscription(SE2Grid,"/car_0_fsgp_bgk/fused_map",candidate_probe,1)
    processes=[]; logs=[]
    def start(name,command):
        file=(args.output/(name+".log")).open("w"); logs.append(file)
        process=subprocess.Popen(command,cwd=ROOT,stdout=file,stderr=subprocess.STDOUT,start_new_session=True)
        processes.append(process)
    result=1
    try:
        if args.predict:
            start("prediction",["ros2","run","fsgp_bgk_se2","fsgp_bgk_se2_node","--ros-args",
                "-p","use_sim_time:=true","-p","input_fused_map_topic:=/terrain_fixture/input_map",
                "-p","prediction_backend:=feature_sgp","-p","gpu_backend:=cupy","-p","yaw_bins:=36"])
        start("display",["ros2","run","fsgp_bgk_se2","navigation_display","--ros-args","-p","use_sim_time:=true",
                         "-p","presentation_context:=RECORDED UI FIXTURE"])
        start("surface",["ros2","run","se2_grid_converter","se2_grid_to_pointcloud_node","--ros-args",
                         "-p","use_sim_time:=true","-p","input_topic:=/car_0_fsgp_bgk/fused_map",
                         "-p","layer_name:=elevation","-p","valid_layer_name:=valid",
                         "-p","publish_surface:=true","-p","surface_probability_colors:=true",
                         "-p","surface_risk_layer:=failure_probability_upper",
                         "-p","surface_body_collision_overlay:=false",
                         "-p","surface_use_navigation_height:=true",
                         "-p","height_layer_name:=probability_elevation",
                         "-p","surface_topic:=/car_0_terrain_analyzer/terrain_surface",
                         "-p","height_surface_topic:=/terrain_display/elevation_surface",
                         "-p","z_offset:=0.025",
                         "-p","follow_odom_yaw:=true","-p","odom_topic:=/Odometry"])
        start("risk",["ros2","run","se2_grid_converter","se2_grid_to_pointcloud_node","--ros-args",
            "-p","use_sim_time:=true","-p","input_topic:=/car_0_fsgp_bgk/fused_map",
            "-p","output_topic:=/car_0_terrain_analyzer/risk_cloud","-p","layer_name:=failure_probability_upper",
            "-p","valid_layer_name:=probability_valid","-p","colormap:=probability",
            "-p","min_height:=0.0","-p","max_height:=1.0","-p","drape_on_height:=true",
            "-p","height_layer_name:=elevation","-p","height_as_z:=false","-p","z_offset:=0.035",
            "-p","follow_odom_yaw:=true","-p","odom_topic:=/Odometry"])
        start("vehicle",["ros2","run","vehicle_visualizer","vehicle_visualizer_node","--ros-args",
                         "-p","use_sim_time:=true","-p","mesh.resource:=package://fake_ugv/meshes/car.dae",
                         "-p","vehicle.wheel_base:=0.549","-p","mesh.offset_z:=-0.13",
                         "-r","odom:=/Odometry","-r","vehicle_model:=/vehicle_model"])
        configuration=yaml.safe_load((ROOT/"src/fsgp_bgk_se2/rviz/hunter_terrain_navigation.rviz").read_text())
        for panel in configuration["Panels"]:
            if panel["Name"]=="Navigation Status":panel["ViewMode"]=("risk","height","cloud","source").index(args.view)
        config_path=args.output/"hunter_terrain_navigation.rviz"
        config_path.write_text(yaml.safe_dump(configuration,sort_keys=False))
        start("rviz",["rviz2","-d",str(config_path),
                      "--ros-args","-p","use_sim_time:=true"])
        pose=Odometry(); pose.header.frame_id="map"; pose.child_frame_id="base_link"
        pose.pose.pose.position.x,pose.pose.pose.position.y,pose.pose.pose.position.z=info["pose"][:3]
        yaw=info["pose"][3]; pose.pose.pose.orientation.z=math.sin(yaw/2); pose.pose.pose.orientation.w=math.cos(yaw/2)
        clicked=PointStamped(); clicked.header.frame_id="map"
        clicked.point.x=pose.pose.pose.position.x+args.probe_distance*math.cos(yaw)
        clicked.point.y=pose.pose.pose.position.y+args.probe_distance*math.sin(yaw)
        begin=time.monotonic(); next_map=0
        def pump():
            nonlocal next_map
            seconds=time.monotonic()-begin+100
            stamp=Clock(); stamp.clock.sec=int(seconds); stamp.clock.nanosec=int(seconds%1*1e9)
            clock.publish(stamp); pose.header.stamp=stamp.clock; odom.publish(pose)
            tf=TransformStamped(); tf.header=pose.header; tf.child_frame_id="base_link"
            tf.transform.translation.x=pose.pose.pose.position.x; tf.transform.translation.y=pose.pose.pose.position.y
            tf.transform.translation.z=pose.pose.pose.position.z; tf.transform.rotation=pose.pose.pose.orientation
            transform.sendTransform(tf)
            if time.monotonic()>=next_map:
                msg.info.header.stamp=stamp.clock; grid.publish(msg)
                if args.probe_candidate:point.publish(clicked)
                next_map=time.monotonic()+.3
            rclpy.spin_once(node,timeout_sec=.04)
            if any(p.poll() is not None for p in processes): raise RuntimeError("a UI fixture node exited early")
        while time.monotonic()-begin<(20 if args.predict else 10):pump()
        if not readouts or not readouts[-1].get("banner","").startswith("RECORDED UI FIXTURE"):
            raise RuntimeError("no qualified fixture readout")
        if args.predict and int(readouts[-1].get("candidate_cells","0"))<=0:
            raise RuntimeError("the GP fixture produced no visible candidate cells")
        if args.probe_candidate and not readouts[-1].get("source","").startswith(("GP candidate","GP navigation support")):
            raise RuntimeError("the display probe did not identify an actual GP candidate")
        frames=[]
        for i in range(3):
            path=args.output/("panel.png" if i==2 else f"panel_{i}.png")
            capture=subprocess.Popen(["python3","scripts/sim/checks/rviz_snapshot.py","--output",str(path)],cwd=ROOT)
            deadline=time.monotonic()+15
            try:
                while capture.poll() is None and time.monotonic()<deadline:pump()
                if capture.poll() is None:raise RuntimeError("snapshot timed out")
                if capture.returncode:raise RuntimeError("snapshot failed")
            finally:
                if capture.poll() is None:capture.terminate();capture.wait(timeout=3)
            pixels=np.asarray(Image.open(path).convert("RGB"));h,w=pixels.shape[:2]
            scene=pixels[int(.16*h):int(.85*h),int(.28*w):int(.74*w)].astype(int)
            neutral=(scene.mean(axis=2)>=135)&(scene.max(axis=2)-scene.min(axis=2)<=20)
            frames.append(dict(file=path.name,bright_neutral_pixels=int(neutral.sum())))
            rclpy.spin_once(node,timeout_sec=.1)
        if args.view=='risk' and not any(s.get('mode')=='0' and s.get('terrain_risk_surface')=='1' and
                s.get('body_pose_risk_surface')=='0' and s.get('body_pose_risk_cloud')=='0' for s in view_states):
            raise RuntimeError('RViz risk preset not confirmed: body configuration-space overlays must be disabled')
        if args.view=="height":
            if not height_meshes or min(height_meshes)<100:raise RuntimeError("no usable elevation mesh")
            if min(f["bright_neutral_pixels"] for f in frames)<10:
                raise RuntimeError("vehicle-colored pixels missing in stationary height frames")
        sources=("src/fsgp_bgk_se2/rviz/hunter_terrain_navigation.rviz",
                 "src/fsgp_bgk_se2/fsgp_bgk_se2/navigation_display_node.py",
                 "src/fsgp_bgk_se2/fsgp_bgk_se2/terrain_visuals.py",
                 "src/fsgp_bgk_se2/fsgp_bgk_se2/prediction_worker.py",
                 "src/fsgp_bgk_se2/fsgp_bgk_se2/fsgp_bgk_se2_node.py",
                 "src/fsgp_bgk_se2/fsgp_bgk_se2/probabilistic_risk.py",
                 "src/visualization/ptr_navigation_rviz/src/status_panel.cpp",
                 "scripts/sim/checks/rviz_recorded_probe.py")
        record={"scope":"stationary_UI_fixture_not_navigation_or_timing_evidence",
                "retimestamped_for_display":True,"source_map_sha256":info["map_sha256"],"readouts":readouts,"view_states":view_states,
                "probe_selection":"nearest existing model candidate (display only)" if args.probe_candidate else f"forward {args.probe_distance} m"}
        record.update(view=args.view,stationary_frames=frames,height_mesh_triangles=height_meshes[-3:])
        record["source_sha256"]={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sources}
        (args.output/"fixture.json").write_text(json.dumps(record,indent=2)+"\n")
        print(json.dumps(record),flush=True); result=0
    finally:
        for p in processes:
            if p.poll() is None: os.killpg(p.pid,signal.SIGINT)
        for p in processes:
            try: p.wait(timeout=4)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid,signal.SIGKILL); p.wait()
        for file in logs: file.close()
        node.destroy_node()
        if rclpy.ok(): rclpy.shutdown()
    return result


if __name__=="__main__": raise SystemExit(main())
