#!/usr/bin/env python3
"""Capture one live probability map and inspect support, not navigation success."""
import argparse
from collections import deque
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from diagnostic_msgs.msg import DiagnosticArray
from rclpy.qos import qos_profile_sensor_data
from rclpy.serialization import serialize_message
from se2_grid_msgs.msg import SE2Grid
from sensor_msgs.msg import PointCloud2

from fsgp_bgk_se2.navigation_display_node import inspect_grid, yaw_of
from fsgp_bgk_se2.se2_grid_tools import layer_refs, multiarray_to_matrix


def finite_json(value):
    if isinstance(value,dict): return {k:finite_json(v) for k,v in value.items()}
    if isinstance(value,list): return [finite_json(v) for v in value]
    if isinstance(value,float) and not math.isfinite(value): return None
    return value


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--timeout",type=float,default=15.)
    args=parser.parse_args()
    if not math.isfinite(args.timeout) or not 0<args.timeout<=30: parser.error("timeout must be in (0,30]")
    args.output.mkdir(parents=True,exist_ok=False)
    rclpy.init(); node=rclpy.create_node("probability_contract_probe")
    sample={}; clouds=deque(maxlen=20); start=time.monotonic()
    def odom(msg): sample["odom"]=msg
    def grid(msg): sample["grid"]=msg
    def status(msg):
        for item in msg.status:
            if item.name=="terrain_nav/chance_constraints":
                sample["control"]={"message":item.message,**{v.key:v.value for v in item.values}}
    node.create_subscription(Odometry,"/Odometry",odom,qos_profile_sensor_data)
    node.create_subscription(SE2Grid,"/car_0_fsgp_bgk/fused_map",grid,1)
    node.create_subscription(PointCloud2,"/cloud_registered",clouds.append,qos_profile_sensor_data)
    node.create_subscription(DiagnosticArray,"/terrain_nav/chance_status",status,1)
    try:
        while time.monotonic()-start<args.timeout and not {"odom","grid"}.issubset(sample):
            rclpy.spin_once(node,timeout_sec=.1)
        if not {"odom","grid"}.issubset(sample): raise RuntimeError("missing live odometry or probability map")
        # Allow discovery and a complete source-map cycle, not a fixed number
        # of unrelated callbacks which can all drain in a few milliseconds.
        collect_until=time.monotonic()+.75
        while time.monotonic()<collect_until:rclpy.spin_once(node,timeout_sec=.05)
        msg=sample["grid"]; pose=sample["odom"].pose.pose
        yaw=yaw_of(pose.orientation)
        if yaw is None or msg.info.header.frame_id!=sample["odom"].header.frame_id:
            raise RuntimeError("invalid heading or frame mismatch")
        refs=layer_refs(msg)
        for name in ("probability_valid","failure_probability_upper","probability_contract"):
            if name not in refs: raise RuntimeError(f"missing {name}")
        bounds=np.stack([multiarray_to_matrix(msg.data[i]) for i in refs["failure_probability_upper"].data_indices])
        probes=[]
        for distance in (0.,.5,1.,2.):
            x=pose.position.x+distance*math.cos(yaw); y=pose.position.y+distance*math.sin(yaw)
            probes.append(dict(distance_m=distance,probe=inspect_grid(msg,x,y,yaw)))
        archive=serialize_message(msg)
        (args.output/"map.cdr").write_bytes(archive)
        (args.output/"odom.cdr").write_bytes(serialize_message(sample["odom"]))
        cloud_record=None
        age_ref=refs.get("gp_completion_age")
        if age_ref is not None:
            age=float(multiarray_to_matrix(msg.data[age_ref.data_indices[0]]).flat[0])
            if math.isfinite(age):
                target=(msg.info.header.stamp.sec*1_000_000_000+msg.info.header.stamp.nanosec
                        -round(age*1e9))
                matches=[c for c in clouds if c.header.frame_id==msg.info.header.frame_id and
                    abs(c.header.stamp.sec*1_000_000_000+c.header.stamp.nanosec-target)<=5]
                if matches:
                    payload=serialize_message(matches[-1])
                    (args.output/"completion_cloud.cdr").write_bytes(payload)
                    cloud_record={"source_age_s":age,"sha256":hashlib.sha256(payload).hexdigest(),
                                  "scope":"timestamp-matched conditioning cloud; not hidden ground truth"}
        nearby_record=None
        if cloud_record is None and clouds:
            cloud=clouds[-1]
            payload=serialize_message(cloud)
            (args.output/"nearby_cloud.cdr").write_bytes(payload)
            nearby_record={"stamp_sec":cloud.header.stamp.sec,"stamp_nanosec":cloud.header.stamp.nanosec,
                "sha256":hashlib.sha256(payload).hexdigest(),
                "scope":"unmatched nearby scan; cannot reproduce the exact original computation"}
        center=probes[0]["probe"] or {}
        record={"scope":"one_live_map_diagnostic_not_navigation_acceptance",
                "frame":msg.info.header.frame_id,"map_stamp_s":msg.info.header.stamp.sec+1e-9*msg.info.header.stamp.nanosec,
                "odom_stamp_s":sample["odom"].header.stamp.sec+1e-9*sample["odom"].header.stamp.nanosec,
                "pose":[pose.position.x,pose.position.y,pose.position.z,yaw],
                "probability_shape":list(bounds.shape),"finite_pose_fraction":float(np.isfinite(bounds).mean()),
                "state_limit_005_fraction":float((np.isfinite(bounds)&(bounds<=.05)).mean()),
                "center_supported":center.get("probability_valid",0)>.5,
                "probes":probes,"control_status":sample.get("control"),"map_sha256":hashlib.sha256(archive).hexdigest(),
                "completion_cloud":cloud_record,"nearby_cloud":nearby_record}
        record=finite_json(record)
        (args.output/"diagnostic.json").write_text(json.dumps(record,indent=2,allow_nan=False)+"\n")
        print(json.dumps(record,allow_nan=False),flush=True)
        return 0
    finally:
        node.destroy_node()
        if rclpy.ok(): rclpy.shutdown()


if __name__=="__main__": raise SystemExit(main())
