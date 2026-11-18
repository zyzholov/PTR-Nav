#!/usr/bin/env python3
"""Small controlled GP-to-production-planner qualification, not Isaac physics.

One hidden 0.4 m strip separates the start and goal on a flat measured surface.
The hidden heights never enter GP training. The same map without masking is
also checked for non-regression. Serialize actual enhancement-node outputs for
the production C++ planning harness; never fabricate probability layers.
"""
import argparse
import json
from pathlib import Path
import subprocess
import time
from unittest.mock import patch

import numpy as np
import rclpy
from rclpy.serialization import serialize_message
from sensor_msgs_py.point_cloud2 import create_cloud_xyz32
from std_msgs.msg import Header
from se2_grid_msgs.msg import SE2Grid
from fsgp_bgk_se2.fsgp_bgk_se2_node import FsgpBgkSe2Node
from fsgp_bgk_se2.se2_grid_tools import (get_layer_slices, grid_buffer_to_world,
    matrix_to_multiarray_like, world_to_grid_buffer)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--device",choices=("cpu","cuda"),default="cuda")
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    rclpy.init(args=["--ros-args","-p","prediction_backend:=feature_sgp","-p","gp_async:=false",
        "-p",f"gp_device:={args.device}","-p","gpu_backend:="+("cupy" if args.device=="cuda" else "cpu"),
        "-p","yaw_bins:=36","-p","gp_complete_supported:=true","-p","probability_use_gp_completion:=true"])
    node=FsgpBgkSe2Node()
    timings={};hot_timings={};cases=[];outputs={}
    try:
        x,y=np.meshgrid((np.arange(40)-19.5)*.2,(np.arange(40)-19.5)*.2)
        for name,hidden,gp in (("measured_baseline",False,False),("measured_gp",False,True),
                              ("strip_baseline",True,False),("strip_gp",True,True)):
            msg=SE2Grid();msg.info.header.frame_id="map";msg.info.header.stamp.sec=10
            msg.info.pos_resolution=.2;msg.info.length_x=msg.info.length_y=8.
            msg.info.length_yaw=2*np.pi;msg.info.yaw_resolution=2*np.pi/36
            observed=np.ones(x.shape,bool)
            if hidden:observed[:,19:21]=False
            z=np.where(observed,0.,np.nan).astype(np.float32)
            for layer,value in (("elevation",z),("var",np.full_like(z,.0001)),
                ("valid",observed.astype(np.float32)),("ground_observed",observed.astype(np.float32)),
                ("risk",np.where(observed,0.,.65).astype(np.float32))):
                msg.layers.append(layer);msg.has_so2.append(False)
                msg.data.append(matrix_to_multiarray_like(world_to_grid_buffer(value,msg),None))
            node.latest_cloud=create_cloud_xyz32(Header(frame_id="map",stamp=msg.info.header.stamp),
                np.column_stack((x[observed],y[observed],np.zeros(np.count_nonzero(observed)))))
            node.probability_use_gp_completion=gp
            completion=node.completion_predictor
            node.completion_predictor=completion if gp else None
            try:
                with patch.object(node,"_prediction_for_map",return_value=None):
                    begin=time.perf_counter();_,out=node._process_map(msg)
                    timings[name]=(time.perf_counter()-begin)*1000
                    hot_timings[name]=[]
                    for _ in range(3):
                        begin=time.perf_counter();node._process_map(msg)
                        hot_timings[name].append((time.perf_counter()-begin)*1000)
            finally:node.completion_predictor=completion
            outputs[name]=out
            (args.output/f"{name}.cdr").write_bytes(serialize_message(out))
            cases.append({"id":name,"scenario":"controlled_missing_strip","layout_id":0,"seed":71,
                "expected_stop":name=="strip_baseline","start":[-2.,0.,0.],"goal":[2.,0.,0.],
                "map_file":f"{name}.cdr","terrain":{"kind":"flat","scale":0},"obstacles":[]})
        for layer in ("probability_elevation","probability_variance","probability_valid","failure_probability_upper"):
            a=get_layer_slices(outputs["measured_baseline"],layer)[0]
            b=get_layer_slices(outputs["measured_gp"],layer)[0]
            for aa,bb in zip(a,b):np.testing.assert_array_equal(aa,bb)
        used=get_layer_slices(outputs["strip_gp"],"probability_gp_used")[0][0]
        counts={}
        for name,msg in outputs.items():
            layers=get_layer_slices(msg,"probability_valid")[0]
            counts[name]=sum(int(np.count_nonzero(s)) for s in layers)
        summary={"scope":"controlled observed-point dropout, not physical sensor occlusion or Isaac arrival",
            "fully_measured_nonregression":True,"completed_cells":int(np.count_nonzero(used)),
            "valid_pose_counts":counts,"first_map_enhancement_ms":timings,
            "three_hot_enhancement_ms":hot_timings}
        (args.output/"enhancement.json").write_text(json.dumps(summary,indent=2)+"\n")
        plan={"settings":{"timing_repeats":1,"qualification_only":True},"parameters":{
            "use_sim_time":True,"chance_constrained":True,"chance_allow_gp_completion":True,
            "live_obstacle_enable":False,"chance_state_limit":.05,"chance_path_budget":.10},"cases":cases}
        (args.output/"cases.json").write_text(json.dumps(plan,indent=2)+"\n")
    finally:node.destroy_node();rclpy.shutdown()
    root=Path(__file__).resolve().parents[3]
    subprocess.run([str(root/"build/terrain_nav/terrain_nav_query_benchmark"),
        str(args.output/"cases.json"),str(args.output/"planning")],check=True)
    rows=json.loads((args.output/"planning/results.json").read_text())["runs"]
    print(json.dumps({"enhancement":summary,"planning":[{k:r.get(k) for k in
        ("id","status","returned","full_goal_connector","independent_path_safe","planner_wall_ms","goal_residual_m")}
        for r in rows]},indent=2))
    for row in rows:
        expected=row["id"]!="strip_baseline"
        if bool(row.get("full_goal_connector"))!=expected:raise RuntimeError(f"qualification failed: {row['id']}")
        if expected and not row.get("independent_path_safe"):raise RuntimeError("unsafe path")


if __name__=="__main__":main()
