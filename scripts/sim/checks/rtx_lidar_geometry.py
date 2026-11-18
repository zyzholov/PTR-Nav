#!/usr/bin/env python3
"""Isaac-only geometry qualification: raycast a known plane at three pitches.

Run with Isaac's python.sh, not the system ROS Python. Ground truth is confined
to this isolated test; no test points or corrections enter the navigation stack.
"""
import json
import sys
import argparse
from pathlib import Path
import numpy as np
from isaacsim import SimulationApp

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('output',type=Path)
parser.add_argument('--inspect-gmo',action='store_true')
args,_=parser.parse_known_args()

app=SimulationApp({'headless':True,'width':320,'height':240})
import omni.usd
import omni.replicator.core as rep
from isaacsim.core.utils.extensions import enable_extension
from isaacsim.core.api import SimulationContext
from pxr import Gf,UsdGeom
enable_extension('isaacsim.sensors.rtx')
sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'src/isaac_sim_hunter_livo2/scripts'))
from mid360_scan import Mid360ScanTable,RtxMid360Scan
from isaacsim.sensors.rtx import get_gmo_data

stage=omni.usd.get_context().get_stage()
UsdGeom.SetStageUpAxis(stage,UsdGeom.Tokens.z);UsdGeom.SetStageMetersPerUnit(stage,1.)
floor=UsdGeom.Cube.Define(stage,'/floor');floor.GetSizeAttr().Set(1.)
xf=UsdGeom.XformCommonAPI(floor);xf.SetTranslate(Gf.Vec3d(0,0,-.5));xf.SetScale(Gf.Vec3f(200,200,1))
table=Mid360ScanTable();sensors=[]
for pitch in (0.,25.,-25.):
    mount=stage.DefinePrim('/mount'+str(int(pitch)).replace('-','m'),'Xform')
    pose=UsdGeom.XformCommonAPI(mount);pose.SetTranslate(Gf.Vec3d(0,0,1.));pose.SetRotate((0.,pitch,0.))
    prim=stage.DefinePrim(str(mount.GetPath())+'/sensor','OmniLidar');prim.ApplyAPI('OmniSensorGenericLidarCoreAPI')
    scan=RtxMid360Scan(prim,table)
    if args.inspect_gmo:
        prim.GetAttribute('omni:sensor:Core:skipDroppingInvalidPoints').Set(True)
        prim.GetAttribute('omni:sensor:Core:auxOutputType').Set('FULL')
    product=rep.create.render_product(prim.GetPath(),(1,1),render_vars=['GenericModelOutput','RtxSensorMetadata'])
    ann=rep.AnnotatorRegistry.get_annotator('IsaacExtractRTXSensorPointCloudNoAccumulator')
    ann.attach([product.path])
    raw=rep.AnnotatorRegistry.get_annotator('GenericModelOutput') if args.inspect_gmo else None
    if raw is not None:raw.attach([product.path])
    sensors.append((pitch,scan,prim,ann,raw))
ctx=SimulationContext(physics_dt=.01,rendering_dt=.1,stage_units_in_meters=1.)
ctx.initialize_physics();ctx.play();records=[]
try:
    for i in range(15):
        for _,scan,_,_,_ in sensors:scan.update(i*.1)
        for _ in range(10):ctx.step(render=False)
        ctx.render()
        if i<5:continue
        for pitch,_,prim,ann,raw in sensors:
            data=ann.get_data();p=np.asarray(data.get('data',[])).reshape(-1,3)
            p=p[np.isfinite(p).all(axis=1)&(np.linalg.norm(p,axis=1)>.2)]
            if len(p)<30:continue
            c=np.mean(p,axis=0);_,_,v=np.linalg.svd(p-c,full_matrices=False)
            n=v[-1];n*=np.sign(n[2]);a=np.deg2rad(pitch)
            r=np.array([[np.cos(a),0,np.sin(a)],[0,1,0],[-np.sin(a),0,np.cos(a)]])
            world=p@r.T+np.array([0,0,1.])
            row=dict(frame=i,pitch=pitch,points=len(p),normal_sensor=n.tolist(),
                     z_world_median=float(np.median(world[:,2])),
                     z_world_abs_p95=float(np.quantile(np.abs(world[:,2]),.95)),
                     raw_z_median=float(np.median(p[:,2])),
                     transform=np.asarray(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0)).tolist())
            if raw is not None:
                buffer=raw.get_data()  # Keep the backing buffer alive for pointer-backed GMO fields.
                gmo=get_gmo_data(buffer)
                flags=np.asarray(gmo.flags);distance=np.asarray(gmo.z)
                row['gmo']=dict(elements=gmo.numElements,stamp_ns=gmo.timestampNs,
                    frame_id=gmo.frameId,scan_complete=gmo.scanComplete,
                    coords=str(gmo.elementsCoordsType),flags=np.unique(flags,return_counts=True)[0].tolist(),
                    flag_counts=np.unique(flags,return_counts=True)[1].tolist(),
                    distance_min=float(np.nanmin(distance)),distance_max=float(np.nanmax(distance)),
                    zero_distances=int(np.count_nonzero(distance==0)),
                    first_x=np.asarray(gmo.x)[:5].tolist(),first_y=np.asarray(gmo.y)[:5].tolist(),
                    first_z=distance[:5].tolist(),first_flags=flags[:5].tolist())
            records.append(row);print('GEOMETRY '+json.dumps(row),flush=True)
    by_pitch={str(p):[r for r in records if r['pitch']==p] for p in (0.,25.,-25.)}
    passed=all(len(rows)==10 and max(r['z_world_abs_p95'] for r in rows)<.015
               and max(abs(r['z_world_median']) for r in rows)<.005 for rows in by_pitch.values())
    result=dict(passed=passed,scope='actual RTX known-plane geometry at 0,+25,-25deg; not navigation',
                world_height_p95_limit_m=.015,world_height_median_limit_m=.005,records=records)
    # Kit may terminate the interpreter in app.close(); retain the artifact first.
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print('GEOMETRY_RESULT '+json.dumps(dict(passed=passed,frames=len(records))),flush=True)
    if not passed:raise RuntimeError('RTX known-plane geometry qualification failed')
finally:
    ctx.stop();app.close()
