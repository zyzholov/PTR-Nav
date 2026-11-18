#!/usr/bin/env python3
"""Boundary qualification using real sensor/navigation/physics processes.

Truth and authored surfaces are evaluation-only. This observer publishes only
the requested relative goal, and a final stop. It supplies no maps or waypoints.
"""
import argparse
import json
import math
from pathlib import Path
import time
import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped,Twist
from navigation_trial import Trial,yaw_from_quaternion,truth_speed,point_to_polyline_distance
from clearance_metrics import static_course_clearances,sampled_body_clearances,localization_agreement
from boundary_courses import load_spec,fixtures,movers,goal_offset
from vehicle_geometry import swept_bounds,load_spec as load_vehicle_spec
from boundary_cloud_evidence import ClearingEvidence


def single_goal_contract(requests,acknowledgements,expected):
    """Audit external mission inputs, never internal rolling planner targets."""
    if len(requests)!=1 or len(acknowledgements)!=1:return False
    for row in (*requests,*acknowledgements):
        if row.get('frame')!='map' or row.get('source_stamp_s')!=expected['source_stamp_s']:return False
        for key in ('x','y','yaw'):
            value=row.get(key,math.nan)
            if not math.isfinite(value) or abs(value-expected[key])>1e-6:return False
    return True


def unexpected_goal(requests,acknowledgements,expected):
    if len(requests)>1 or len(acknowledgements)>1:return True
    return any(not single_goal_contract([row],[row],expected) for row in (*requests,*acknowledgements))


def localization_interlock_engaged(status_samples):
    """A latched navigation fault cannot recover during this unchanged mission."""
    if not status_samples:return False
    return any(row.get('name')=='terrain_nav/chance_constraints' and
               row.get('values',{}).get('localization_fault')=='1'
               for row in status_samples[-1].get('status',[]))


def current_dynamic_clearance_failure(truth,moving,bodies,geometry,minimum=.10):
    """Observer-only failure stop, using the SAME bracketed assessment geometry.

    Bounded recent buffers avoid rescanning the whole mission on every spin.
    Missing/unbracketed poses cannot prove either safety or a violation.
    This never supplies target truth, a route, or a passing result to navigation.
    """
    if not math.isfinite(minimum) or minimum<0:raise ValueError('invalid clearance threshold')
    earliest=None
    for body in bodies:
        samples=sampled_body_clearances(truth[-12:],moving.get(body['name'],[])[-10:],
            radius=body['radius'],**geometry)
        for stamp,clearance,gap in samples:
            if clearance<minimum and (earliest is None or stamp<earliest['stamp_s']):
                earliest=dict(body=body['name'],stamp_s=stamp,clearance_m=clearance,
                    required_m=minimum,bracketing_gap_s=gap,
                    scope='independent assessor failure stop; truth not provided to navigation')
    return earliest


def mission_inputs_ready(node):
    # The auditor itself subscribes to /goal_pose. Its subscription cannot
    # stand in for discovery of the actual planner, nor can a zero /clock.
    ready=(node.truth is not None and node.odom is not None and node.last_map is not None and node.clock_ns>0 and
            node.goal_pub.get_subscription_count()>=2 and
            any(e.node_name=='terrain_nav_node' for e in node.get_subscriptions_info_by_topic('/goal_pose')))
    if not ready:return False
    # Discovery can deliver the first old queued odometry/map before the
    # participant catches up with /clock. A goal stamped with that old pose
    # expires before fixed-stance qualification. Wait for CURRENT sources;
    # retain the original one-goal protocol and stale-ack rejection.
    for header,maximum in ((node.truth.header,.15),(node.odom.header,.35),(node.last_map.info.header,.35)):
        age=(node.clock_ns-header.stamp.sec*1_000_000_000-header.stamp.nanosec)*1e-9
        if not 0<=age<=maximum:return False
    return True


class BoundaryTrial(Trial):
    def __init__(self,spec,*,capture_motion=False):
        super().__init__('boundary',capture_motion=capture_motion);self.spec=spec;self.moving={o['name']:[] for o in movers(spec,0.)}
        self.clearing_evidence=ClearingEvidence(spec['kind']=='single_crossing')
        self.goal_acknowledgements=[];self.goal_requests=[]
        self.map_orientation=[]
        from rclpy.qos import QoSProfile,DurabilityPolicy
        self.create_subscription(PoseStamped,'/terrain_nav/received_goal',
            lambda m:self.on_goal_evidence(m,'/terrain_nav/received_goal',True),
            QoSProfile(depth=1,durability=DurabilityPolicy.TRANSIENT_LOCAL))
        for topic in ('/goal_pose','/move_base_simple/goal'):
            self.create_subscription(PoseStamped,topic,lambda m,t=topic:self.on_goal_evidence(m,t,False),10)
        for name in self.moving:
            self.create_subscription(PoseStamped,'/isaac/boundary/'+name+'/pose',
                lambda m,n=name:self.on_boundary_actor(m,n),10)

    def on_boundary_actor(self,msg,name):
        stamp=msg.header.stamp.sec+msg.header.stamp.nanosec*1e-9
        self.moving[name].append((stamp,msg.pose.position.x,msg.pose.position.y))
        self.clearing_evidence.actor(stamp,msg.pose.position.x,msg.pose.position.y)

    def on_laser_map(self,msg):
        self.clearing_evidence.retain('laser_map',msg,
            msg.header.stamp.sec+msg.header.stamp.nanosec*1e-9,self.clock_ns*1e-9)
        super().on_laser_map(msg)

    def on_registered(self,msg):
        self.clearing_evidence.retain('registered',msg,
            msg.header.stamp.sec+msg.header.stamp.nanosec*1e-9,self.clock_ns*1e-9)
        super().on_registered(msg)

    def on_map(self,msg):
        super().on_map(msg)
        if not self.clearing_evidence.enabled:return
        # Retain only requested layers, not the entire large posterior map.
        # Original arrays/info remain immutable; no numeric conversion here.
        from se2_grid_msgs.msg import SE2Grid
        from fsgp_bgk_se2.se2_grid_tools import layer_refs
        selected=SE2Grid();selected.info=msg.info
        selected.start_index_row=msg.start_index_row;selected.start_index_col=msg.start_index_col
        wanted=('collision_risk','probability_valid','failure_probability_upper','ground_observed',
                'terrain_observed','elevation','probability_elevation','collision_subcell_bits')
        refs=layer_refs(msg)
        for name in wanted:
            if name not in refs:continue
            layer=refs[name];selected.layers.append(name);selected.has_so2.append(layer.has_so2)
            selected.data.extend(msg.data[i]for i in layer.data_indices)
        stamp=msg.info.header.stamp.sec+msg.info.header.stamp.nanosec*1e-9
        self.clearing_evidence.retain('risk',selected,stamp,self.clock_ns*1e-9)

    def on_goal_evidence(self,msg,topic,acknowledgement):
        target=self.goal_acknowledgements if acknowledgement else self.goal_requests
        if len(target)>=100:raise ValueError('external goal evidence overflow')
        target.append(dict(topic=topic,frame=msg.header.frame_id,
            source_stamp_s=msg.header.stamp.sec+msg.header.stamp.nanosec*1e-9,
            receipt_sim_stamp_s=self.clock_ns*1e-9,x=msg.pose.position.x,y=msg.pose.position.y,
            yaw=yaw_from_quaternion(msg.pose.orientation)))

    def on_odom(self,msg):
        super().on_odom(msg)
        q=msg.pose.pose.orientation
        if len(self.map_orientation)<20000:
            self.map_orientation.append((msg.header.stamp.sec+msg.header.stamp.nanosec*1e-9,q.x,q.y,q.z,q.w))
        else:self.localization_overflow=True


def numeric_json(value):
    """Preserve measured values, including numpy scalars; reject other types."""
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,np.ndarray):return value.tolist()
    raise TypeError(f'unsupported measurement type: {type(value).__name__}')


def deadend_recovery_evidence(rows,gate,entry_x=1.8):
    """Observer-only exercise check, independent of controller commands."""
    if not rows or not gate:return dict(exercised=False,entered=False,gate_closed=False,reverse_travel_m=0.)
    a=np.asarray(rows);g=np.asarray(gate)
    moving=g[g[:,2]<2.99];closed=g[g[:,2]<.75]
    start=float(moving[0,0]) if len(moving) else math.inf
    # Verify that the robot actually entered the physical 1.4m-wide lane.
    entry_deadline=float(closed[0,0]) if len(closed) else start
    entered=bool(np.any((a[:,1]>=entry_x)&(a[:,1]<entry_x+2.2)&(abs(a[:,2])<.4)&(a[:,6]<=entry_deadline)))
    delta=np.diff(a[:,1:3],axis=0);turn=(np.diff(a[:,7])+np.pi)%(2*np.pi)-np.pi
    yaw=a[:-1,7]+.5*turn
    longitudinal=delta[:,0]*np.cos(yaw)+delta[:,1]*np.sin(yaw)
    dt=np.diff(a[:,6]);moving_reverse=(dt>1e-4)&(longitudinal<-.05*np.maximum(dt,0.))
    reverse=float(np.maximum(0.,-longitudinal[(a[:-1,6]>=start)&moving_reverse]).sum())
    return dict(exercised=bool(entered and len(closed) and reverse>=.8),entered=entered,
                gate_closed=bool(len(closed)),reverse_travel_m=reverse,
                scope='truth observer only; no external intermediate targets or controller-command proxy')


def await_mission_gate(node,prepared_file,start_file,timeout=20.):
    """Initialize observation before timed measurement; never authorize motion.

    The pipeline releases the gate only AFTER its unfiltered stream monitor
    has armed. Recheck live inputs before deriving/publishing the single goal.
    """
    if prepared_file is None and start_file is None:return
    if prepared_file is None or start_file is None:raise ValueError('both observer gate paths are required')
    prepared_file=Path(prepared_file);start_file=Path(start_file)
    if prepared_file.exists() or start_file.exists():raise ValueError('observer gate paths must be new')
    temporary=prepared_file.with_suffix('.tmp')
    temporary.write_text(json.dumps(dict(observer_initialized=True,clock_ns=node.clock_ns,
        wall_monotonic_s=time.monotonic(),scope='observer ready; no mission goal sent'))+'\n')
    temporary.replace(prepared_file)
    until=time.monotonic()+timeout
    while time.monotonic()<until:
        rclpy.spin_once(node,timeout_sec=.05)
        if start_file.is_file() and mission_inputs_ready(node):return
    raise RuntimeError('mission monitor gate did not become ready with fresh inputs')


def run(node,args,spec):
    until=time.monotonic()+25
    while time.monotonic()<until:
        rclpy.spin_once(node,timeout_sec=.05)
        if mission_inputs_ready(node):break
    else:raise RuntimeError('no live boundary inputs')
    await_mission_gate(node,getattr(args,'prepared_file',None),getattr(args,'start_file',None))
    p=node.truth.pose.pose.position;op=node.odom.pose.pose.position
    yaw=yaw_from_quaternion(node.odom.pose.pose.orientation)
    tyaw=yaw_from_quaternion(node.truth.pose.pose.orientation)
    start=(p.x,p.y,p.z);dx,dy=goal_offset(spec)
    target=(p.x+dx*math.cos(tyaw)-dy*math.sin(tyaw),p.y+dx*math.sin(tyaw)+dy*math.cos(tyaw))
    goal=PoseStamped();goal.header.frame_id='map';goal.header.stamp=node.odom.header.stamp
    goal.pose.position.x=op.x+dx*math.cos(yaw)-dy*math.sin(yaw)
    goal.pose.position.y=op.y+dx*math.sin(yaw)+dy*math.cos(yaw);goal.pose.orientation=node.odom.pose.pose.orientation
    node.truth_samples.clear();node.recording=True;node.truth_start=start;node.truth_yaw_start=tyaw
    expected_goal=dict(frame='map',source_stamp_s=goal.header.stamp.sec+goal.header.stamp.nanosec*1e-9,
                       x=goal.pose.position.x,y=goal.pose.position.y,yaw=yaw_from_quaternion(goal.pose.orientation))
    node.goal_requests.clear();node.goal_acknowledgements.clear()
    wall=time.monotonic();t0=node.clock_ns*1e-9;node.goal_pub.publish(goal)
    settled_since=None;reached=False;settled=False;failure=None;expired_input_ages=None
    clearance_abort=None;assessment_geometry=swept_bounds();assessment_bodies=movers(spec,0.)
    while time.monotonic()-wall<args.timeout:
        rclpy.spin_once(node,timeout_sec=.04)
        if unexpected_goal(node.goal_requests,node.goal_acknowledgements,expected_goal):
            failure='external mission goal changed during single-goal qualification';break
        if localization_interlock_engaged(node.safety_status_samples):
            failure='navigation localization interlock engaged; mission cannot recover without restart';break
        clearance_abort=current_dynamic_clearance_failure(node.truth_samples,node.moving,
            assessment_bodies,assessment_geometry)
        if clearance_abort is not None:
            failure='independent dynamic body clearance below unchanged qualification minimum';break
        if not node.goal_acknowledgements and time.monotonic()-wall>5.:
            failure='initial mission goal not acknowledged by planner';break
        input_ages={name:time.monotonic()-stamp for name,stamp in
                    (('odometry',node.odom_wall),('simulation_truth',node.truth_wall),('risk_map',node.map_wall))}
        if max(input_ages.values())>1.:
            expired_input_ages=input_ages
            failure='input stream expired';break
        p=node.truth.pose.pose.position;speed=truth_speed(node.truth_samples)
        reached=math.hypot(p.x-target[0],p.y-target[1])<=.3
        resting=reached and speed is not None and speed<.05 and abs(node.last_command[0])<.03
        settled_since=(settled_since if settled_since is not None else node.clock_ns*1e-9) if resting else None
        settled=settled_since is not None and node.clock_ns*1e-9-settled_since>=.5
        if settled:break
        if node.truth_samples and max(abs(v) for v in node.truth_samples[-1][4:6])>.60:
            failure='attitude safety abort';break
    stop=node.create_publisher(Twist,'/cmd_vel',10);stop.publish(Twist())
    end_file=args.output.parent/'navigation_motion_end.json'
    temporary=end_file.with_suffix('.tmp')
    temporary.write_text(json.dumps(dict(last_truth_stamp_s=node.truth_samples[-1][6] if node.truth_samples else None,
        clock_stamp_s=node.clock_ns*1e-9,failure_reason=failure,
        scope='assessor final stop boundary; no map or target truth supplied to navigation'))+'\n')
    temporary.replace(end_file)
    from rclpy.serialization import serialize_message
    clearing_manifest=node.clearing_evidence.archive(args.output.parent/'clearing_clouds',serialize_message)
    if failure is None and not settled:failure='mission timeout before settled arrival'
    rows=node.truth_samples
    if len(rows)<20:raise RuntimeError('insufficient independent truth samples')
    static=[o for o in fixtures(spec) if not o['terrain']]
    vehicle=swept_bounds();physical_spec=load_vehicle_spec()
    clearances=static_course_clearances(rows,static,**vehicle)
    minima={n:min(v for _,v in a) for n,a in clearances.items()}
    moving_results={}
    for o in movers(spec,0.):
        values=sampled_body_clearances(rows,node.moving[o['name']],radius=o['radius'],**vehicle)
        moving_results[o['name']]=dict(minimum_m=min((r[1] for r in values),default=None),samples=len(values))
    chassis_min=math.inf
    terrain=[o for o in fixtures(spec) if o['terrain']]
    # Reference is the actual URDF body box underside, not wheel-ground contact.
    # It is deliberately separate from the larger planar collision envelope.
    if terrain:
        from vehicle_geometry import geometry,chassis_vertical_bounds
        chassis=geometry();bottom=chassis_vertical_bounds()[0]
        u,v=np.meshgrid(np.linspace(-chassis['back'],chassis['front'],15),np.linspace(-chassis['half_width'],chassis['half_width'],11));u=u.ravel();v=v.ravel()
        for row in rows:
            _,x,y,z,pitch,roll,_,heading=row
            cp,sp=np.cos(pitch),np.sin(pitch);cr,sr=np.cos(roll),np.sin(roll)
            bx=cp*u+sp*sr*v+sp*cr*bottom;by=cr*v-sr*bottom
            wz=z-sp*u+cp*sr*v+cp*cr*bottom
            wx=x+np.cos(heading)*bx-np.sin(heading)*by;wy=y+np.sin(heading)*bx+np.cos(heading)*by
            h=np.full_like(wx,-.02)
            for o in terrain:
                inside=(abs(wx-o['x'])<=o['length']/2)&(abs(wy-o['y'])<=o['width']/2)
                h=np.where(inside,np.maximum(h,-.02+o['height']),h)
            chassis_min=min(chassis_min,float(np.min(wz-h)))
    accepted_status=[];violations=[]
    for sample in node.safety_status_samples:
        for status in sample['status']:
            v=status['values']
            if v.get('command_accepted')!='1':continue
            accepted_status.append(sample['stamp_s'])
            try:
                state=float(v['state_probability_upper']);brake=float(v['braking_probability_upper'])
                ok=math.isfinite(state) and 0<=state<=float(v['state_limit'])+1e-7
                ok&=math.isfinite(brake) and 0<=brake<=float(v['braking_budget'])+1e-7
            except (KeyError,ValueError):ok=False
            if not ok:violations.append(sample)
    a=np.asarray(rows);cmd=np.asarray(node.control_samples)
    dt=np.diff(cmd[:,0]) if len(cmd)>1 else np.array([])
    commanded=(np.maximum(abs(cmd[:-1,1]),abs(cmd[1:,1]))>.03) if len(cmd)>1 else np.array([],bool)
    active_dt=dt[commanded]
    localization=localization_agreement(node.localization_poses,node.localization_truth)
    body_ok=all(v>=.10 for v in minima.values()) and all(v['samples']>=20 and v['minimum_m']>=.10 for v in moving_results.values())
    operational_angle=math.radians(physical_spec['max_climb_deg'])
    physical_ok=body_ok and (not terrain or chassis_min>=-.005) and np.max(abs(a[:,4]))<=operational_angle and np.max(abs(a[:,5]))<=operational_angle
    temporal_ok=bool(node.map_ages) and max(node.map_ages)<=.5 and len(accepted_status)>2
    recorder_ok=not (node.safety_status_overflow or node.localization_overflow or node.motion_grid_overflow)
    one_goal=single_goal_contract(node.goal_requests,node.goal_acknowledgements,expected_goal)
    recovery_evidence=None
    if spec['kind'] in ('deadend_entry','deadend_entry_observed_approach'):
        recovery_evidence=deadend_recovery_evidence(rows,node.moving.get('ClosingGate',[]),
            entry_x=3.8 if spec['kind']=='deadend_entry_observed_approach' else 1.8)
        if not recovery_evidence['exercised'] and failure is None:failure='deadend entry/closure/actual reverse exercise incomplete'
    passed=bool(reached and settled and physical_ok and temporal_ok and localization['accepted'] and not violations and not failure and recorder_ok and one_goal)
    summary=dict(scenario='boundary',kind=spec['kind'],reached=reached,settled=settled,
        truth_error_m=math.hypot(a[-1,1]-target[0],a[-1,2]-target[1]),duration_s=float(a[-1,6]-t0),
        truth_path_length_m=float(np.linalg.norm(np.diff(a[:,1:3],axis=0),axis=1).sum()),
        risk_map_age_p95_s=float(np.quantile(node.map_ages,.95)) if node.map_ages else None,
        risk_map_age_max_s=max(node.map_ages,default=None),source_timebase='simulation stamp age; command intervals are not CPU timing',
        command_interval_p95_s=float(np.quantile(dt,.95)) if len(dt) else None,
        command_interval_max_s=float(max(dt)) if len(dt) else None,accepted_status_samples=len(accepted_status),
        nonzero_command_interval_p95_s=float(np.quantile(active_dt,.95)) if len(active_dt) else None,
        nonzero_command_interval_max_s=float(max(active_dt)) if len(active_dt) else None,
        active_interval_scope='adjacent intervals with at least one nonzero speed command; zero-speed coasting is not covered by this subset',
        reported_probability_violations=len(violations),risk_contract_scope='audit of reported conditional bounds; not empirical probability calibration',
        static_clearance_minima_m=minima,moving_clearance=moving_results,
        chassis_to_terrain_min_m=chassis_min if terrain else None,
        pitch_max_rad=float(max(abs(a[:,4]))),roll_max_rad=float(max(abs(a[:,5]))),
        tracking_error_p95_m=float(np.quantile([r[1] for r in node.tracking_errors],.95)) if node.tracking_errors else None,
        route_scope='single requested goal; no author-provided intermediate route',single_external_goal=one_goal,failure_reason=failure,
        input_stream_wall_ages_at_abort_s=expired_input_ages,
        independent_clearance_abort=clearance_abort,
        goal_beyond_initial_local_map=bool(abs(dx)>10.4 or abs(dy)>10.4),physical_ok=bool(physical_ok),temporal_ok=bool(temporal_ok),recorder_ok=recorder_ok,
        evaluation_body_geometry=vehicle,vehicle_specification=physical_spec)
    if recovery_evidence is not None:summary['deadend_recovery']=recovery_evidence
    data=dict(schema_version='boundary_v1',accepted=passed,summary=summary,spec=spec,truth_start=start,truth_goal=target,
        truth_columns=['wall_s','x','y','z','pitch','roll','stamp_s','yaw'],truth=rows,
        controls=node.control_samples,path_history=node.path_history,probability_violations=violations,
        safety_status=node.safety_status_samples,moving_obstacles=node.moving,localization_assessment=localization,
        clearing_cloud_evidence=clearing_manifest,
        requested_goal=expected_goal,goal_requests=node.goal_requests,goal_acknowledgements=node.goal_acknowledgements,
        map_orientation_columns=['stamp_s','qx','qy','qz','qw'],map_orientation=node.map_orientation,
        localization_pose_columns=['stamp_s','x','y','z','yaw'],localization_poses=node.localization_poses,
        localization_attitude_columns=['stamp_s','qx','qy','qz','qw'],localization_attitudes=node.localization_attitudes,
        scope='physical probability-navigation boundary qualification; truth never feeds navigation')
    args.output.write_text(json.dumps(data,indent=2,allow_nan=False,default=numeric_json)+'\n')
    if args.motion_grid_output:
        args.motion_grid_output.write_text(json.dumps(dict(schema_version=1,overflow=node.motion_grid_overflow,
            scope='raw measured columns, not traversability truth',frames=node.motion_grids,
            tracking_frames=node.motion_tracking_grids,
            tracking_scope='actual tracker input, including measured upper witnesses; not collision/free-space evidence',
            topology_grids=node.topology_grids,topology_routes=node.topology_routes,
            topology_scope='optimistic guidance only; no execution certificate'),allow_nan=False,default=numeric_json)+'\n')
    print(json.dumps(summary,allow_nan=False,default=numeric_json),flush=True);print('PASS' if passed else 'FAIL',flush=True)
    return int(not passed)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--spec',required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--timeout',type=float,default=100.)
    p.add_argument('--motion-grid-output',type=Path)
    p.add_argument('--prepared-file',type=Path)
    p.add_argument('--start-file',type=Path)
    p.add_argument('--validate-only',action='store_true',help='Resolve scene/vehicle contracts without starting ROS or publishing')
    a=p.parse_args();spec=load_spec(a.spec)
    if a.validate_only:
        print(json.dumps(dict(scene=spec,vehicle=load_vehicle_spec(),body=swept_bounds(),goal=goal_offset(spec)),ensure_ascii=False))
        return 0
    rclpy.init();node=BoundaryTrial(spec,capture_motion=a.motion_grid_output is not None)
    try:return run(node,a,spec)
    finally:
        node.create_publisher(Twist,'/cmd_vel',10).publish(Twist())
        node.destroy_node();rclpy.shutdown()


if __name__=='__main__':raise SystemExit(main())
