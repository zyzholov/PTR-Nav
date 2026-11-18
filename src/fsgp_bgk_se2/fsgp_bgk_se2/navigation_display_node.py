"""Read-only navigation presentation: actual track, command projection, probe.

No goal or command publishers and no navigation map modifications. The sigma
probe depicts a marginal height model, not a calibrated safety certificate.
"""
from collections import deque
import math
import time

import numpy as np
import rclpy
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from geometry_msgs.msg import Point, PointStamped, PoseStamped, Twist
from nav_msgs.msg import Odometry, Path
from sensor_msgs.msg import PointCloud2
from rclpy.node import Node
from rclpy.qos import QoSProfile, qos_profile_sensor_data
from rclpy.serialization import deserialize_message
from se2_grid_msgs.msg import SE2Grid
from visualization_msgs.msg import Marker, MarkerArray

from .display_geometry import command_projection, project_path
from .terrain_visuals import terrain_products
from .se2_grid_tools import grid_buffer_to_world, layer_refs, multiarray_to_matrix
from .probabilistic_risk import ProbabilityParams,_geometry
from .grid_cdr import deserialize_grid


def stamp_s(stamp):
    return stamp.sec+stamp.nanosec*1e-9


def yaw_of(q):
    norm=q.x*q.x+q.y*q.y+q.z*q.z+q.w*q.w
    if not math.isfinite(norm) or norm<1e-12: return None
    return math.atan2(2*(q.w*q.z+q.x*q.y)/norm,1-2*(q.y*q.y+q.z*q.z)/norm)


def inspect_grid(msg,x,y,yaw,state_limit=.05):
    res=float(msg.info.pos_resolution)
    if res<=0 or not all(math.isfinite(v) for v in (x,y,yaw,res,state_limit)) or not 0<state_limit<1: return None
    ix=math.floor((x-msg.info.pose.position.x+msg.info.length_x/2)/res)
    iy=math.floor((y-msg.info.pose.position.y+msg.info.length_y/2)/res)
    if not 0<=ix<round(msg.info.length_x/res) or not 0<=iy<round(msg.info.length_y/res): return None
    refs=layer_refs(msg)
    result={"x":msg.info.pose.position.x-msg.info.length_x/2+(ix+.5)*res,
            "y":msg.info.pose.position.y-msg.info.length_y/2+(iy+.5)*res,
            "stamp_s":stamp_s(msg.info.header.stamp)}
    collision_matrix=None;fine_matrix=None
    for name in ("elevation","var","valid","ground_observed","terrain_observed","probability_gp_used",
                 "gp_elevation_observed","gp_variance_observed","gp_mean","gp_latent_var","gp_candidate","gp_prediction_age",
                 "startup_prior_used","probability_elevation","probability_variance","probability_joint_model","probability_joint_used",
                 "probability_valid","failure_probability_upper","probability_pose_event","height_error_tail_upper","probability_height_std_floor",
                 "probability_variance_scale","ground_pending","ground_pending_effective","ground_pending_column_side","ground_pending_elevation",
                 "ground_pending_variance","probability_raw_veto","collision_risk",
                 "probability_front","probability_back","probability_half_width",
                 "probability_contract","probability_max_pitch","probability_max_roll","probability_max_step",
                 "probability_max_roughness","probability_ground_clearance","probability_underbody_margin",
                 "probability_floor_covariance_model",
                 "collision_subcell_contract","collision_subcell_factor","collision_subcell_bits",
                 "collision_body_front","collision_body_back","collision_body_half_width",
                 "terrain_failure_probability_upper"):
        ref=refs.get(name)
        if ref is None: continue
        index=int((yaw%(2*math.pi))/max(msg.info.yaw_resolution,1e-9)) if ref.has_so2 else 0
        data_index=ref.data_indices[min(index,len(ref.data_indices)-1)]
        if data_index>=len(msg.data): raise ValueError("probe layer lies beyond message")
        matrix=grid_buffer_to_world(multiarray_to_matrix(msg.data[data_index]),msg)
        result[name]=float(matrix[iy,ix])
        if name=='collision_risk':collision_matrix=matrix
        if name=='collision_subcell_bits':fine_matrix=matrix
    source="unknown";height=math.nan;variance=math.nan
    measured=result.get("terrain_observed",result.get("ground_observed",result.get("valid",1)))
    raw_height=result.get("gp_elevation_observed",result.get("elevation",math.nan))
    pending=result.get('ground_pending_effective',result.get('ground_pending',0))>.5
    if pending:
        source='ground transition pending'
    elif math.isfinite(measured) and measured>.5 and math.isfinite(raw_height):
        source="observed";height=raw_height
        variance=result.get("gp_variance_observed",result.get("var",math.nan))
    elif result.get("startup_prior_used",0)>.5:
        source="initial assumption (simulation)";height=result.get("probability_elevation",math.nan)
        variance=result.get("probability_variance",math.nan)
    elif result.get("probability_gp_used",0)>.5 and math.isfinite(result.get("probability_elevation",math.nan)):
        source="GP navigation support";height=result["probability_elevation"];variance=result.get("probability_variance",math.nan)
    elif result.get("gp_candidate",0)>.5 and math.isfinite(result.get("gp_mean",math.nan)) and result.get("gp_latent_var",-1)>=0:
        source="GP candidate (not navigation)";height=result["gp_mean"];variance=result["gp_latent_var"]
    elif math.isfinite(result.get("elevation",math.nan)) and result.get("valid",1)>.5:
        source="restored, not measured"
    result["source"]=source;result["display_height"]=height
    if math.isfinite(variance) and variance>=0:
        result["raw_sigma"]=math.sqrt(variance)
        floor=result.get("probability_height_std_floor",math.nan)
        scale=result.get("probability_variance_scale",math.nan)
        if math.isfinite(floor) and floor>=0 and math.isfinite(scale) and scale>0:
            result["model_sigma"]=math.sqrt(scale*variance+floor*floor)
    nav_variance=result.get('probability_variance',math.nan)
    floor=result.get('probability_height_std_floor',math.nan);scale=result.get('probability_variance_scale',math.nan)
    if math.isfinite(nav_variance) and nav_variance>=0 and math.isfinite(floor) and floor>=0 and math.isfinite(scale) and scale>0:
        result['navigation_sigma']=math.sqrt(scale*nav_variance+floor*floor)
    occupied=result.get('collision_risk',0)>=.5
    dimensions=[result.get(n,math.nan) for n in ('probability_front','probability_back','probability_half_width')]
    if collision_matrix is not None and math.isfinite(msg.info.yaw_resolution) and msg.info.yaw_resolution>0 and all(math.isfinite(v) and v>0 for v in dimensions):
        bins=max(1,round(2*math.pi/msg.info.yaw_resolution))
        physical=ProbabilityParams(resolution=res,yaw_bins=bins,front=dimensions[0],back=dimensions[1],
                                   half_width=dimensions[2],margin=0.)
        k=int(yaw%(2*math.pi)/msg.info.yaw_resolution)%bins
        occupied=any(0<=iy+dr<collision_matrix.shape[0] and 0<=ix+dc<collision_matrix.shape[1] and
            collision_matrix[iy+dr,ix+dc]>=.5 for dr,dc in _geometry(physical,k)[0])
    result['footprint_collision']=bool(occupied)
    upper=result.get('failure_probability_upper',math.nan)
    modern=result.get('collision_subcell_contract')==3
    if modern:upper=math.nan;result['body_geometry_known']=False
    if modern and result.get('collision_subcell_factor') in (2,4) and collision_matrix is not None and fine_matrix is not None:
        from .obstacle_columns import inspect_body_collision
        factor=int(result['collision_subcell_factor']);bounds=None
        names=[f'collision_subcell_xy_bounds_{k}' for k in range(factor*factor)]
        tag=refs.get('collision_subcell_xy_bounds_contract')
        if tag is not None and not tag.has_so2 and all(n in refs and not refs[n].has_so2 for n in names):
            contract=grid_buffer_to_world(multiarray_to_matrix(msg.data[tag.data_indices[0]]),msg)
            if contract[iy,ix]==1:
                bounds=np.stack([grid_buffer_to_world(multiarray_to_matrix(msg.data[refs[n].data_indices[0]]),msg) for n in names])
        body=[result.get(n,math.nan) for n in ('collision_body_front','collision_body_back','collision_body_half_width')]
        if all(math.isfinite(v) and v>0 for v in body):
            occupied,geometry_known=inspect_body_collision(collision_matrix,fine_matrix,
                (msg.info.pose.position.x-msg.info.length_x/2,msg.info.pose.position.y-msg.info.length_y/2),
                res,x,y,yaw,*body,factor=factor,bounds=bounds)
            result['body_geometry_known']=geometry_known
            result['footprint_collision']=bool(occupied and geometry_known)
            # Keep the published raster value as a separate diagnostic. It is
            # the representative cell-centre/yaw pose, not this actual query.
            upper=(1. if occupied and geometry_known else
                   result.get('terrain_failure_probability_upper',math.nan) if geometry_known else math.nan)
    result['effective_pose_probability_upper']=upper
    result['query_x']=x;result['query_y']=y;result['query_yaw']=yaw
    valid=result.get('probability_valid',0)>.5 and math.isfinite(upper)
    if modern and result.get('body_geometry_known') is False:code,text='geometry_unknown','完整车身的几何支撑窗口未知'
    elif occupied:code,text='collision','整车包络存在碰撞占用' if modern else '地形接触足迹内存在碰撞占用；另有车身包络检查'
    elif result.get('probability_raw_veto',0)>.5:code,text='raw_veto','当前回波与地面支撑矛盾'
    elif pending:code,text='pending','地面变化尚待观测确认'
    elif not valid and 'probability_valid' in result:
        code,text=('footprint_support','查询格有高程，地形接触足迹支撑不完整') if measured>.5 else ('query_support','查询格或地形接触足迹支撑未知')
    elif valid and upper>state_limit:code,text='terrain_limit','整车地形风险上界超过姿态预算'
    elif valid:code,text='pose_ok','姿态约束满足；路径与制动须另查'
    else:code,text='unavailable','概率风险数据不可用'
    result['risk_cause_code']=code;result['risk_reason']=text
    return result


class NavigationDisplay(Node):
    def __init__(self):
        super().__init__("navigation_display")
        self.frame=self.declare_parameter("frame_id","map").value
        self.presentation_context=str(self.declare_parameter("presentation_context","LIVE").value)
        self.wheel_base=float(self.declare_parameter("wheel_base",.549).value)
        self.steer_sign=float(self.declare_parameter("steer_output_sign",1.).value)
        if self.wheel_base<=0 or not math.isfinite(self.wheel_base) or self.steer_sign not in (-1.,1.):
            raise ValueError("invalid visualization vehicle model")
        odom_topic=self.declare_parameter("odom_topic","/Odometry").value
        grid_topic=self.declare_parameter("grid_topic","/car_0_fsgp_bgk/fused_map").value
        cmd_topic=self.declare_parameter("cmd_topic","/cmd_vel").value
        self.odom=None; self.grid=None; self.path=[]; self.command=None
        self.last_products=None;self.product_info={};self.product_wall=-math.inf
        self.map_points=0;self.map_stamp=-math.inf
        self.command_received=-math.inf; self.status={}; self.status_stamp=-math.inf
        self.status_message="planner unavailable"
        self.invalid_display_packets={}
        self.selection=None; self.probe=None; self.probe_wall=-math.inf
        self.trail=deque(maxlen=1200); self.last_odom_stamp=-math.inf
        self.markers=self.create_publisher(MarkerArray,"/terrain_nav/presentation",QoSProfile(depth=1))
        self.track=self.create_publisher(Path,"/terrain_nav/executed_path",QoSProfile(depth=1))
        self.readout=self.create_publisher(DiagnosticArray,"/terrain_nav/display_status",QoSProfile(depth=1))
        self.terrain_publishers={name:self.create_publisher(PointCloud2,"/terrain_display/"+name,QoSProfile(depth=1))
            for name in ("measured_height","predicted_height","prediction_support","navigation_prediction","sources","startup_height")}
        for cls,topic,callback,qos in (
            (Odometry,odom_topic,self.on_odom,qos_profile_sensor_data),
            (SE2Grid,grid_topic,self.on_grid,QoSProfile(depth=1)),
            (Path,"/terrain_nav/path",self.on_path,QoSProfile(depth=1)),
            (Twist,cmd_topic,self.on_command,QoSProfile(depth=1)),
            (PointStamped,"/clicked_point",self.on_click,QoSProfile(depth=1)),
            (DiagnosticArray,"/terrain_nav/chance_status",self.on_status,QoSProfile(depth=1)),
            (PointCloud2,"/Laser_map",self.on_accumulated,QoSProfile(depth=1))):
            # Decode inside an attributable error boundary. An unconvertible
            # DDS sample must not kill the entire display in take_message().
            # Retained good samples keep their ORIGINAL age and expire normally.
            self.create_subscription(cls,topic,
                lambda payload,c=cls,t=topic,f=callback:self.consume_payload(payload,c,t,f),qos,raw=True)
        self.create_timer(.1,self.render)

    def consume_payload(self,payload,cls,topic,callback):
        try:msg=deserialize_grid(payload) if cls is SE2Grid else deserialize_message(payload,cls)
        except Exception as exc:
            self.invalid_display_packets[topic]=self.invalid_display_packets.get(topic,0)+1
            self.get_logger().warn(f'display rejected serialized input on {topic}: {type(exc).__name__}: {exc}',
                                   throttle_duration_sec=2.)
            return
        callback(msg)

    def on_grid(self,msg):
        if msg.info.header.frame_id==self.frame: self.grid=msg

    def on_accumulated(self,msg):
        if msg.header.frame_id==self.frame:
            self.map_points=msg.width*msg.height;self.map_stamp=stamp_s(msg.header.stamp)

    def on_path(self,msg):
        if msg.header.frame_id==self.frame:
            self.path=[(p.pose.position.x,p.pose.position.y,p.pose.position.z) for p in msg.poses]

    def on_command(self,msg):
        self.command=msg; self.command_received=self.get_clock().now().nanoseconds*1e-9

    def on_status(self,msg):
        self.status={p.key:p.value for s in msg.status if s.name=="terrain_nav/chance_constraints" for p in s.values}
        self.status_message=next((s.message for s in msg.status if s.name=="terrain_nav/chance_constraints"),"status unavailable")
        self.status_stamp=stamp_s(msg.header.stamp)

    def on_click(self,msg):
        if msg.header.frame_id==self.frame: self.selection=(msg.point.x,msg.point.y)

    def on_odom(self,msg):
        p=msg.pose.pose.position; t=stamp_s(msg.header.stamp)
        if msg.header.frame_id!=self.frame or yaw_of(msg.pose.pose.orientation) is None: return
        if not all(math.isfinite(v) for v in (p.x,p.y,p.z,t)): return
        previous=self.trail[-1].pose.position if self.trail else None
        if t<self.last_odom_stamp or previous and math.hypot(p.x-previous.x,p.y-previous.y)>4:
            self.trail.clear(); previous=None; self.selection=None; self.command=None
        if previous is None or math.hypot(p.x-previous.x,p.y-previous.y)>=.06:
            pose=PoseStamped(); pose.header=msg.header; pose.pose=msg.pose.pose
            self.trail.append(pose)
        self.odom=msg; self.last_odom_stamp=t

    def marker(self,index,namespace,kind,color):
        m=Marker(); m.header.frame_id=self.frame; m.header.stamp=self.get_clock().now().to_msg()
        m.ns=namespace; m.id=index; m.type=kind; m.action=Marker.ADD; m.pose.orientation.w=1.
        m.scale.x=m.scale.y=m.scale.z=1.
        m.color.r,m.color.g,m.color.b,m.color.a=color
        return m

    @staticmethod
    def point(x,y,z):
        return Point(x=float(x),y=float(y),z=float(z))

    def render(self):
        if not any(p.get_subscription_count() for p in (self.markers,self.track,self.readout,*self.terrain_publishers.values())): return
        result=MarkerArray(); clear=Marker(); clear.action=Marker.DELETEALL; result.markers.append(clear)
        now=self.get_clock().now().nanoseconds*1e-9
        wall=time.monotonic()
        if self.grid is not None and wall-self.product_wall>=.25 and (self.grid is not self.last_products or wall-self.product_wall>=1.):
            products,self.product_info=terrain_products(self.grid)
            for name,product in products.items():
                if self.terrain_publishers[name].get_subscription_count(): self.terrain_publishers[name].publish(product)
            self.product_wall=wall;self.last_products=self.grid
        if self.odom is None or not -.1<=now-self.last_odom_stamp<=.6:
            self.markers.publish(result); return
        p=self.odom.pose.pose.position; yaw=yaw_of(self.odom.pose.pose.orientation)
        trace=Path(); trace.header=self.odom.header; trace.poses=list(self.trail); self.track.publish(trace)
        projection=project_path(p.x,p.y,self.path)
        error=projection[0] if projection else math.nan
        if projection:
            connector=self.marker(0,"tracking_error",Marker.LINE_LIST,(.90,.87,.70,.9))
            connector.scale.x=.022
            connector.points=[self.point(p.x,p.y,p.z+.04),self.point(*projection[1])]
            result.markers.append(connector)
        if self.command is not None and 0<=now-self.command_received<=.5:
            preview=self.marker(0,"command_projection",Marker.LINE_STRIP,(.82,.48,.74,.95))
            preview.scale.x=.035
            preview.points=[self.point(x,y,p.z+.08) for x,y in command_projection(
                p.x,p.y,yaw,self.command.linear.x,self.command.angular.z/self.steer_sign,self.wheel_base)]
            result.markers.append(preview)
        fresh_status=0<=now-self.status_stamp<=.6
        controller=self.status.get("tracking_controller","unknown") if fresh_status else "planner unavailable/stale"
        def number(name):
            try: return f"{float(self.status[name]):.2f}" if fresh_status else "?"
            except (KeyError,ValueError): return "?"
        speed=number("measured_speed_mps"); steering=number("command_steering_rad")
        age=now-stamp_s(self.grid.info.header.stamp) if self.grid else math.inf
        error_text=f"{error:.2f} m" if math.isfinite(error) else "N/A"
        state=self.status_message if fresh_status else "no current controller status"
        if self.grid and time.monotonic()-self.probe_wall>=.5:
            self.probe_wall=time.monotonic()
            try: self.probe=inspect_grid(self.grid,*(self.selection or (p.x,p.y)),yaw,
                                        float(self.status.get('state_limit',.05)))
            except (ValueError,IndexError,TypeError) as exc:
                self.probe=None; self.get_logger().warn(f"terrain display probe rejected malformed input: {exc}",throttle_duration_sec=2.)
        if self.selection is not None and self.probe and -.1<=now-self.probe["stamp_s"]<=1.:
            self.render_probe(result)
        q=self.probe if self.probe and -.1<=now-self.probe["stamp_s"]<=1. else {}
        measured=math.isfinite(q.get("display_height",math.nan))
        def value(number,unit):
            return f"{number:.3f} {unit}" if math.isfinite(number) else "unknown"
        height=q.get("display_height",math.nan)
        sigma=q.get("model_sigma",math.nan) if measured else math.nan
        upper=q.get("effective_pose_probability_upper",math.nan) if q.get("probability_valid",0)>.5 else math.nan
        recovery=self.status.get("recovery_mode","inactive") if fresh_status else "stale"
        recovery={"inactive":"未激活","retreat_then_stop_replan":"倒车退让 → 停车 → 重新找路",
                  "staged_gear_plan":"分段换向：停稳后重新校验下一段",
                  "paired_retreat_forward_exit":"倒车与前进出口已联合校验","stale":"数据过期"}.get(recovery,recovery)
        values={"banner":self.presentation_context+" / "+state,"controller":controller,"state":state,
                "recovery":recovery,"recovery_attempts":(
                    self.status.get("recovery_maneuver_attempts","--")+" / "+self.status.get("recovery_maneuver_total","--")) if fresh_status else "--",
                "speed":speed+" m/s","steering":steering+" rad","track_error":error_text,
                "map_age":f"{age:.2f} s","source":q.get("source","unavailable"),
                "height":value(height,"m"),"sigma":value(sigma,"m"),
                "interval":value(3.92*sigma,"m"),"pose_upper":value(100*upper,"%"),
                "risk_reason":q.get('risk_reason','unavailable'),
                "probability_contract":value(q.get('probability_contract',math.nan),''),
                "local_step_limit":value(100*q.get('probability_max_step',math.nan),'cm'),
                "roughness_span_limit":value(100*q.get('probability_max_roughness',math.nan),'cm'),
                "ground_clearance":value(100*q.get('probability_ground_clearance',math.nan),'cm'),
                "underbody_margin":value(100*q.get('probability_underbody_margin',math.nan),'cm'),
                "floor_model":('逐格独立高斯误差（仿真假设）' if q.get('probability_floor_covariance_model')==1 else
                    '误差相关性未知（保守上界）' if q.get('probability_contract') in (6,7) else 'unknown'),
                "navigation_height":value(q.get('probability_elevation',math.nan),'m'),
                "navigation_sigma":value(q.get('navigation_sigma',math.nan),'m'),
                "pending_height":value(q.get('ground_pending_elevation',math.nan),'m'),
                "heading":f"{math.degrees(yaw):.1f} deg",
                "probe_xy":f"({q['x']:.1f}, {q['y']:.1f}) m" if q else "unavailable"}
        values.update(self.product_info)
        values['invalid_display_packets']=str(sum(self.invalid_display_packets.values()))
        if self.product_info.get("startup_assumption")=="simulation only":
            values["banner"]=self.presentation_context+" / SIM STANCE ASSUMPTION / "+state
        values["accumulated_points"]=str(self.map_points) if now-self.map_stamp<3 else "unavailable/stale"
        report=DiagnosticArray(); report.header.stamp=self.get_clock().now().to_msg()
        status=DiagnosticStatus(); status.name="ptr_nav/presentation"
        status.values=[KeyValue(key=k,value=v) for k,v in values.items()]
        report.status.append(status); self.readout.publish(report)
        self.markers.publish(result)

    def render_probe(self,result):
        q=self.probe; mean=q.get("display_height",math.nan); sigma=q.get("model_sigma",math.nan)
        measured=math.isfinite(mean)
        if not math.isfinite(mean): mean=self.odom.pose.pose.position.z
        if math.isfinite(sigma) and measured:
            bar=self.marker(0,"height_model",Marker.LINE_LIST,(.95,.76,.35,1.))
            bar.scale.x=.035; x,y=q["x"],q["y"]
            for a,b in [((x,y,mean-1.96*sigma),(x,y,mean+1.96*sigma)),
                        ((x-.12,y,mean-1.96*sigma),(x+.12,y,mean-1.96*sigma)),
                        ((x-.12,y,mean+1.96*sigma),(x+.12,y,mean+1.96*sigma))]:
                bar.points.extend([self.point(*a),self.point(*b)])
            result.markers.append(bar)
        label=self.marker(0,"height_model_label",Marker.TEXT_VIEW_FACING,(.95,.84,.62,1.))
        label.scale.z=.23; label.pose.position=self.point(q["x"]-3.,q["y"]+2.5,mean+.55)
        upper=q.get("effective_pose_probability_upper",math.nan)
        risk=f"{100*upper:.2f}%" if q.get("probability_valid",0)>.5 and math.isfinite(upper) else "unknown"
        height=f"{mean:.3f} m" if measured else "unqualified"
        sigma_text=f"{sigma:.3f} m" if measured and math.isfinite(sigma) else "unavailable"
        label.text=(f"{q['source']} | height {height}\nsigma {sigma_text} | bound {risk}\n95% model interval (not calibrated)"
                    if measured else f"{q['source']} ground\npose probability: {risk}")
        label.text+='\n'+q.get('risk_cause_code','unavailable')
        result.markers.append(label)


def main(args=None):
    rclpy.init(args=args); node=NavigationDisplay()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally:
        node.destroy_node()
        if rclpy.ok(): rclpy.shutdown()


if __name__=="__main__": main()
