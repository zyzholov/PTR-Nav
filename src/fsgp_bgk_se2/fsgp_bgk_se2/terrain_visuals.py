"""Read-only measured/predicted height products; no interpolation for display."""
import numpy as np
from sensor_msgs.msg import PointCloud2,PointField
from sensor_msgs_py.point_cloud2 import read_points
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker
from .se2_grid_tools import get_layer_2d,grid_buffer_to_world,layer_refs,matrix_shape


def terrain_products(msg):
    shape=matrix_shape(msg.data[0]); refs=layer_refs(msg)
    def layer(name,default=np.nan):
        value,_=get_layer_2d(msg,name,default,shape)
        return grid_buffer_to_world(value,msg)
    height=layer("gp_elevation_observed") if "gp_elevation_observed" in refs else layer("elevation")
    observed=(layer("terrain_observed") if "terrain_observed" in refs else layer("ground_observed",1))>.5
    observed &= np.isfinite(height)
    mean=layer("gp_mean"); variance=layer("gp_latent_var")
    candidate=(layer("gp_candidate",0)>.5 if "gp_candidate" in refs else layer("gp_accepted",0)>.5)
    candidate &= ~observed&np.isfinite(mean)&np.isfinite(variance)&(variance>=0)
    supported=(layer("gp_supported",0)>.5)&~observed&np.isfinite(mean)&np.isfinite(variance)&(variance>=0)
    navigation=layer("probability_gp_used",0)>.5
    startup=layer("startup_prior_used",0)>.5
    startup_height=layer("probability_elevation")
    values=height[observed] if observed.any() else mean[candidate]
    low=float(np.quantile(values,.02)) if len(values) else 0.
    high=float(np.quantile(values,.98)) if len(values) else 1.
    high=max(high,low+.2)
    resolution=float(msg.info.pos_resolution)
    world_shape=height.shape
    x,y=np.meshgrid(msg.info.pose.position.x-msg.info.length_x/2+(np.arange(world_shape[1])+.5)*resolution,
                    msg.info.pose.position.y-msg.info.length_y/2+(np.arange(world_shape[0])+.5)*resolution)
    def cloud(values,mask,colors):
        count=np.count_nonzero(mask)
        data=np.empty((count,4),np.float32)
        data[:,0]=x[mask];data[:,1]=y[mask];data[:,2]=values[mask]+.025
        rgb=np.asarray(colors,dtype=np.uint32)
        if rgb.ndim==1: rgb=np.broadcast_to(rgb,(count,3))
        data[:,3]=(rgb[:,0]<<16 | rgb[:,1]<<8 | rgb[:,2]).view(np.float32)
        out=PointCloud2();out.header=msg.info.header;out.height=1;out.width=count
        out.fields=[PointField(name=name,offset=4*i,datatype=PointField.FLOAT32,count=1) for i,name in enumerate(("x","y","z","rgb"))]
        out.point_step=16;out.row_step=count*16;out.is_dense=True;out.data=data.tobytes()
        return out
    def height_color(z):
        t=np.clip((z-low)/(high-low),0,1)
        stops=np.array([[72,103,184],[54,164,176],[168,220,166]],np.float32)
        index=(t>=.5).astype(int);u=(t-index*.5)*2
        return (stops[index]*(1-u[:,None])+stops[index+1]*u[:,None]).astype(np.uint32)
    products={"measured_height":cloud(height,observed,height_color(height[observed])),
              "predicted_height":cloud(mean,candidate,height_color(mean[candidate])),
              "prediction_support":cloud(mean,supported,np.array([105,172,245])),
              "startup_height":cloud(startup_height,startup&np.isfinite(startup_height),np.array([235,183,81])),
              "navigation_prediction":cloud(startup_height,navigation&np.isfinite(startup_height),np.array([91,229,190]))}
    source_mask=observed|candidate|(navigation&np.isfinite(startup_height))
    source_colors=np.where(candidate[source_mask,None],np.array([129,147,251]),np.array([65,190,165]))
    source_colors[navigation[source_mask]]=np.array([91,229,190])
    products["sources"]=cloud(np.where(observed,height,np.where(navigation,startup_height,mean)),source_mask,source_colors)
    age=layer("gp_prediction_age")
    finite_age=age[np.isfinite(age)]
    completion_age=layer("gp_completion_age")
    finite_completion_age=completion_age[np.isfinite(completion_age)]
    conditioning_age=layer('gp_conditioning_age')
    finite_conditioning_age=conditioning_age[navigation&np.isfinite(conditioning_age)]
    info={"measured_cells":str(np.count_nonzero(observed)),"candidate_cells":str(np.count_nonzero(candidate)),
          "navigation_prediction_cells":str(np.count_nonzero(navigation)),
          "startup_cells":str(np.count_nonzero(startup)),
          "startup_assumption":"simulation only" if (layer("probability_startup_prior_enabled",0)>.5).any() else "disabled",
          "unknown_cells":str(np.count_nonzero(~observed&~candidate)),
          "height_scale":f"{low:.2f} .. {high:.2f} m",
          "prediction_age":f"{float(finite_age[0]):.2f} s" if len(finite_age) else "pending/off",
          "completion_age":f"{float(finite_completion_age[0]):.2f} s" if len(finite_completion_age) else "pending/off"}
    info['conditioning_age']=f'{float(finite_conditioning_age.max()):.2f} s' if len(finite_conditioning_age) else 'pending/off'
    return products,info


def elevation_surface(msg,products):
    """Offline reference for validating the C++ measured-elevation mesh.

    GP candidates remain a separate dotted layer; this mesh cannot imply that
    predicted or assumed heights were measured. Resolution is not upsampled.
    """
    cloud=products["measured_height"]
    points=read_points(cloud,field_names=("x","y","z","rgb"))
    h=grid_buffer_to_world(get_layer_2d(msg,"elevation",np.nan,matrix_shape(msg.data[0]))[0],msg)
    rows,cols=h.shape;res=float(msg.info.pos_resolution)
    z=np.full_like(h,np.nan);colors=np.zeros((*h.shape,3),np.float32)
    ix=np.rint((points["x"]-(msg.info.pose.position.x-msg.info.length_x/2+.5*res))/res).astype(int)
    iy=np.rint((points["y"]-(msg.info.pose.position.y-msg.info.length_y/2+.5*res))/res).astype(int)
    z[iy,ix]=points["z"]
    rgb=np.asarray(points["rgb"],np.float32).view(np.uint32)
    colors[iy,ix]=np.column_stack(((rgb>>16)&255,(rgb>>8)&255,rgb&255))/255.
    a=z[:-1,:-1];b=z[:-1,1:];c=z[1:,:-1];d=z[1:,1:]
    corners=np.stack((a,b,c,d))
    valid=np.all(np.isfinite(corners),axis=0)&((np.max(corners,axis=0)-np.min(corners,axis=0))<=.65)
    r,col=np.where(valid)
    # Counterclockwise winding. Each vertex keeps its own height/color.
    rr=np.column_stack((r,r,r+1,r,r+1,r+1)).ravel()
    cc=np.column_stack((col,col+1,col,col+1,col+1,col)).ravel()
    marker=Marker();marker.header=msg.info.header;marker.ns="measured_elevation";marker.id=0
    marker.type=Marker.TRIANGLE_LIST;marker.action=Marker.ADD if rr.size else Marker.DELETE
    marker.pose.orientation.w=1.;marker.scale.x=marker.scale.y=marker.scale.z=1.
    marker.color.a=1.
    marker.points=[Point(x=float(msg.info.pose.position.x-msg.info.length_x/2+(j+.5)*res),
        y=float(msg.info.pose.position.y-msg.info.length_y/2+(i+.5)*res),z=float(z[i,j])) for i,j in zip(rr,cc)]
    marker.colors=[ColorRGBA(r=float(v[0]),g=float(v[1]),b=float(v[2]),a=1.) for v in colors[rr,cc]]
    return marker
