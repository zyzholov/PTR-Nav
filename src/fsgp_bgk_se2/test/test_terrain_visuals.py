import math
import numpy as np
from se2_grid_msgs.msg import SE2Grid
from sensor_msgs_py.point_cloud2 import read_points
from fsgp_bgk_se2.se2_grid_tools import matrix_to_multiarray_like,world_to_grid_buffer
from fsgp_bgk_se2.terrain_visuals import terrain_products,elevation_surface


def test_measured_predicted_and_unqualified_sources_stay_separate():
    msg=SE2Grid();msg.info.header.frame_id="map"
    msg.info.pos_resolution=.2;msg.info.length_x=.8;msg.info.length_y=.6
    msg.info.length_yaw=2*math.pi;msg.info.yaw_resolution=math.pi/2
    msg.start_index_row=1;msg.start_index_col=2
    h=np.arange(12,dtype=np.float32).reshape(3,4)*.1
    observed=np.ones_like(h);observed[0,0]=observed[1,1]=0
    h[observed==0]=np.nan;accepted=np.ones_like(h);accepted[1,1]=0
    for name,value in (("elevation",h),("terrain_observed",observed),("gp_mean",np.ones_like(h)),
                       ("gp_latent_var",np.full_like(h,.0004)),("gp_accepted",accepted),
                       ("gp_supported",np.ones_like(h)),("probability_gp_used",np.zeros_like(h))):
        msg.layers.append(name);msg.has_so2.append(False)
        msg.data.append(matrix_to_multiarray_like(world_to_grid_buffer(value,msg),None))
    products,info=terrain_products(msg)
    assert products["measured_height"].width==10
    assert products["predicted_height"].width==1
    assert products["prediction_support"].width==2
    assert products["navigation_prediction"].width==0
    assert products["sources"].width==11
    sources=read_points(products["sources"],field_names=("rgb",))
    rgb=np.asarray(sources["rgb"],np.float32).view(np.uint32)
    assert np.count_nonzero(rgb==(129<<16|147<<8|251))==1
    assert np.count_nonzero(rgb==(65<<16|190<<8|165))==10
    assert info["candidate_cells"]=="1" and info["unknown_cells"]=="1"
    prediction=list(read_points(products["predicted_height"],field_names=("x","y","z")))[0]
    np.testing.assert_allclose(list(prediction),[-.3,-.2,1.025],atol=1e-6)
    surface=elevation_surface(msg,products)
    assert len(surface.points)%6==0
    for point in surface.points:
        ix=round((point.x+.3)/.2);iy=round((point.y+.2)/.2)
        assert observed[iy,ix]>.5
        assert abs(point.z-h[iy,ix]-.025)<1e-6


def test_elevation_mesh_does_not_bridge_unknown_cells_or_vertical_steps():
    msg=SE2Grid();msg.info.pos_resolution=.2;msg.info.length_x=msg.info.length_y=.8
    msg.info.yaw_resolution=math.pi/2;msg.info.length_yaw=2*math.pi
    h=np.zeros((4,4),np.float32);h[1,1]=np.nan;h[:,3]=1.
    msg.layers=["elevation"];msg.has_so2=[False]
    msg.data=[matrix_to_multiarray_like(world_to_grid_buffer(h,msg),None)]
    products,_=terrain_products(msg);mesh=elevation_surface(msg,products)
    assert len(mesh.points)==12
    assert all(abs(p.z-.025)<1e-6 for p in mesh.points)
