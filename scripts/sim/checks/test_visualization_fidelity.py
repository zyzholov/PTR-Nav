import numpy as np
import pytest
from se2_grid_msgs.msg import SE2Grid
from fsgp_bgk_se2.se2_grid_tools import matrix_to_multiarray_like,world_to_grid_buffer
from visualization_probe import cloud_evidence,require_cloud_fidelity,smooth_quad_exists


def sample():
    m=SE2Grid();m.info.pos_resolution=.2;m.info.length_x=1.2;m.info.length_y=.8
    m.info.yaw_resolution=np.pi;m.info.length_yaw=2*np.pi
    m.start_index_row=2;m.start_index_col=1
    h=np.full((4,6),np.nan,np.float32);h[1:3,2:4]=.4
    for n,a in (('elevation',h),('valid',np.isfinite(h).astype(np.float32)),
                ('probability_elevation',h+.01)):
        m.layers.append(n);m.has_so2.append(False)
        m.data.append(matrix_to_multiarray_like(world_to_grid_buffer(a,m),None))
    for n in ('failure_probability_upper','probability_valid'):
        m.layers.append(n);m.has_so2.append(True)
        for k in (0,1):
            a=np.ones_like(h) if n=='probability_valid' else np.full_like(h,.01)
            a[1+k,2]=np.nan
            m.data.append(matrix_to_multiarray_like(world_to_grid_buffer(a,m),None))
    return m


def test_sparse_products_must_contain_every_same_stamp_source_cell_at_its_own_height():
    m=sample();expected=cloud_evidence(m,'elevation')[0]
    assert len(expected)==4 and smooth_quad_exists(expected,.2)
    points=[(*xy,z)for xy,z in expected.items()]
    require_cloud_fidelity(points,[expected],'elevation')
    with pytest.raises(RuntimeError):require_cloud_fidelity(points[:-1],[expected],'elevation')
    with pytest.raises(RuntimeError):require_cloud_fidelity(points+[points[0]],[expected],'elevation')
    broken=points.copy();broken[0]=(*broken[0][:2],0.)
    with pytest.raises(RuntimeError):require_cloud_fidelity(broken,[expected],'elevation')
    risk=cloud_evidence(m,'risk');assert len(risk)==2 and all(len(v)==3 for v in risk)
    for e in risk:require_cloud_fidelity([(*xy,z)for xy,z in e.items()],risk,'risk')
    union={**risk[0],**risk[1]}
    with pytest.raises(RuntimeError):require_cloud_fidelity([(*xy,z)for xy,z in union.items()],risk,'risk')


def test_absent_drape_height_cannot_be_fabricated_as_a_zero_plane():
    m=sample();expected=cloud_evidence(m,'risk')
    # Many valid risk scalars have no finite height; none belongs in a drape.
    assert all(len(v)==3 for v in expected)
    require_cloud_fidelity([(*xy,z)for xy,z in expected[0].items()],expected,'risk')
    with pytest.raises(RuntimeError):require_cloud_fidelity([(-.5,-.3,.035)],expected,'risk')
    assert not smooth_quad_exists({(0.,0.):.1,(.2,0.):.1},.2)
