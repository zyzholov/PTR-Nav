import numpy as np
import pytest
from capture_ground_fault import fault_cells


def test_fault_audit_keeps_only_nearby_measured_downward_innovations_and_never_writes_inputs():
 x,y=np.meshgrid(np.arange(5,dtype=float),np.arange(4,dtype=float))
 h=np.zeros(x.shape);v=np.full(x.shape,.0001);p=np.ones(x.shape);z=np.full(x.shape,-.5);seen=np.ones(x.shape)
 h0=h.copy();z0=z.copy()
 cells=fault_cells(h,v,p,z,seen,x,y,(1.,1.),max_distance=.5)
 np.testing.assert_array_equal(cells,[[1,1]])
 np.testing.assert_array_equal(h,h0);np.testing.assert_array_equal(z,z0)
 for a,value in [(p,0),(seen,0),(v,np.nan),(z,.5)]:
  old=a[1,1];a[1,1]=value
  assert not len(fault_cells(h,v,p,z,seen,x,y,(1.,1.),max_distance=.5))
  a[1,1]=old
 assert not len(fault_cells(h,v,p,z,seen,x,y,None))
 with pytest.raises(ValueError):fault_cells(h,v,p,z,seen,x[:-1],y,(1.,1.))
