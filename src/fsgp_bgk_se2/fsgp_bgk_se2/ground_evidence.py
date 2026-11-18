"""Bounded raw-return history. Predictions and assumed heights are not inputs."""
import math
import numpy as np


def temporal_std_bound(weights,source_ages,process_std_mps):
    """Triangle bound for fixed GP source weights, without independence."""
    weights=np.asarray(weights);source_ages=np.asarray(source_ages)
    if (weights.ndim!=2 or weights.shape!=source_ages.shape or
            not np.isfinite(weights).all() or not np.isfinite(source_ages).all() or
            (source_ages<0).any() or not math.isfinite(process_std_mps) or process_std_mps<=0):
        raise ValueError('invalid temporal source weights, ages or drift model')
    return process_std_mps*np.sum(np.abs(weights)*source_ages,axis=1)


class GroundEvidenceHistory:
    def __init__(self,max_age_s,voxel_m=.05):
        if not math.isfinite(max_age_s) or not 0<max_age_s<=2 or not math.isfinite(voxel_m) or voxel_m<=0:
            raise ValueError('invalid ground-evidence history bounds')
        self.max_age_s=max_age_s;self.voxel_m=voxel_m
        self.points=np.empty((0,3),np.float32);self.stamps=np.empty(0)
        self.frame=None;self.stamp=None;self.pose=None
        self.vetoes={};self.clear_stamp=None

    def update(self,points,stamp,frame,pose_xy,source_age_s=0.):
        if (not math.isfinite(stamp) or not frame or not np.isfinite(pose_xy).all() or
                not math.isfinite(source_age_s) or source_age_s<0):
            raise ValueError('invalid evidence frame/time/pose')
        if (frame!=self.frame or self.stamp is not None and stamp<self.stamp or
                self.pose is not None and np.linalg.norm(np.asarray(pose_xy)-self.pose)>1.):
            self.points=np.empty((0,3),np.float32);self.stamps=np.empty(0)
            self.vetoes={};self.clear_stamp=None
        incoming=np.asarray(points,np.float32)
        if incoming.ndim!=2 or incoming.shape[1]!=3:raise ValueError('expected raw XYZ returns')
        incoming=incoming[np.isfinite(incoming).all(axis=1)]
        points=np.vstack((self.points,incoming))
        stamps=np.r_[self.stamps,np.full(len(incoming),stamp-source_age_s)]
        keep=(stamp-stamps>=0)&(stamp-stamps<=self.max_age_s)
        points=points[keep];stamps=stamps[keep]
        if len(points):
            cell=np.floor(points[:,:2]/self.voxel_m).astype(np.int64)
            order=np.lexsort((-stamps,cell[:,0],cell[:,1]))
            first=np.r_[True,np.any(np.diff(cell[order],axis=0),axis=1)]
            ids=order[first];points=points[ids];stamps=stamps[ids]
        self.points=points;self.stamps=stamps
        self.frame=frame;self.stamp=stamp;self.pose=np.asarray(pose_xy).copy()
        return self.points.copy(),stamp-self.stamps

    def retain(self,mask):
        """Contradiction/occupancy removal persists after the current frame."""
        self.points=self.points[mask];self.stamps=self.stamps[mask]

    def veto_mask(self,x,y,new_veto,current_ground_clear,source_stamp,*,strong_ground_clear=None):
        if strong_ground_clear is not None:
            strong_ground_clear=np.asarray(strong_ground_clear,bool)
            if strong_ground_clear.shape!=x.shape:raise ValueError('ground witness shape mismatch')
        if not self.vetoes and not new_veto.any():
            self.clear_stamp=source_stamp
            return np.zeros(x.shape,bool)
        resolution=float(x[0,1]-x[0,0])
        xx=np.floor(x/resolution).astype(int);yy=np.floor(y/resolution).astype(int)
        # ROS maps are rectilinear unit-step lattices. Locate only the stored
        # vetoes instead of rebuilding a Python dictionary for every map cell.
        regular=(np.all(xx==xx[0:1,:]) and np.all(yy==yy[:,0:1]) and
                 np.all(np.diff(xx[0])==1) and np.all(np.diff(yy[:,0])==1))
        if regular:
            gx0,gy0=int(xx[0,0]),int(yy[0,0]);rows,cols=x.shape
            def locate(key):
                r,c=key[1]-gy0,key[0]-gx0
                return (r,c) if 0<=r<rows and 0<=c<cols else None
        else:
            lookup={(int(a),int(b)):(i,j) for i,j,a,b in
                    zip(*np.indices(x.shape).reshape(2,-1),xx.ravel(),yy.ravel())}
            locate=lookup.get
        fresh=source_stamp!=self.clear_stamp
        for key in list(self.vetoes):
            location=locate(key)
            if location is None:del self.vetoes[key];continue
            i,j=location
            # A current real plane, bracketed in XY and independently enclosed
            # by already measured lower terrain, may retire an old mistaken
            # height contradiction. It cannot override NEW negative evidence.
            if fresh and strong_ground_clear is not None and strong_ground_clear[i,j] and \
                    current_ground_clear[i,j] and not new_veto[i,j]:
                del self.vetoes[key];continue
            if fresh:self.vetoes[key]=self.vetoes[key]+1 if current_ground_clear[i,j] else 0
            if self.vetoes[key]>=3:del self.vetoes[key]
        for i,j in zip(*np.where(new_veto)):self.vetoes[(int(xx[i,j]),int(yy[i,j]))]=0
        self.clear_stamp=source_stamp
        mask=np.zeros(x.shape,bool)
        for key in self.vetoes:mask[locate(key)]=True
        return mask
