"""Prototype isolated rigid-box track fitter. Full scene occlusion fitting is a later stage."""
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from .geometry import box_matrix,primitive_depth,agreement

def fit_rigid_box(depth,masks,K,c2w,centers,size,rotations=None,optimize_shape=False,optimize_rotation=False,max_nfev=60):
    n,h,w=depth.shape
    centers=np.asarray(centers,float);size=np.asarray(size,float)
    if masks.shape!=depth.shape or any(not m.any() for m in masks):raise ValueError("Every prototype fit frame needs a visible mask")
    rotations=np.tile(np.eye(3),(n,1,1)) if rotations is None else np.asarray(rotations)
    r0=Rotation.from_matrix(rotations).as_rotvec()
    params=[centers.ravel()]
    if optimize_shape:params.append(np.log(size))
    if optimize_rotation:params.append(r0.ravel())
    x0=np.concatenate(params);samples=[];boxes=[]
    for mask in masks:
        y,x=np.nonzero(mask);stride=max(1,int(np.ceil(len(x)/300)))
        samples.append((y[::stride],x[::stride]));boxes.append([x.min(),y.min(),x.max(),y.max()])
    boxes=np.asarray(boxes)
    corners=np.asarray([[x,y,z] for x in [-.5,.5] for y in [-.5,.5] for z in [-.5,.5]])
    def unpack(x):
        offset=n*3;c=x[:offset].reshape(n,3)
        s=np.exp(x[offset:offset+3]) if optimize_shape else size
        offset+=3 if optimize_shape else 0
        R=Rotation.from_rotvec(x[offset:].reshape(n,3)).as_matrix() if optimize_rotation else rotations
        return c,s,R
    def residual(x):
        c,s,rot=unpack(x);res=[]
        for t,(y,u) in enumerate(samples):
            M=box_matrix(c[t],s,rot[t]);d=primitive_depth("box",M,K[t],c2w[t],h,w)
            pred=d[y,u];target=depth[t,y,u]
            z=np.where(np.isfinite(pred),np.minimum(np.abs(pred-target)/target,3),3)
            res.extend(z/np.sqrt(len(z)))
            world=corners@M[:3,:3].T+M[:3,3];inv=np.linalg.inv(c2w[t])
            cam=world@inv[:3,:3].T+inv[:3,3]
            uv=cam@K[t].T;uv=uv[:,:2]/np.maximum(cam[:,2:3],1e-5)
            box=np.r_[uv.min(0),uv.max(0)]
            res.extend((box-boxes[t])/np.array([w,h,w,h])*2)
            res.extend(np.maximum(.001-cam[:,2],0))
        if n>=3:res.extend((.02*np.diff(c,n=2,axis=0)/np.median(depth)).ravel())
        if optimize_shape:res.extend(.05*np.log(s/size))
        return np.asarray(res)
    initial=float(np.mean(residual(x0)**2))
    result=least_squares(residual,x0,loss="soft_l1",f_scale=.1,max_nfev=max_nfev)
    c,s,R=unpack(result.x)
    matrices=[box_matrix(c[t],s,R[t]) for t in range(n)]
    metrics=[agreement(primitive_depth("box",matrices[t],K[t],c2w[t],h,w),depth[t],masks[t]) for t in range(n)]
    return {"centers":c.tolist(),"size":s.tolist(),"rotations":R.tolist(),"matrices":[m.tolist() for m in matrices],
        "initial_mse":initial,"final_mse":float(np.mean(residual(result.x)**2)),"nfev":result.nfev,
        "optimizer_success":bool(result.success),"message":result.message,"per_frame":metrics,
        "limitations":["isolated box component test; no full-scene occlusion optimization","silhouette term currently projected bbox, not full mask","cameras fixed; depth agreement is not physical accuracy"]}
