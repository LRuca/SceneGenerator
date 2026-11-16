"""Analytic camera-Z rendering, using SceneBlocker's primitive intersections."""
import os,sys
import numpy as np
from .storage import ROOT,BASE,inside
REF=inside(os.environ.get("SCENEGEN_REFERENCE", BASE/"dependencies/SceneBlocker"))
sys.path.insert(0,str(REF/"src"))
from proxy_observations import ray_interval
from primitive_composition import unit_mesh,transform_mesh

def rays(K,c2w,h,w):
    v,u=np.mgrid[:h,:w]
    local=np.column_stack([u.ravel(),v.ravel(),np.ones(h*w)])@np.linalg.inv(K).T
    # Directions have camera Z=1; ray parameter is camera-Z, not Euclidean distance.
    return c2w[:3,3],local@c2w[:3,:3].T

def primitive_depth(kind,matrix,K,c2w,h,w):
    origin,direction=rays(K,c2w,h,w);inv=np.linalg.inv(np.asarray(matrix))
    near,_=ray_interval(inv[:3,:3]@origin+inv[:3,3],direction@inv[:3,:3].T,"box" if kind=="plane" else kind)
    return near.reshape(h,w)

def box_matrix(center,size,rotation=None):
    M=np.eye(4);M[:3,:3]=(np.eye(3) if rotation is None else rotation)@np.diag(size);M[:3,3]=center
    return M

def render_parts(parts,K,c2w,h,w):
    depth=np.full((h,w),np.inf);labels=np.full((h,w),-1,dtype="int32")
    for index,p in enumerate(parts):
        d=primitive_depth(p["kind"],p["canonical_to_world"],K,c2w,h,w)
        take=d<depth;depth[take]=d[take];labels[take]=index
    return depth,labels

def scene_parts(scene,t):
    parts=[]
    for e in scene["entities"]:
        if not e.get("exists",e.get("renderable",[True]*(t+1)))[t]:continue
        transform=np.asarray(e["entity_to_world"][0 if e["motion"]=="static" else t])
        for p in e["parts"]:
            parts.append({"kind":p["kind"],"entity_id":e["id"],
                "canonical_to_world":(transform@np.asarray(p["canonical_to_entity"])).tolist(),
                "state_source":e.get("state_source",["legacy_unspecified"]*(t+1))[t]})
    return parts

def mesh_preview(parts,K,c2w,w,h):
    """Neutral white analytic surface preview; camera-relative lighting, unchanged depth."""
    depth,labels=render_parts(parts,K,c2w,h,w)
    origin,dirs=rays(K,c2w,h,w)
    frame=np.full((h*w,3),[43.,48.,57.])
    # Soft two-sided lighting keeps thin, inferred surfaces inspectable from both sides.
    light=c2w[:3,:3]@np.array([-.45,-.65,-.8])
    light/=np.linalg.norm(light)
    for index,part in enumerate(parts):
        hit=np.flatnonzero(labels.ravel()==index)
        if not len(hit):continue
        mat=np.asarray(part["canonical_to_world"]);inv=np.linalg.inv(mat)
        world=origin+dirs[hit]*depth.ravel()[hit,None]
        q=world@inv[:3,:3].T+inv[:3,3]
        kind=part["kind"]
        if kind in ("box","plane"):
            axis=np.argmin(np.abs(np.abs(q)-.5),axis=1)
            normal=np.zeros_like(q);normal[np.arange(len(q)),axis]=np.sign(q[np.arange(len(q)),axis])
        elif kind=="sphere":normal=q
        else:
            radial=np.linalg.norm(q[:,[0,2]],axis=1).clip(1e-12)
            normal=np.column_stack([q[:,0]/radial,np.zeros(len(q)),q[:,2]/radial])
            if kind=="cone":normal[:,1]=.5
            cap=(np.abs(q[:,1]+.5)<.0001)
            if kind=="cylinder":cap|=(np.abs(q[:,1]-.5)<.0001)
            normal[cap]=0;normal[cap,1]=np.sign(q[cap,1])
        normal=normal@inv[:3,:3];normal/=np.linalg.norm(normal,axis=1,keepdims=True).clip(1e-12)
        shade=.46+.52*np.abs(normal@light)
        frame[hit]=shade[:,None]*245
    return frame.reshape(h,w,3).astype("uint8")

def agreement(rendered,target,mask=None):
    valid=np.isfinite(target)&(target>0)
    if mask is not None:valid &=mask
    hit=valid&np.isfinite(rendered)
    n=int(valid.sum());n_hit=int(hit.sum())
    relative=np.abs(rendered[hit]-target[hit])/target[hit]
    # Missing predictions are penalized, never silently counted as zero error.
    return {"target_pixels":n,"covered_pixels":n_hit,"coverage":n_hit/max(n,1),
        "median_relative_depth_error_on_covered":float(np.median(relative)) if n_hit else None,
        "capped_relative_error_with_misses":float((np.minimum(relative,1).sum()+n-n_hit)/max(n,1))}
