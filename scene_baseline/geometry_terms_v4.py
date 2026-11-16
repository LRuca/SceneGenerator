"""Opt-in proxy correspondence terms; observations remain immutable."""
import numpy as np
import torch
from .storage import read_json,verify
from .fit_scene_visibility_v3 import _surface_distance

def signed_parts(points, matrices, kinds):
    """Entity-local points (...,3), signed approximate distance in entity units."""
    ds=[]
    for mat,kind in zip(matrices,kinds):
        inv=torch.linalg.inv(mat[:3,:3])
        q=(points-mat[:3,3])@inv.T
        lengths=torch.linalg.norm(mat[:3,:3],dim=0)
        if kind in ("box","plane"):
            delta=(q.abs()-.5)*lengths
            dist=torch.linalg.norm(delta.clamp_min(0),dim=-1)+torch.minimum(delta.max(-1).values,torch.zeros_like(delta[...,0]))
        else:
            dist=_surface_distance(q,kind)*lengths.min()
        ds.append(dist)
    return torch.stack(ds).min(0).values

def prepare(scene,a,entity,max_own=96,max_other=96):
    """Deterministic foreground samples and scene-observed free rays, not bbox fill."""
    md=read_json(verify(scene["masks_manifest"]))
    with np.load(verify(md["data"])) as z: masks=z["masks"];ids=z["ids"].tolist()
    index=ids.index(entity["id"]);own=masks[index];known=masks.sum(0)==1
    rows=[]
    for t in range(len(own)):
        chosen=[]
        for region,cap,owned in [(own[t]&known[t],max_own,True),(known[t]&~own[t],max_other,False)]:
            v,u=np.nonzero(region)
            ii=np.linspace(0,len(u)-1,min(len(u),cap)).round().astype(int) if len(u) else np.array([],int)
            for j in ii:chosen.append((t,int(v[j]),int(u[j]),owned))
        rows+=chosen
    if not rows:return None
    tt,vv,uu,owned=map(np.asarray,zip(*rows));tt=tt.astype(int);vv=vv.astype(int);uu=uu.astype(int)
    pixels=np.column_stack([uu,vv,np.ones(len(tt))])
    ray=np.einsum("nij,nj->ni",np.linalg.inv(a["K"][tt]),pixels)
    direction=np.einsum("nij,nj->ni",a["c2w"][tt,:3,:3],ray)
    origin=a["c2w"][tt,:3,3];depth=a["depth"][tt,vv,uu]
    confidence=a["confidence"][tt,vv,uu]
    good=np.isfinite(depth)&(depth>0)&np.isfinite(confidence)&(confidence>0)
    w=np.clip(confidence/max(float(np.median(confidence[good])),1e-8),.1,4)*good
    return dict(t=tt,owned=owned.astype(bool),points=origin+direction*depth[:,None],origin=origin,
                free=np.stack([origin+direction*(depth*f)[:,None] for f in [.35,.65,.90]],axis=1),weight=w)

def losses(samples,R,C,matrices,kinds,scale,extent):
    zero=C.sum()*0
    if samples is None:return zero,zero
    t=samples["t"];rot=R[t];ctr=C[t]
    points=samples["points"]/scale
    q=torch.einsum("nij,nj->ni",rot.transpose(1,2),points-ctr)
    origins=samples["origin"]/scale
    origin_local=torch.einsum("nij,nj->ni",rot.transpose(1,2),origins-ctr)
    # Bounded camera-visible faces give gradients even when a ray misses the proxy.
    distances=[]
    for mat,kind in zip(matrices,kinds):
        inverse=torch.linalg.inv(mat[:3,:3]);p=(q-mat[:3,3])@inverse.T
        o=(origin_local-mat[:3,3])@inverse.T
        lengths=torch.linalg.norm(mat[:3,:3],dim=0)
        if kind in ("box","plane"):
            faces=[]
            for ax in range(3):
                side=torch.where(o[:,ax]>=0,1.,-1.)
                delta=torch.relu(p.abs()-.5)
                delta=delta.clone();delta[:,ax]=p[:,ax]-side*.5
                dist=torch.sqrt((delta*lengths).square().sum(-1)+1e-12)
                faces.append(torch.where(o[:,ax].abs()>.5,dist,torch.full_like(dist,1e3)))
            distances.append(torch.stack(faces).min(0).values)
        else:
            distances.append(signed_parts(q,[mat],[kind]).abs())
    dist=torch.stack(distances).min(0).values/extent
    ow=samples["weight"]*samples["owned"]
    visible=(torch.sqrt(dist.square()+.02**2)-.02)*ow
    visible=visible.sum()/ow.sum().clamp_min(1)
    free_world=samples["free"]/scale
    free_local=torch.einsum("nij,nkj->nki",rot.transpose(1,2),free_world-ctr[:,None,:])
    penetration=torch.relu(-signed_parts(free_local,matrices,kinds)/extent-.01)
    free=(penetration.square()*samples["weight"][:,None]).sum()/(samples["weight"].sum()*3).clamp_min(1)
    return visible,free

def outside_fraction(points,parts,tolerance=.03):
    p=np.asarray(points);inside=np.zeros(len(p),bool)
    for part in parts:
        mat=np.asarray(part["canonical_to_entity"]);q=(p-mat[:3,3])@np.linalg.inv(mat[:3,:3]).T
        if part["kind"] in ("box","plane"):hit=(np.abs(q)<=.5+tolerance).all(-1)
        elif part["kind"]=="sphere":hit=np.linalg.norm(q,axis=-1)<=.5+tolerance
        else:hit=(np.linalg.norm(q[:,[0,2]],axis=-1)<=.5+tolerance)&(np.abs(q[:,1])<=.5+tolerance)
        inside|=hit
    return float((~inside).mean()) if len(p) else None
