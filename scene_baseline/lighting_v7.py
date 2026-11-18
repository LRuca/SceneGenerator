"""Fixed world light and soft ray-cast shadows. No depth/pose/geometry edits."""
import numpy as np
from .geometry import render_parts,rays,ray_interval
def render(parts,K,c2w,h,w,up,colors=None,light_right=None):
    depth,labels=render_parts(parts,K,c2w,h,w);origin,dirs=rays(K,c2w,h,w)
    hit=np.flatnonzero(np.isfinite(depth.ravel()));world=origin+dirs[hit]*depth.ravel()[hit,None]
    normals=np.zeros_like(world);owner=labels.ravel()[hit]
    for j,p in enumerate(parts):
        ids=np.flatnonzero(owner==j)
        if not len(ids):continue
        if p["kind"]=="cage":
            from .mesh_raster_v4 import intersect
            _,tri,nn=intersect(origin,dirs[hit[ids]],p["vertices"],p["faces"]);normals[ids]=nn[tri.clip(0)];continue
        M=np.asarray(p["canonical_to_world"]);inv=np.linalg.inv(M);q=world[ids]@inv[:3,:3].T+inv[:3,3]
        if p["kind"] in ["box","plane"]:
            axis=np.argmin(abs(abs(q)-.5),axis=1);nn=np.zeros_like(q);nn[np.arange(len(q)),axis]=np.sign(q[np.arange(len(q)),axis])
        else:nn=q
        nn=nn@inv[:3,:3];normals[ids]=nn/np.linalg.norm(nn,axis=1,keepdims=True).clip(1e-9)
    up=np.asarray(up,float);up/=np.linalg.norm(up)
    right=np.asarray(c2w[:3,0] if light_right is None else light_right).copy();right=right-up*(right@up);right/=np.linalg.norm(right)
    front=np.cross(right,up);light=up*.85-right*.5-front*.35;light/=np.linalg.norm(light)
    diffuse=np.abs(normals@light);shadow=np.zeros(len(hit))
    eps=max(float(np.median(depth[np.isfinite(depth)]))*.001,1e-5)
    for delta in [-.055,0,.055]:
        L=light+delta*right;L/=np.linalg.norm(L);blocked=np.zeros(len(hit),bool)
        for j,p in enumerate(parts):
            if p["kind"]=="cage":continue
            inv=np.linalg.inv(np.asarray(p["canonical_to_world"]));o=(world+L*eps)@inv[:3,:3].T+inv[:3,3];direction=np.broadcast_to(L@inv[:3,:3].T,o.shape)
            near,far=ray_interval(o,direction,"box" if p["kind"]=="plane" else p["kind"])
            blocked|=(owner!=j)&np.isfinite(near)&(far>eps)&(near>eps)
        shadow+=blocked/3
    shade=(.23+.74*diffuse)*(1-.62*shadow)
    frame=np.full((h*w,3),[43.,48.,57.])
    albedo=np.array([(colors or {}).get(parts[j]["entity_id"],[245,245,245]) for j in owner])
    frame[hit]=albedo*shade[:,None]
    return frame.reshape(h,w,3).clip(0,255).astype("uint8")
