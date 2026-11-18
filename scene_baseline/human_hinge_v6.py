"""Coarse person: ground root, shared scale, knee/waist hinges, smooth transitions, sphere head."""
import numpy as np
from .cage_v4 import faces
SCHEMA="scene_baseline.human_hinge.v6"
SECTIONS=25
CIRCUMFERENCE=12
def vertices(height,width,depth,angles,head_ratio=.08,simple_cylinder=False):
    import torch
    n=len(angles);f=lambda x:torch.as_tensor(x,dtype=angles.dtype,device=angles.device)
    # Body axis ends below the head; integrated unit tangents preserve axis arc length.
    end=.84
    s=torch.linspace(0,end,SECTIONS,device=angles.device,dtype=angles.dtype)
    def step(pivot,halfwidth):
        u=((s-pivot+halfwidth)/(2*halfwidth)).clamp(0,1)
        return u*u*(3-2*u)
    theta=angles[:,1,None]*step(.52,.10) if simple_cylinder else angles[:,0,None]*step(.27,.055)+angles[:,1,None]*step(.52,.075)
    tangent=torch.stack([torch.sin(theta),torch.cos(theta),torch.zeros_like(theta)],-1)
    ds=end/(SECTIONS-1)
    increments=.5*(tangent[:,1:]+tangent[:,:-1])*height*ds
    axis=torch.cat([torch.zeros((n,1,3),dtype=angles.dtype,device=angles.device),torch.cumsum(increments,1)],1)
    normal=torch.stack([torch.cos(theta),-torch.sin(theta),torch.zeros_like(theta)],-1)
    profile=f(np.interp(np.linspace(0,.84,SECTIONS),[0,.15,.27,.43,.52,.70,.78,.84],[.68,.75,.72,.95,1.,1.08,.95,.45]))
    if simple_cylinder:profile=torch.ones_like(s)
    phi=torch.arange(CIRCUMFERENCE,device=angles.device,dtype=angles.dtype)*2*torch.pi/CIRCUMFERENCE
    offset=normal[:,:,None,:]*(width*.5*profile[None,:,None,None]*torch.cos(phi)[None,None,:,None])
    offset=offset+f([0,0,1])[None,None,None,:]*(depth*.5*profile[None,:,None,None]*torch.sin(phi)[None,None,:,None])
    body=(axis[:,:,None]+offset).reshape(n,-1,3)
    radius=height*head_ratio
    head=axis[:,-1]+tangent[:,-1]*radius
    # Same sphere samples used by fitting and analytic head rendering.
    lat=torch.linspace(-torch.pi/2,torch.pi/2,9,device=angles.device,dtype=angles.dtype)
    ring=torch.stack([torch.cos(lat)[:,None]*torch.cos(phi)[None],torch.sin(lat)[:,None].expand(-1,CIRCUMFERENCE),torch.cos(lat)[:,None]*torch.sin(phi)[None]],-1).reshape(-1,3)
    head_samples=head[:,None]+radius*ring[None]
    return body,head,head_samples,axis
def validate(state,n):
    if state.get("schema")!=SCHEMA:raise ValueError("Human hinge schema")
    size=np.asarray(state["shared_size"],float);ang=np.asarray(state["angles"],float)
    if size.shape!=(3,) or not np.isfinite(size).all() or np.any(size<=0):raise ValueError("Human shared size")
    if ang.shape!=(n,2) or not np.isfinite(ang).all() or np.any(np.abs(ang[:,0])>.65) or np.any(np.abs(ang[:,1])>1.85):raise ValueError("Human joint bounds")
    if len(state.get("state_source",[]))!=n:raise ValueError("Human evidence timeline")
def parts(entity,t):
    import torch
    state=entity["deformation"];width,height,depth=state["shared_size"]
    body,head,_,_=vertices(torch.tensor(height,dtype=torch.float64),torch.tensor(width,dtype=torch.float64),torch.tensor(depth,dtype=torch.float64),torch.tensor(state["angles"],dtype=torch.float64)[t:t+1],head_ratio=state.get("head_radius_height_fraction",.08),simple_cylinder=state.get("simple_cylinder",False))
    T=np.asarray(entity["entity_to_world"])[t];v=body[0].numpy()@T[:3,:3].T+T[:3,3];c=head[0].numpy()@T[:3,:3].T+T[:3,3]
    M=np.eye(4);M[:3,:3]=T[:3,:3]*height*state.get("head_radius_height_fraction",.08)*2;M[:3,3]=c
    bodypart=dict(kind="cage",entity_id=entity["id"],component="body",vertices=v.tolist(),faces=faces(SECTIONS,CIRCUMFERENCE).tolist(),canonical_to_world=np.eye(4).tolist())
    if state.get("simple_cylinder",False):
        bodypart["smooth_shading"]=True
        return [bodypart,dict(kind="sphere",entity_id=entity["id"],component="head",canonical_to_world=M.tolist())]
    return [dict(kind="cage",entity_id=entity["id"],component="body",vertices=v.tolist(),faces=faces(SECTIONS,CIRCUMFERENCE).tolist(),canonical_to_world=np.eye(4).tolist()),
      dict(kind="sphere",entity_id=entity["id"],component="head",canonical_to_world=M.tolist())]
