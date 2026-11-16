"""Coarse shared tube fit: fixed input cameras, root motion and constrained bend."""
import copy,time
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt
from .storage import reference,write_json,new_run,verify,read_json
from .scene import load_scene,validate_scene
from .cage_v4 import torch_vertices,so3,make_state,projector

def fit(scene_path,entity_id,out,steps=500,allow_bend=True,device="cpu",sample_count=160):
    scene,m,a=load_scene(scene_path);entity=next(e for e in scene["entities"] if e["id"]==entity_id)
    if entity["motion"]!="deformable":raise ValueError("Explicit cage representation required")
    from .modeling_policy_v4 import entity_budget
    steps=entity_budget(scene,entity_id,steps)
    if np.any(np.asarray(entity["deformation"].get("radial_log",0))!=0):raise ValueError("Cage fitter currently requires zero radial controls; rendering supports nonzero controls")
    md=read_json(verify(scene["masks_manifest"]))
    with np.load(verify(md["data"])) as z:masks=z["masks"];ids=z["ids"].tolist()
    mask=masks[ids.index(entity_id)]&(masks.sum(0)==1);n,H,W=mask.shape
    observed=mask.reshape(n,-1).sum(-1)>=24
    if observed.sum()<2:raise ValueError("Insufficient cage observations")
    torch.set_num_threads(6);dtype=torch.float32
    f=lambda x:torch.as_tensor(np.asarray(x),dtype=dtype,device=device)
    scale=float(np.median(a["depth"]));initial=np.asarray(entity["entity_to_world"])
    base_R=f(initial[:,:3,:3]);center=torch.nn.Parameter(f(initial[:,:3,3]/scale))
    rotation=torch.nn.Parameter(torch.zeros((n,3),dtype=dtype,device=device))
    raw_bend=torch.nn.Parameter(torch.zeros((n,5,2),dtype=dtype,device=device),requires_grad=allow_bend)
    shared0=f(entity["deformation"]["shared_size"])/scale
    log_size=torch.nn.Parameter(torch.zeros(3,dtype=dtype,device=device))
    parameters=[center,rotation,log_size]+([raw_bend] if allow_bend else [])
    optimizer=torch.optim.Adam(parameters,lr=.008)
    P=f(projector());K=f(a["K"]);wc=f(np.linalg.inv(a["c2w"]))
    pts=np.zeros((n,sample_count,3));weights=np.zeros((n,sample_count))
    yy=np.zeros((n,sample_count));xx=yy.copy()
    for t in range(n):
        v,u=np.nonzero(mask[t])
        if not len(u):continue
        ii=np.linspace(0,len(u)-1,min(sample_count,len(u))).round().astype(int);v,u=v[ii],u[ii];count=len(u)
        ray=np.column_stack([u,v,np.ones(count)])@np.linalg.inv(a["K"][t]).T
        world=(ray*a["depth"][t,v,u,None])@a["c2w"][t,:3,:3].T+a["c2w"][t,:3,3]
        pts[t,:count]=world/scale;weights[t,:count]=np.clip(a["confidence"][t,v,u]/max(float(np.median(a["confidence"][t,v,u])),1e-8),.1,3)
        yy[t,:count]=v;xx[t,:count]=u
    target=f(pts);wt=f(weights);available=f(observed)[:,None]
    sdf=f(np.stack([distance_transform_edt(~x) for x in mask]))[:,None]
    mask_image=f(mask)[:,None];depth_image=f(a["depth"]/scale)[:,None]
    def state():
        R=base_R@so3(.9*torch.tanh(rotation))
        size=shared0*torch.exp(log_size.clamp(-.4,.4))
        bend=torch.einsum("ij,tjk->tik",P,.12*torch.tanh(raw_bend))
        return R,center,size,bend
    def objective():
        R,C,size,bend=state()
        q=torch.einsum("nij,npj->npi",R.transpose(1,2),target-C[:,None])
        level=((q[:,:,1]/size[1]+.5)*4).clamp(0,4-1e-5);lo=level.long();alpha=level-lo
        t=torch.arange(n,device=device)[:,None]
        offset=bend[t,lo]*(1-alpha[...,None])+bend[t,lo+1]*alpha[...,None]
        radial=q[:,:,[0,2]]-size[1]*offset
        ellipse=torch.sqrt((radial/size[[0,2]]*.2e1).square().sum(-1)+1e-8)-1
        side=ellipse*size[[0,2]].min()*.5
        cap=q[:,:,1].abs()-size[1]*.5
        signed=torch.linalg.norm(torch.stack([side.clamp_min(0),cap.clamp_min(0)],-1),dim=-1)+torch.minimum(torch.maximum(side,cap),torch.zeros_like(side))
        surface=((torch.sqrt(signed.square()+.01**2)-.01)*wt).sum()/wt.sum().clamp_min(1)
        local=torch_vertices(size,bend,circumference=16)
        world=torch.einsum("nij,npj->npi",R,local)+C[:,None]
        cam=torch.einsum("nij,npj->npi",wc[:,:3,:3],world*scale)+wc[:,:3,3][:,None]
        pixel=torch.einsum("nij,npj->npi",K,cam);uv=pixel[...,:2]/cam[...,2:].clamp_min(.01)
        grid=torch.stack([2*(uv[...,0]+.5)/W-1,2*(uv[...,1]+.5)/H-1],-1)[:,None]
        image_distance=F.grid_sample(sdf,grid,align_corners=False,padding_mode="border")[:,0,0]
        silhouette=(image_distance/max(H,W)*available).sum()/(available.sum()*local.shape[1]).clamp_min(1)
        # Surface samples facing camera; depth is never inferred from bbox interiors.
        phi=torch.arange(16,device=device)*2*torch.pi/16
        normal=torch.stack([torch.cos(phi)/size[0],torch.zeros(16,device=device),torch.sin(phi)/size[2]],-1).repeat(5,1)
        normal=normal/torch.linalg.norm(normal,dim=-1,keepdim=True)
        worldnormal=torch.einsum("nij,pj->npi",R,normal)
        origin=f(a["c2w"][:,:3,3])/scale
        facing=(worldnormal*(origin[:,None]-world)).sum(-1)>0
        target_mask=F.grid_sample(mask_image,grid,align_corners=False,padding_mode="zeros")[:,0,0]>.5
        depth=F.grid_sample(depth_image,grid,align_corners=False,padding_mode="border")[:,0,0]
        use=(facing&target_mask&(cam[...,2]>0)).float()*available
        residual=cam[...,2]/scale-depth
        visible=((torch.sqrt(residual.square()+.01**2)-.01)*use).sum()/use.sum().clamp_min(1)
        # Regularization is dimensionless: avoid shape pumping and unsupported rapid bends.
        temporal=(torch.diff(C,n=2,dim=0).square().mean()+.1*torch.diff(rotation,n=2,dim=0).square().mean()) if n>=3 else C.sum()*0
        bend_time=torch.diff(bend,n=2,dim=0).square().mean() if n>=3 else bend.sum()*0
        bend_space=torch.diff(bend,n=2,dim=1).square().mean()
        prior=log_size.square().mean()
        loss=surface+.25*visible+.3*silhouette+.03*prior+.02*temporal+.03*bend_time+.01*bend_space
        return loss,{"surface":surface,"visible_depth":visible,"silhouette":silhouette,"size_prior":prior,"motion":temporal,"bend_time":bend_time}
    dest=new_run(out);best=None;bestvalue=float("inf");history=[];start=time.perf_counter()
    for step in range(steps+1):
        optimizer.zero_grad();loss,terms=objective()
        if not torch.isfinite(loss):raise ValueError("Nonfinite cage fit")
        value=float(loss.detach())
        if value<bestvalue:bestvalue=value;best=[p.detach().clone() for p in parameters]
        if step%100==0 or step==steps:
            row={"step":step,"loss":value,**{k:float(v.detach()) for k,v in terms.items()}}
            history.append(row);print(row,flush=True)
        if step==steps:break
        loss.backward();torch.nn.utils.clip_grad_norm_(parameters,2);optimizer.step()
    with torch.no_grad():
        for p,v in zip(parameters,best):p.copy_(v)
        R,C,size,bend=state();R=R.cpu().numpy();C=C.cpu().numpy()*scale;size=size.cpu().numpy()*scale;bend=bend.cpu().numpy()
    # Missing masks do not delete entities. Freeze shape at nearest observed frame and mark inference.
    keys=np.flatnonzero(observed)
    for t in np.flatnonzero(~observed):bend[t]=bend[keys[np.argmin(abs(keys-t))]]
    result=copy.deepcopy(scene);e=next(e for e in result["entities"] if e["id"]==entity_id)
    poses=np.tile(np.eye(4),(n,1,1));poses[:,:3,:3]=R;poses[:,:3,3]=C;e["entity_to_world"]=poses.tolist()
    e["deformation"]=make_state(size,bend,["observed_fit" if x else "shape_held_motion_prior_unverified" for x in observed])
    e["observed"]=observed.tolist();e["renderable"]=[True]*n
    for key in ["continuous_motion","exists","observation_available","observation_used","state_source"]:e.pop(key,None)
    result["stage"]="coarse_deformable_fit_v4_candidate";result.pop("continuous_camera",None)
    result["limitations"]+=["coarse tube cannot express separated limbs or arbitrary topology","predicted depth and masks are fallible","surface objective is approximate, no guaranteed collision-free deformation"]
    validate_scene(result,n);write_json(dest/"scene.json",result)
    report={"source_scene":reference(scene_path),"entity":entity_id,"allow_bend":allow_bend,"camera_optimized":False,
      "steps":steps,"history":history,"seconds":time.perf_counter()-start,"observed_frames":int(observed.sum()),
      "max_bend_height_fraction":float(np.abs(bend).max()),"shared_size":size.tolist(),"automatic_acceptance":False}
    write_json(dest/"report.json",report);return report
