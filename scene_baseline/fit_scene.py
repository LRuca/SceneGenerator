"""Fixed-camera scene fitting with shared rigid shapes and visibility-aware evidence."""
import copy,time
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt
from .scene import load_scene,load_observations,validate_scene
from .storage import reference,verify,new_run,write_json

def _intersection(o,d,kind):
    # o,d: T,S,3 in canonical primitive coordinates. Ray parameter remains camera-Z.
    eps=1e-8
    safe=torch.where(d.abs()>eps,d,torch.where(d>=0,eps,-eps))
    a=(-.5-o)/safe;b=(.5-o)/safe
    near=torch.minimum(a,b);far=torch.maximum(a,b)
    invalid=(d.abs()<eps)&(o.abs()>.5)
    if kind in ("box","plane"):
        lo=near.max(-1).values;hi=far.min(-1).values;valid=~invalid.any(-1)
    else:
        axes=[0,1,2] if kind=="sphere" else [0,2]
        qa=d[...,axes].square().sum(-1);qb=2*(o[...,axes]*d[...,axes]).sum(-1)
        qc=o[...,axes].square().sum(-1)-.25
        if kind=="cone":
            radius=.25-.5*o[...,1];slope=-.5*d[...,1]
            qa=qa-slope.square();qb=qb-2*radius*slope;qc=o[...,[0,2]].square().sum(-1)-radius.square()
        disc=qb.square()-4*qa*qc
        root=torch.sqrt(disc.clamp_min(1e-12));den=2*torch.where(qa.abs()>eps,qa,torch.full_like(qa,eps))
        r1=(-qb-root)/den;r2=(-qb+root)/den
        low=torch.minimum(r1,r2);high=torch.maximum(r1,r2)
        if kind=="sphere":
            lo=low;hi=high;valid=disc>=0
        else:
            lo=near[...,1];hi=far[...,1];valid=~invalid[...,1]
            positive=qa>eps
            lo=torch.where(positive,torch.maximum(lo,low),lo)
            hi=torch.where(positive,torch.minimum(hi,high),hi)
            valid=valid&~(positive&(disc<0))
            linear=(qa.abs()<=eps)&(qb.abs()>eps)
            bound=-qc/torch.where(qb.abs()>eps,qb,torch.full_like(qb,eps))
            lo=torch.where(linear&(qb<0),torch.maximum(lo,bound),lo)
            hi=torch.where(linear&(qb>0),torch.minimum(hi,bound),hi)
            valid=valid&~((qa.abs()<=eps)&(qb.abs()<=eps)&(qc>0))
            negative=(qa<-eps)&(disc>=0);left=negative&(lo<low)
            hi=torch.where(left,torch.minimum(hi,low),hi)
            lo=torch.where(negative&~left,torch.maximum(lo,high),lo)
    lo=lo.clamp_min(0.)
    hit=valid&(hi>lo+1e-6)&(lo>1e-5)
    return torch.where(hit,lo,torch.full_like(lo,1e6))

def _surface_distance(q,kind):
    if kind in ("box","plane"):
        a=q.abs()-.5
        return torch.linalg.norm(a.clamp_min(0),dim=-1)+torch.minimum(a.max(-1).values,torch.zeros_like(a[...,0]))
    if kind=="sphere":return torch.linalg.norm(q,dim=-1)-.5
    radial=torch.linalg.norm(q[...,[0,2]],dim=-1)
    if kind=="cylinder":
        a=torch.stack([radial-.5,q[...,1].abs()-.5],dim=-1)
        return torch.linalg.norm(a.clamp_min(0),dim=-1)+torch.minimum(a.max(-1).values,torch.zeros_like(radial))
    lateral=(radial-(.25-.5*q[...,1])).abs()/np.sqrt(1.25)
    return torch.minimum(lateral,(q[...,1]+.5).abs())+torch.relu(q[...,1].abs()-.5)

def fit(initial_scene,out,steps=160,resolution=48,device="cpu",lr=.015,entity_ids=None):
    if not 1<=steps<=10000 or not 8<=resolution<=256 or not 0<lr<1:raise ValueError("Invalid fit budget")
    torch.set_num_threads(8)
    scene,m,a=load_scene(initial_scene)
    all_ids={e["id"] for e in scene["entities"]}
    selected=all_ids if entity_ids is None else set(entity_ids)
    if not selected or not selected<=all_ids:raise ValueError("Unknown or empty optimization entity selection")
    _,_,h,md,masks,ids=load_observations(verify(scene["geometry_manifest"]),verify(scene["hypothesis"]),verify(scene["masks_manifest"]))
    if {e["id"] for e in scene["entities"]}-set(ids):raise ValueError("Scene has unmasked identities")
    n,H,W=a["depth"].shape;rh=min(resolution,H);rw=max(8,round(W/H*rh))
    yy=np.minimum(((np.arange(rh)+.5)*H/rh).astype(int),H-1)
    xx=np.minimum(((np.arange(rw)+.5)*W/rw).astype(int),W-1)
    vv,uu=np.meshgrid(yy,xx,indexing="ij")
    pix=np.column_stack([uu.ravel(),vv.ravel(),np.ones(rh*rw)])
    f=lambda x:torch.as_tensor(np.asarray(x),dtype=torch.float32,device=device)
    depth=f(a["depth"][:,vv,uu].reshape(n,-1));confidence=f(a["confidence"][:,vv,uu].reshape(n,-1))
    confidence=(confidence/confidence.median().clamp_min(1e-6)).clamp(.1,4)
    local=np.einsum("sj,nkj->nsk",pix,np.linalg.inv(a["K"]))
    dirs=f(np.einsum("nsj,nkj->nsk",local,a["c2w"][:,:3,:3]))
    origin=f(a["c2w"][:,:3,3])[:,None,:];points=origin+dirs*depth[...,None]
    sampled=masks[:,:,vv,uu].reshape(len(ids),n,-1)
    ambiguous=sampled.sum(0)>1
    known=f(sampled.any(0)&~ambiguous)
    world_scale=float(np.median(a["depth"]))
    w2c=f(np.linalg.inv(a["c2w"]));K=f(a["K"])
    params=[];nodes=[]
    for e in scene["entities"]:
        pose=np.asarray(e["entity_to_world"]);N=len(pose)
        center=torch.nn.Parameter(f(pose[:,:3,3]));yaw=torch.nn.Parameter(torch.zeros(N,device=device))
        logscale=torch.nn.Parameter(torch.zeros(3,device=device))
        plane=all(p["kind"]=="plane" for p in e["parts"])
        scale_free=f([1,0,1] if plane else [1,1,1])
        if e["id"] in selected:
            params +=[center,logscale]
            if e.get("fit_constraints",{}).get("lock_yaw",False):yaw.requires_grad_(False)
            else:params.append(yaw)
        else:
            for parameter in [center,yaw,logscale]:parameter.requires_grad_(False)
        base=f(pose[:,:3,:3]);mask=f(sampled[ids.index(e["id"])]&~ambiguous)
        sdf=np.stack([distance_transform_edt(~x.reshape(rh,rw)) for x in sampled[ids.index(e["id"])]])
        from .geometry import unit_mesh
        parts=[]
        for p in e["parts"]:
            mat=f(p["canonical_to_entity"]);mesh=unit_mesh("box" if p["kind"]=="plane" else p["kind"])[0]
            mesh=mesh[::max(1,int(np.ceil(len(mesh)/64)))]
            parts.append({"source":p,"matrix":mat,"samples":f(mesh)})
        nodes.append({"entity":e,"center":center,"center0":center.detach().clone(),"yaw":yaw,"logscale":logscale,
            "base":base,"mask":mask,"sdf":f(sdf)[:,None],"parts":parts,"scale_free":scale_free,
            "renderable":f(e["renderable"])})
    def transform(node):
        y=node["yaw"];c=torch.cos(y);s=torch.sin(y);z=torch.zeros_like(y);one=torch.ones_like(y)
        ry=torch.stack([c,z,s,z,one,z,-s,z,c],dim=-1).reshape(-1,3,3)
        R=node["base"]@ry;C=node["center"]
        if len(C)==1:R=R.expand(n,-1,-1);C=C.expand(n,-1)
        scale=torch.exp(node["logscale"]*node["scale_free"])
        return R,C,scale
    def loss_value():
        total_surface=torch.zeros((),device=device);total_sil=torch.zeros_like(total_surface)
        total_landmark=torch.zeros_like(total_surface)
        predicted=[];free=torch.zeros_like(total_surface)
        for node in nodes:
            R,C,scale=transform(node);part_depth=[];part_sdf=[]
            lm=node["entity"].get("landmarks")
            if lm is not None:
                lp=f(lm["entity_points"]);target_uv=f(lm["image_points"]);valid=f(lm["valid"])
                world=torch.einsum("nij,pj->npi",R,lp*scale)+C[:,None,:]
                cam=torch.einsum("nij,npj->npi",w2c[:,:3,:3],world)+w2c[:,:3,3][:,None,:]
                pix=torch.einsum("nij,npj->npi",K,cam)
                uv=pix[...,:2]/cam[...,2:].clamp_min(1e-5)
                rel=(uv-target_uv)/max(H,W)
                error=torch.sqrt(rel.square().sum(-1)+1e-6)-.001
                total_landmark=total_landmark+float(lm.get("weight",8.))*(error*valid).sum()/valid.sum().clamp_min(1)

            for part in node["parts"]:
                mat=part["matrix"];linear=torch.diag(scale)@mat[:3,:3]
                shift=(R@(scale*mat[:3,3])[None,:,None]).squeeze(-1)+C
                full=R@linear;inverse=torch.linalg.inv(full)
                o=torch.einsum("nij,nsj->nsi",inverse,origin-shift[:,None,:])
                d=torch.einsum("nij,nsj->nsi",inverse,dirs)
                dp=_intersection(o,d,part["source"]["kind"])
                dp=torch.where(node["renderable"][:,None]>0,dp,torch.full_like(dp,1e6))
                part_depth.append(dp)
                q=torch.einsum("nij,nsj->nsi",inverse,points-shift[:,None,:])
                # Canonical distance -> approximate world scale; robust to gross depth outliers.
                axis_lengths=torch.linalg.norm(full,dim=1)
                if part["source"]["kind"] in ("box","plane"):
                    delta=(q.abs()-.5)*axis_lengths[:,None,:]
                    signed=torch.linalg.norm(delta.clamp_min(0),dim=-1)+torch.minimum(delta.max(-1).values,torch.zeros_like(delta[...,0]))
                else:
                    signed=_surface_distance(q,part["source"]["kind"])*axis_lengths.min(1).values[:,None]
                part_sdf.append(signed.abs()/world_scale)
                surface=torch.einsum("nij,pj->npi",full,part["samples"])+shift[:,None,:]
                cam=torch.einsum("nij,npj->npi",w2c[:,:3,:3],surface)+w2c[:,:3,3][:,None,:]
                projected=torch.einsum("nij,npj->npi",K,cam)
                uv=projected[...,:2]/cam[...,2:].clamp_min(1e-5)
                grid=torch.stack([2*(uv[...,0]+.5)/W-1,2*(uv[...,1]+.5)/H-1],-1)[:,None]
                distance=F.grid_sample(node["sdf"],grid,align_corners=False,padding_mode="border")[:,0,0]
                visible_depth=F.grid_sample(depth.reshape(n,1,rh,rw),grid,align_corners=False,padding_mode="border")[:,0,0]
                visible=((cam[...,2]>0)&(cam[...,2]<=visible_depth*1.05)&(grid[:,0].abs().max(-1).values<1)).float()
                visible=visible*node["renderable"][:,None]
                total_sil=total_sil+((distance/max(rh,rw))*visible).sum()/visible.sum().clamp_min(1)
            ed=torch.stack(part_depth).min(0).values;predicted.append(ed)
            dist=torch.stack(part_sdf).min(0).values;mask=node["mask"]
            total_surface=total_surface+((torch.sqrt(dist.square()+.0004)-.02)*mask*confidence).sum()/(mask*confidence).sum().clamp_min(1)
            other=(1-mask)*known;hit=(ed<1e5).float()
            free=free+(torch.relu((depth-ed)/depth-.04)*other*hit).sum()/other.sum().clamp_min(1)
        rendered=torch.stack(predicted).min(0).values
        hit=(rendered<1e5).float();rel=torch.where(hit>0,(rendered-depth)/depth,torch.ones_like(depth))
        robust=torch.sqrt(rel.square()+.0025)-.05
        depth_loss=(robust*known*confidence).sum()/(known*confidence).sum().clamp_min(1)
        shape=sum((node["logscale"]*node["scale_free"]).square().mean() for node in nodes)
        motion=torch.zeros((),device=device)
        for node in nodes:
            if node["center"].shape[0]>=3:
                dt=f(np.diff(a["times_s"]))[:,None]
                vel=torch.diff(node["center"],dim=0)/dt
                motion=motion+torch.diff(vel,dim=0).square().mean()/world_scale**2
                motion=motion+.1*torch.diff(node["yaw"],n=2).square().mean()
        support=torch.zeros((),device=device);by_id={node["entity"]["id"]:node for node in nodes}
        up=f(scene["ground_normal"])
        for r in scene.get("relations",[]):
            if r.get("relation")!="supported_by" or r.get("subject") not in by_id or r.get("object") not in by_id:continue
            child,parent=by_id[r["subject"]],by_id[r["object"]]
            R,C,scale=transform(child);_,P,_=transform(parent)
            min_heights=[]
            for part in child["parts"]:
                local=(part["samples"]@part["matrix"][:3,:3].T+part["matrix"][:3,3])*scale
                world=torch.einsum("nij,pj->npi",R,local)+C[:,None,:]
                min_heights.append(((world-P[:,None,:])*up).sum(-1).min(-1).values)
            contact=torch.stack(min_heights).min(0).values/world_scale
            support=support+contact.square().mean()
        pieces={"surface":total_surface,"depth":depth_loss,"silhouette":total_sil,"free_space":free,
            "shape_prior":shape,"motion":motion,"support":support,"landmark":total_landmark}
        loss=total_surface+depth_loss+.4*total_sil+.5*free+.01*shape+.0001*motion+.3*support+total_landmark
        return loss,pieces
    optimizer=torch.optim.Adam(params,lr=lr)
    history=[];start=time.perf_counter();best=None;best_value=float("inf")
    for step in range(steps+1):
        optimizer.zero_grad();loss,terms=loss_value()
        if not torch.isfinite(loss):raise ValueError("Nonfinite scene fit loss")
        value=float(loss.detach())
        if value<best_value:
            best_value=value;best=[p.detach().clone() for p in params]
        if step%20==0 or step==steps:
            row={"step":step,"loss":value,**{k:float(v.detach()) for k,v in terms.items()}}
            history.append(row);print(row,flush=True)
        if step==steps:break
        loss.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in params):raise ValueError("Nonfinite fit gradient")
        torch.nn.utils.clip_grad_norm_(params,5.);optimizer.step()
        with torch.no_grad():
            for node in nodes:node["logscale"].clamp_(-1.5,1.5)
    with torch.no_grad():
        for parameter,value in zip(params,best):parameter.copy_(value)
    result=copy.deepcopy(scene)
    with torch.no_grad():
        for e,node in zip(result["entities"],nodes):
            if e["id"] not in selected:continue
            R,C,scale=transform(node)
            count=1 if e["motion"]=="static" else n
            poses=np.tile(np.eye(4),(count,1,1));poses[:,:3,:3]=R[:count].cpu().numpy();poses[:,:3,3]=C[:count].cpu().numpy()
            e["entity_to_world"]=poses.tolist()
            if "landmarks" in e:
                e["landmarks"]["entity_points"]=(np.asarray(e["landmarks"]["entity_points"])*scale.cpu().numpy()).tolist()
            for p in e["parts"]:
                mat=np.asarray(p["canonical_to_entity"]);mat[:3,:]=np.diag(scale.cpu().numpy())@mat[:3,:]
                p["canonical_to_entity"]=mat.tolist()
    result["stage"]="fitted_candidate_not_validated";result["fit_source"]=reference(initial_scene)
    result["limitations"]+=["silhouette uses projected surface-sample distance, not exact differentiable rasterization",
        "VGGT depth and predicted cameras are not ground truth","unknown/ambiguous mask pixels excluded from fit, reported by delivery"]
    validate_scene(result,n);p=new_run(out)
    write_json(p/"scene.json",result)
    report={"history":history,"best_loss":best_value,"steps":steps,"resolution_hw":[rh,rw],"device":device,
        "optimized_entities":sorted(selected),"frozen_entities":sorted(all_ids-selected),
        "seconds":time.perf_counter()-start,"known_pixel_fraction":float(known.mean().cpu()),"camera_optimized":False,
        "train_protocol":"all selected times; independent heldout evaluation not implemented",
        "inputs":{"scene":reference(initial_scene),"geometry":scene["geometry_manifest"],"masks":scene["masks_manifest"]}}
    write_json(p/"report.json",report);return report
