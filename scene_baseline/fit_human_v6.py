"""Fit only shared human dimensions, two smooth joint angles, and one planted-foot root."""
import copy
import numpy as np
import torch
import torch.nn.functional as F
from scipy.interpolate import CubicSpline
from scipy.ndimage import distance_transform_edt,binary_erosion
from .storage import new_run,reference,verify,read_json,write_json
from .scene import load_scene,validate_scene
from .human_hinge_v6 import vertices,SCHEMA
def fit(scene_path,entity_id,out,steps=700,visible_weight=.2):
    scene,m,a=load_scene(scene_path);e=next(x for x in scene["entities"] if x["id"]==entity_id)
    md=read_json(verify(scene["masks_manifest"]))
    with np.load(verify(md["data"])) as z:mask=z["masks"][z["ids"].tolist().index(entity_id)]
    n,H,W=mask.shape;torch.set_num_threads(6)
    f=lambda x:torch.as_tensor(np.asarray(x),dtype=torch.float32)
    scale=float(np.median(a["depth"]));up=np.asarray(scene["ground_normal"],float);up/=np.linalg.norm(up)
    right=a["c2w"][0,:3,0].copy();right-=up*(up@right);right/=np.linalg.norm(right)
    basis=np.stack([right,up,np.cross(right,up)],1)
    ground=next(x for x in scene["entities"] if x["id"]=="ground")
    floor=np.asarray(ground["entity_to_world"])[0,:3,3]
    feet=[];extents=[];pixels=[];boundaries=[]
    for t in range(n):
        v,u=np.nonzero(mask[t])
        if len(u)<24:raise ValueError("Need explicit sparse/missing human fallback before fitting this clip")
        bottom=np.quantile(v,.98);ux=np.median(u[v>=bottom-2]);vy=float(bottom)
        ray=a["c2w"][t,:3,:3]@(np.linalg.inv(a["K"][t])@np.array([ux,vy,1.]))
        den=ray@up
        if abs(den)<1e-6:raise ValueError("Foot ray parallel to ground")
        foot=a["c2w"][t,:3,3]+ray*((floor-a["c2w"][t,:3,3])@up)/den;feet.append(foot)
        ids=np.linspace(0,len(u)-1,min(220,len(u))).round().astype(int)
        pixels.append(np.column_stack([u[ids],v[ids]]))
        boundary=mask[t]&~binary_erosion(mask[t],iterations=1)
        vv,uu=np.nonzero(boundary);ii=np.linspace(0,len(uu)-1,min(180,len(uu))).round().astype(int)
        boundaries.append(np.column_stack([uu[ii],vv[ii]]))
        world=(np.column_stack([u,v,np.ones(len(u))])@np.linalg.inv(a["K"][t]).T*a["depth"][t,v,u,None])@a["c2w"][t,:3,:3].T+a["c2w"][t,:3,3]
        extents.append(np.quantile(world@up,.99)-np.quantile(world@up,.01))
    # Upright frames have maximal vertical extent; no hand-authored per-frame poses.
    height0=float(np.quantile(extents,.90));foot0=np.median(feet,0)
    root=torch.nn.Parameter(torch.zeros(2));size_raw=torch.nn.Parameter(torch.zeros(3))
    controls=torch.nn.Parameter(torch.zeros((min(n,7),2)))
    B=f(CubicSpline(np.linspace(0,1,min(n,7)),np.eye(min(n,7)),axis=0,bc_type="natural")(np.linspace(0,1,n)))
    # Initialize waist flexion from observed silhouette covariance in camera pixels.
    init=[]
    for pix in pixels:
        q=pix-pix.mean(0);_,vec=np.linalg.eigh(np.cov(q.T));axis=vec[:,-1]
        if axis[1]<0:axis=-axis
        init.append(np.clip(-np.arctan2(axis[0],axis[1]),-1.2,1.2))
    with torch.no_grad():controls[:,1]=f(np.arctanh(np.clip(np.interp(np.linspace(0,n-1,min(n,7)),np.arange(n),init)/1.75,-.8,.8)))
    optimizer=torch.optim.Adam([root,size_raw,controls],lr=.015)
    K=f(a["K"]);wc=f(np.linalg.inv(a["c2w"]));R=f(basis);U=f(up)
    sdf=f(np.stack([distance_transform_edt(~mm) for mm in mask]))[:,None]
    maskim=f(mask)[:,None];depthim=f(a["depth"]/scale)[:,None]
    pp=np.stack([p[np.linspace(0,len(p)-1,180).round().astype(int)] for p in pixels])
    target=f(pp);targetdepth=f(np.stack([a["depth"][t,pp[t,:,1],pp[t,:,0]] for t in range(n)]))/scale
    def state():
        height=f(height0/scale)*torch.exp(.25*torch.tanh(size_raw[1]))
        width=height*(.25+.18*torch.sigmoid(size_raw[0]))
        depth=height*(.16+.16*torch.sigmoid(size_raw[2]))
        angles=(B@torch.tanh(controls))*f([.6,1.75])
        body,head,headpoints,axis=vertices(height,width,depth,angles)
        C=f(foot0/scale)+R[:,[0,2]]@(.15*height*torch.tanh(root))
        local=torch.cat([body,headpoints],1)
        world=local@R.T+C
        return height,width,depth,angles,world,C,axis
    def objective():
        height,width,depth,angles,world,C,axis=state()
        cam=world*scale@wc[:,:3,:3].transpose(1,2)+wc[:,:3,3][:,None]
        pix=cam@K.transpose(1,2);uv=pix[...,:2]/cam[...,2:].clamp_min(.01)
        grid=torch.stack([2*(uv[...,0]+.5)/W-1,2*(uv[...,1]+.5)/H-1],-1)[:,None]
        outside=F.grid_sample(sdf,grid,align_corners=False,padding_mode="border")[:,0,0].mean()/H
        # Target -> projected geometry prevents shrinking to fit inside the silhouette.
        distances=torch.cdist(target,uv)
        nearest,ix=distances.min(-1)
        coverage=torch.sqrt((nearest/H).square()+.002**2).mean()
        known=F.grid_sample(maskim,grid,align_corners=False,padding_mode="zeros")[:,0,0]>.5
        targetz=F.grid_sample(depthim,grid,align_corners=False,padding_mode="border")[:,0,0]
        # Front surface: for a projected bin use only its nearest visible sample.
        with torch.no_grad():
            xx=uv[...,0].round().long().clamp(0,W-1);yy=uv[...,1].round().long().clamp(0,H-1);idx=yy*W+xx
            zbuf=torch.full((n,H*W),float("inf"));zbuf.scatter_reduce_(1,idx,cam[...,2]/scale,reduce="amin",include_self=True)
            front=(cam[...,2]/scale)<=zbuf.gather(1,idx)+height*.03
        use=(known&front).float()
        dz=cam[...,2]/scale-targetz
        visible=((torch.sqrt(dz.square()+.005**2)-.005)*use).sum()/use.sum().clamp_min(1)
        temporal=torch.diff(angles,n=2,dim=0).square().mean()
        knee_prior=angles[:,0].square().mean()
        loss=1.5*coverage+.65*outside+visible_weight*visible+.01*temporal+.008*knee_prior+.004*size_raw.square().mean()+.002*root.square().mean()
        return loss,dict(coverage=coverage,outside=outside,visible_depth=visible,angle_accel=temporal)
    out=new_run(out);best=None;bestloss=float("inf");history=[]
    for it in range(steps+1):
        optimizer.zero_grad();loss,terms=objective()
        if not torch.isfinite(loss):raise ValueError("Nonfinite hinge fit")
        if float(loss)<bestloss:bestloss=float(loss.detach());best=[x.detach().clone() for x in [root,size_raw,controls]]
        if it%100==0 or it==steps:
            row=dict(step=it,loss=float(loss.detach()),**{k:float(v.detach()) for k,v in terms.items()});history.append(row);print(row,flush=True)
        if it==steps:break
        loss.backward();torch.nn.utils.clip_grad_norm_([root,size_raw,controls],2);optimizer.step()
    with torch.no_grad():
        for x,y in zip([root,size_raw,controls],best):x.copy_(y)
        height,width,depth,angles,world,C,axis=state()
    result=copy.deepcopy(scene);ee=next(x for x in result["entities"] if x["id"]==entity_id)
    T=np.tile(np.eye(4),(n,1,1));T[:,:3,:3]=basis;T[:,:3,3]=C.detach().numpy()*scale
    ee["entity_to_world"]=T.tolist();ee["deformation"]={"schema":SCHEMA,"shared_size":[float(width.detach())*scale,float(height.detach())*scale,float(depth.detach())*scale],"angles":angles.detach().tolist(),"state_source":["same_stage_mask_depth_joint_fit"]*n,"root_model":"shared planted-foot root; selected from observed stationary feet","joint_locations_height_fraction":[.27,.52],"transition_halfwidth_height_fraction":[.055,.075],"head_radius_height_fraction":.08}
    ee["parts"]=[dict(id="body",kind="cylinder",canonical_to_entity=np.diag([float(width.detach())*scale,float(height.detach())*.84*scale,float(depth.detach())*scale,1.]).tolist()),dict(id="head",kind="sphere",canonical_to_entity=np.diag([float(height.detach())*.16*scale]*3+[1.]).tolist())]
    ee["renderable"]=[True]*n
    for k in ["continuous_motion","exists","state_source","observation_available","observation_used"]:ee.pop(k,None)
    result.pop("modeling_execution_plan",None);ee.pop("modeling_policy",None)
    result["stage"]="coarse_knee_waist_head_v6_candidate"
    result["limitations"]+=["Shared planted feet inferred for this stationary action; locomotion needs a root trajectory.","Two coarse joint bends, no detailed limbs; body covering includes clothes and arm envelope.","Depth is predicted, not ground truth; circle head has fixed shared proportions."]
    validate_scene(result,n);write_json(out/"scene.json",result)
    write_json(out/"report.json",dict(source_scene=reference(scene_path),steps=steps,visible_weight=visible_weight,history=history,shared_size=ee["deformation"]["shared_size"],free_parameters=int(sum(x.numel() for x in [root,size_raw,controls])),fixed_cameras=True,head="attached sphere",joint_count=2,automatic_acceptance=False))
    return result
