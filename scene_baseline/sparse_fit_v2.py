"""Shared-shape pose fitting from sparse evidence; masks are optional observations."""
import argparse,copy
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from .scene import load_scene,validate_scene
from .storage import new_run,reference,verify,read_json,write_json
from .geometry import unit_mesh,transform_mesh

def fit(scene_path,tracks_manifest,review_path,out,steps=1000,device="cpu"):
 scene,m,a=load_scene(scene_path);tr=read_json(tracks_manifest);review=read_json(review_path)
 if tr["geometry_manifest"]!=scene["geometry_manifest"]:raise ValueError("Cross-geometry tracks")
 if review["tracks_manifest"]!=reference(tracks_manifest) or review["decision"]!="accept_candidate_evidence":
  raise ValueError("Agent identity review missing or bound to other tracks")
 e=next(x for x in scene["entities"] if x["id"]==tr["entity_id"])
 with np.load(verify(tr["data"])) as z:
  uv=z["uv"].copy();valid=z["valid"].copy();anchor=z["anchor_frame"].copy()
  photo=z["photo_error"].copy();fb=z["fb_error"].copy()
 valid &= np.isfinite(uv).all(-1)&np.isfinite(photo)&np.isfinite(fb)&(photo>=0)&(fb>=0)
 uv=np.where(valid[...,None],uv,0)
 n,P=valid.shape;H,W=a["depth"].shape[1:]
 if P<4 or valid.sum()<8:raise ValueError("Insufficient sparse evidence; current-chat agent review required")
 if e["motion"]!="rigid":raise ValueError("Sparse pose fitting requires rigid entity")
 initial=np.asarray(e["entity_to_world"])
 up=np.asarray(scene["ground_normal"],float);up/=np.linalg.norm(up)
 right=a["c2w"][0,:3,0].copy();right-=up*(right@up);right/=np.linalg.norm(right)
 basis=np.stack([right,up,np.cross(right,up)],axis=1)
 rel=basis.T@initial[:,:3,:3]
 yaw0=np.unwrap(2*np.arctan2(rel[:,0,2],rel[:,0,0]))/2
 rotation0=basis@Rotation.from_euler("y",yaw0).as_matrix()
 pix=np.rint(uv).astype(int);pix[:,:,0]=pix[:,:,0].clip(0,W-1);pix[:,:,1]=pix[:,:,1].clip(0,H-1)
 tt=np.arange(n)[:,None]
 zdepth=a["depth"][tt,pix[:,:,1],pix[:,:,0]]
 zconf=a["confidence"][tt,pix[:,:,1],pix[:,:,0]]
 ray=np.einsum("nij,npj->npi",np.linalg.inv(a["K"]),np.concatenate([uv,np.ones((n,P,1))],-1))
 observed_world=np.einsum("nij,npj->npi",a["c2w"][:,:3,:3],ray*zdepth[:,:,None])+a["c2w"][:,:3,3][:,None,:]
 q0=np.stack([rotation0[t].T@(observed_world[t,j]-initial[t,:3,3]) for j,t in enumerate(anchor)])
 # Every body point is initialized only from its own anchor's foreground correspondence.
 scale=float(np.median(a["depth"]));q0/=scale
 conf=zconf/max(float(np.median(zconf[valid])),1e-6)
 quality=np.zeros_like(photo,dtype=float)
 quality[valid]=np.exp(-np.clip(photo[valid],0,1e4)/25)*np.exp(-np.clip(fb[valid],0,1e4)/1.5)*np.clip(np.sqrt(conf[valid]),.2,2)
 support=(valid.sum(1)>=6)
 # Missing-frame center initialization uses point correspondences and measured depth, never a mask centroid or long extrapolation.
 pred_offset=np.einsum("nij,pj->npi",rotation0,q0*scale)
 cc=initial[:,:3,3].copy()
 for t in np.flatnonzero(support):
  cc[t]=np.median((observed_world[t]-pred_offset[t])[valid[t]],axis=0)
 ground=None
 for r in scene.get("relations",[]):
  if r.get("subject")==e["id"] and r.get("relation")=="supported_by":
   parent=next(x for x in scene["entities"] if x["id"]==r["object"])
   ground=float(np.asarray(parent["entity_to_world"])[0,:3,3]@up)/scale
 verts=np.concatenate([transform_mesh(unit_mesh("box" if p["kind"]=="plane" else p["kind"]),p["canonical_to_entity"])[0] for p in e["parts"]])/scale
 torch.set_num_threads(8)
 f=lambda x:torch.as_tensor(np.asarray(x),dtype=torch.float32,device=device)
 B=f(basis);U=f(up);V=f(verts);K=f(a["K"]);w2c=f(np.linalg.inv(a["c2w"]));uvt=f(uv);deptht=f(zdepth/scale)
 wt=f(quality);qref=f(q0)
 center=torch.nn.Parameter(f(cc/scale));yaw=torch.nn.Parameter(f(yaw0));q=torch.nn.Parameter(qref.clone())
 optimizer=torch.optim.Adam([center,yaw,q],lr=.002)
 def state():
  c,s=torch.cos(yaw),torch.sin(yaw);zz=torch.zeros_like(c);one=torch.ones_like(c)
  rr=torch.stack([c,zz,s,zz,one,zz,-s,zz,c],-1).reshape(n,3,3)
  R=B@rr;C=center
  if ground is not None:
   low=(torch.einsum("nij,pj->npi",R,V)*U).sum(-1).min(-1).values
   C=C+(ground-low-(C*U).sum(-1))[:,None]*U
  return R,C
 def robust(x,delta):
  return delta*(torch.sqrt(x.square()+delta**2)-delta)
 hist=[];best=None;bestloss=float("inf")
 extent=max(float(np.linalg.norm(np.ptp(verts,axis=0))),.01)
 for it in range(steps+1):
  optimizer.zero_grad();R,C=state()
  world=torch.einsum("nij,pj->npi",R,q)+C[:,None,:]
  cam=torch.einsum("nij,npj->npi",w2c[:,:3,:3],world*scale)+w2c[:,:3,3][:,None,:]
  pixels=torch.einsum("nij,npj->npi",K,cam)
  puv=pixels[...,:2]/cam[...,2:].clamp_min(.01*scale)
  reproj=torch.linalg.norm((puv-uvt)/max(H,W),dim=-1)
  point=robust(reproj,.008)
  deptherr=(cam[...,2]/scale-deptht)/deptht.clamp_min(.1)
  dl=robust(deptherr,.025)
  reproj_loss=(point*wt).sum()/wt.sum().clamp_min(1)
  depth_loss=(dl*wt).sum()/wt.sum().clamp_min(1)
  # Shared local body landmarks are soft hypotheses, not fixed ground truth from one VGGT frame.
  anchor_loss=robust(torch.linalg.norm(q-qref,dim=-1)/extent,.15).mean()
  accel=torch.diff(C,n=2,dim=0)/extent
  # Linear tails allow coherent abrupt motion instead of erasing it quadratically.
  motion_loss=robust(accel,.02).mean()+.1*robust(torch.diff(yaw,n=2),.05).mean()
  loss=20*reproj_loss+depth_loss+.02*anchor_loss+.02*motion_loss
  if not torch.isfinite(loss):raise ValueError("Nonfinite sparse fit")
  value=float(loss.detach())
  if value<bestloss:
   bestloss=value;best=[x.detach().clone() for x in [center,yaw,q]]
  if it%100==0 or it==steps:
   row={"step":it,"loss":value,"reprojection_px":float((reproj*wt).sum().detach()/wt.sum())*max(H,W),
    "depth_robust":float(depth_loss.detach()),"motion_robust":float(motion_loss.detach())}
   hist.append(row);print(row,flush=True)
  if it==steps:break
  loss.backward();torch.nn.utils.clip_grad_norm_([center,yaw,q],2.);optimizer.step()
 with torch.no_grad():
  for par,b in zip([center,yaw,q],best):par.copy_(b)
  R,C=state();Rn=R.cpu().numpy();Cn=C.cpu().numpy()*scale;qn=q.cpu().numpy()*scale
 world=np.einsum("nij,pj->npi",Rn,qn)+Cn[:,None,:]
 cam=np.einsum("nij,npj->npi",np.linalg.inv(a["c2w"])[:,:3,:3],world)+np.linalg.inv(a["c2w"])[:,:3,3][:,None,:]
 pixp=np.einsum("nij,npj->npi",a["K"],cam);puv=pixp[...,:2]/cam[...,2:]
 residual=np.linalg.norm(puv-uv,axis=-1)
 med=np.array([np.median(residual[t,valid[t]]) if valid[t].any() else 1e6 for t in range(n)])
 result=copy.deepcopy(scene);ee=next(x for x in result["entities"] if x["id"]==e["id"])
 poses=np.tile(np.eye(4),(n,1,1));poses[:,:3,:3]=Rn;poses[:,:3,3]=Cn
 for k in ["continuous_motion","continuous_camera","persistence","state_semantics"]:result.pop(k,None)
 for obj in result["entities"]:
  for k in ["continuous_motion","exists","observation_available","observation_used","state_source","visibility_state","visible_fraction_estimate","uncertainty_proxy","evidence_summary","existence_source"]:obj.pop(k,None)
 ee["entity_to_world"]=poses.tolist();ee["renderable"]=[True]*n
 ee["sparse_evidence"]={"tracks_manifest":reference(tracks_manifest),"agent_review":reference(review_path),
  "support":support.tolist(),"median_reprojection_error_px":med.tolist(),
  "state_source":["sparse_observation_fit" if s else "underconstrained_joint_fit" for s in support],
  "shape_origin":"fixed shared proxy origin; not mask/visible-point centroid"}
 result["stage"]="sparse_evidence_fit_v2_candidate_not_validated"
 result["limitations"]+=["shared proxy shape retained, not amodal ground truth","LK identity visually reviewed, local drift remains possible",
 "no guaranteed continuous-time collision-free trajectory","unsupported intervals must be reviewed; no hidden-state certainty"]
 validate_scene(result,n);p=new_run(out);write_json(p/"scene.json",result)
 np.savez_compressed(p/"landmark_fit.npz",image_points=uv,valid=valid,predicted_uv=puv,entity_points=qn,reprojection_px=residual,quality=quality)
 report={"source_scene":reference(scene_path),"tracks_manifest":reference(tracks_manifest),"agent_review":reference(review_path),
 "camera_policy":"original VGGT fixed","shared_shape_optimized":False,"body_landmarks_refined":True,
 "history":hist,"median_reprojection_px":float(np.median(residual[valid])),
 "median_per_frame_px":med.tolist(),"supported_frames":int(support.sum()),"total_frames":n,
 "sam_missing_supported_frames":int(np.sum(support&~np.asarray(e["observed"]))),
 "review_frames":np.flatnonzero((~support)|(med>8)).tolist(),"automatic_acceptance":False}
 write_json(p/"report.json",report);return report

if __name__=="__main__":
 ap=argparse.ArgumentParser()
 for x in ["scene","tracks","review","out"]:ap.add_argument("--"+x,required=True)
 ap.add_argument("--steps",type=int,default=1000);ap.add_argument("--device",default="cpu")
 x=ap.parse_args();print(fit(x.scene,x.tracks,x.review,x.out,x.steps,x.device))
