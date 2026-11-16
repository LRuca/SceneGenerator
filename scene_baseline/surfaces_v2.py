"""Fit reviewed visible vertical surfaces without filling unknown depth extent."""
import argparse,copy
import numpy as np
from .scene import load_scene
from .contracts import load_bundle
from .storage import new_run,read_json,write_json,verify,reference
from .masks import import_masks

def vertical_surface(a,mask,up,thickness_ratio=.015):
 points=[]
 for t in range(0,len(mask),3):
  import cv2
  mm=cv2.erode(mask[t].astype("uint8"),np.ones((5,5),np.uint8)).astype(bool)
  v,u=np.nonzero(mm)
  if len(u)<24:continue
  idx=np.arange(0,len(u),max(1,len(u)//800));v,u=v[idx],u[idx]
  conf=a["confidence"][t,v,u];good=conf>=np.quantile(conf,.35)
  v,u=v[good],u[good]
  ray=np.column_stack([u,v,np.ones(len(u))])@np.linalg.inv(a["K"][t]).T
  xyz=ray*a["depth"][t,v,u,None]
  points.append(xyz@a["c2w"][t,:3,:3].T+a["c2w"][t,:3,3])
 if not points:raise ValueError("No reliable surface evidence")
 pts=np.concatenate(points);up=up/np.linalg.norm(up)
 right=a["c2w"][0,:3,0].copy();right-=up*(right@up);right/=np.linalg.norm(right)
 flat_basis=np.stack([right,np.cross(right,up)],1)
 flat=pts@flat_basis
 keep=np.ones(len(pts),bool)
 for _ in range(5):
  q=flat[keep];mid=np.median(q,axis=0)
  _,vec=np.linalg.eigh(np.cov((q-mid).T));normal2=vec[:,0]
  resid=np.abs((flat-mid)@normal2)
  keep=resid<=np.quantile(resid,.8)
 tangent=flat_basis@vec[:,1];normal=np.cross(tangent,up)
 center=np.median(pts,axis=0)
 if normal@(a["c2w"][:,:3,3].mean(0)-center)<0:tangent=-tangent;normal=-normal
 R=np.stack([tangent,up,normal],1);q=pts@R
 lo,hi=np.percentile(q[:,:2],[1,99],axis=0);width,height=hi-lo
 thick=max(width*thickness_ratio,float(np.median(a["depth"])*.001))
 front=float(np.median(q[:,2]));local_center=np.array([*((lo+hi)/2),front-thick/2])
 pose=np.eye(4);pose[:3,:3]=R;pose[:3,3]=R@local_center
 matrix=np.eye(4);matrix[:3,:3]=np.diag([width,height,thick])
 return pose,matrix,{"point_count":len(pts),"normal_residual_median":float(np.median(np.abs(q[:,2]-front))),
  "width":float(width),"height":float(height),"thickness":float(thick),
  "thickness_source":"explicit thin-surface prior; hidden backside unobserved",
  "thickness_to_width_prior":thickness_ratio}

def prepare(scene_path,plan_path,out):
 scene,m,a=load_scene(scene_path);plan=read_json(plan_path);p=new_run(out)
 if plan["geometry_manifest"]!=scene["geometry_manifest"]:raise ValueError("Cross-stage plan")
 hypo=read_json(verify(scene["hypothesis"]));hypo=copy.deepcopy(hypo)
 hypo["actor"]="current-chat Codex, v2 semantic surface review"
 md=read_json(verify(scene["masks_manifest"]))
 with np.load(verify(md["data"])) as z:masks=list(z["masks"]);ids=z["ids"].tolist()
 result=copy.deepcopy(scene);reports={}
 for spec in plan["surfaces"]:
  eid=spec["entity_id"]
  if spec.get("mask_manifest"):
   sm=read_json(verify(spec["mask_manifest"]))
   if sm["geometry_manifest"]!=scene["geometry_manifest"]:raise ValueError("Cross-stage surface masks")
   with np.load(verify(sm["data"])) as z:mask=z["masks"][z["ids"].tolist().index(spec["mask_id"])]
   if eid in ids:masks[ids.index(eid)]=mask
   else:ids.append(eid);masks.append(mask)
  else:mask=masks[ids.index(eid)]
  pose,matrix,report=vertical_surface(a,mask,np.asarray(scene["ground_normal"]),spec["thickness_to_width_prior"])
  part={"id":eid+"_visible_surface","kind":"box","role":"reviewed vertical surface","canonical_to_entity":matrix.tolist()}
  ent=next((e for e in result["entities"] if e["id"]==eid),None)
  if ent is None:
   ent={"id":eid,"label":spec["label"],"motion":"static","observed":mask.any((1,2)).tolist(),"renderable":[True]*len(mask)}
   result["entities"].append(ent)
   hypo["entities"].append({"id":eid,"label":spec["label"],"motion":"static","segmentation_prompt":spec["segmentation_prompt"],
    "confidence":.7,"parts":[{"id":part["id"],"kind":"box","role":part["role"]}]})
  ent["entity_to_world"]=[pose.tolist()];ent["parts"]=[part];ent["surface_prior"]=report
  reports[eid]=report
 for obj in result["entities"]:
  for k in ["continuous_motion","exists","observation_available","observation_used","state_source","visibility_state","visible_fraction_estimate","uncertainty_proxy","evidence_summary","existence_source"]:obj.pop(k,None)
 for k in ["continuous_camera","persistence","state_semantics"]:result.pop(k,None)
 hypo["semantic_surface_plan"]=reference(plan_path)
 write_json(p/"hypothesis.json",hypo)
 np.savez_compressed(p/"masks.npz",ids=np.array(ids),masks=np.stack(masks))
 import_masks(verify(scene["geometry_manifest"]),p/"hypothesis.json",p/"masks.npz",p/"masks")
 result["hypothesis"]=reference(p/"hypothesis.json");result["masks_manifest"]=reference(p/"masks/manifest.json")
 result["surface_plan"]=reference(plan_path);result["stage"]="reviewed_surface_candidate_v2"
 write_json(p/"scene.json",result);write_json(p/"report.json",reports)
 return reports

if __name__=="__main__":
 ap=argparse.ArgumentParser()
 for x in ["scene","plan","out"]:ap.add_argument("--"+x,required=True)
 a=ap.parse_args();print(prepare(a.scene,a.plan,a.out))
