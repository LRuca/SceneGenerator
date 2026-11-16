"""Known-geometry ablations, no later-stage supervision or sample-specific trajectories."""
import argparse,copy,json,contextlib
import numpy as np
from scipy.spatial.transform import Rotation
from scene_baseline.fixtures import create
from scene_baseline.storage import new_run,write_json,read_json,reference,verify
from scene_baseline.sparse_fit_v2 import fit as oldfit
from scene_baseline.sparse_fit_v4 import fit
from scene_baseline.geometry import render_parts,scene_parts
from scene_baseline.physical_audit import audit

VARIANTS={
 "baseline":{},
 "containment":{"containment":4.},
 "reference":{"reference":True},
 "alternating":{"alternating":True},
 "visible_depth":{"visible_depth":.15},
 "free_space":{"free_space":2.},
 "combined":{"containment":4.,"reference":True,"alternating":True,"visible_depth":.15,"free_space":2.},
 "combined_no_reference":{"containment":4.,"alternating":True,"visible_depth":.15,"free_space":2.}}

def evaluate(path,truth,a,mask):
 s=read_json(path);e=next(e for e in s["entities"] if e["id"]=="vehicle")
 target=next(e for e in truth["entities"] if e["id"]=="vehicle")
 pose=np.asarray(e["entity_to_world"]);gt=np.asarray(target["entity_to_world"])
 n,H,W=a["depth"].shape;rows=[]
 for t in range(n):
  dd,ll=render_parts(scene_parts(s,t),a["K"][t],a["c2w"][t],H,W)
  parts=scene_parts(s,t);pred=np.zeros((H,W),bool)
  for j,p in enumerate(parts):
   if p["entity_id"]=="vehicle":pred|=ll==j
  hit=mask[t]&np.isfinite(dd)
  rows.append({"iou":float((pred&mask[t]).sum()/max((pred|mask[t]).sum(),1)),
   "visible_depth_relative_mae":float(np.abs(dd[hit]-a["depth"][t][hit]).mean()/np.median(a["depth"])) if hit.any() else None})
 # Compare occupied volume against known truth, beyond point reprojection.
 rng=np.random.default_rng(10);pts=rng.uniform([-1.,-.7,-.7],[1.,.7,.7],(10000,3))
 local=pts@gt[0,:3,:3].T+gt[0,:3,3]
 def occupied(entity):
  M=np.asarray(entity["entity_to_world"])[0];q=(local-M[:3,3])@M[:3,:3];inside=np.zeros(len(q),bool)
  for p in entity["parts"]:
   m=np.asarray(p["canonical_to_entity"]);z=(q-m[:3,3])@np.linalg.inv(m[:3,:3]).T
   inside|=(np.abs(z)<=.5).all(-1)
  return inside
 x,y=occupied(e),occupied(target)
 return {"mean_position_error":float(np.linalg.norm(pose[:,:3,3]-gt[:,:3,3],axis=-1).mean()),
  "mean_rotation_error_rad":float(Rotation.from_matrix(pose[:,:3,:3]@gt[:,:3,:3].transpose(0,2,1)).magnitude().mean()),
  "volume_iou_frame0":float((x&y).sum()/max((x|y).sum(),1)),
  "mean_mask_iou":float(np.mean([r["iou"] for r in rows])),
  "mean_visible_depth_relative_mae":float(np.mean([r["visible_depth_relative_mae"] for r in rows if r["visible_depth_relative_mae"] is not None])),
  "collision_events":len(audit(s)["events"])}

def run(out,steps=400,variants=None,cases=None):
 variants=variants or VARIANTS
 root=new_run(out);fixture=create(root/"fixture",n=9,size=96)
 truth=read_json(fixture/"truth_scene.json");vehicle=next(e for e in truth["entities"] if e["id"]=="vehicle")
 with np.load(fixture/"geometry.npz") as z:a={k:z[k].copy() for k in z.files}
 with np.load(fixture/"masks.npz") as z:truthm=z["masks"][2].copy()
 pose=np.asarray(vehicle["entity_to_world"])
 q=np.array([[x,y,.3] for x in [-.40,-.15,.15,.40] for y in [-.30,0.,.30]])
 world=np.einsum("nij,pj->npi",pose[:,:3,:3],q)+pose[:,:3,3][:,None,:]
 wc=np.linalg.inv(a["c2w"]);cam=np.einsum("nij,npj->npi",wc[:,:3,:3],world)+wc[:,:3,3][:,None,:]
 pix=np.einsum("nij,npj->npi",a["K"],cam);uv=pix[...,:2]/cam[...,2:]
 results=[];equivalence=None
 base_truth=copy.deepcopy(truth)
 for case in (cases or ["offset_clean","noisy_depth_dropout","proxy_mismatch"]):
  truth=copy.deepcopy(base_truth)
  p=new_run(root/case);data={k:v.copy() for k,v in a.items()};rng=np.random.default_rng(7)
  case_masks=None
  if case=="partial_occlusion":
   wall=next(e for e in truth["entities"] if e["id"]=="wall")
   wall["parts"][0]["canonical_to_entity"]=np.diag([.45,1.1,.12,1.]).tolist()
   T=np.asarray(wall["entity_to_world"]);T[0,:3,3]=[-.4,.65,2.35];wall["entity_to_world"]=T.tolist()
   # Retain a far wall to keep all rays observed; append it after vehicle.
   far=copy.deepcopy(base_truth["entities"][1]);far["id"]="far_wall";truth["entities"].append(far)
   ds=[];ls=[]
   for t in range(9):
    dd,ll=render_parts(scene_parts(truth,t),data["K"][t],data["c2w"][t],96,96);ds.append(dd);ls.append(ll)
   data["depth"]=np.asarray(ds,dtype="float32");labels=np.asarray(ls)
   case_masks=np.stack([labels==j for j in range(4)])
   data["rgb"]=np.array([[.6,.5,.4],[.5,.5,.5],[.9,.12,.08],[.25,.4,.6]],dtype="float32")[labels]
  if case=="noisy_depth_dropout":
   data["depth"]*=1+.025*np.sin(np.arange(9)[:,None,None])+.006*rng.normal(size=data["depth"].shape)
  for t in range(9):
   for j,(x,y) in enumerate(np.rint(uv[t]).astype(int)):
    if case not in {"noisy_depth_dropout","partial_occlusion"}:data["depth"][t,y,x]=cam[t,j,2]
  np.savez_compressed(p/"geometry.npz",**data)
  gm=read_json(fixture/"geometry_manifest.json");gm["geometry"]=reference(p/"geometry.npz");np.save(p/"rgb.npy",data["rgb"]);gm["source_rgb"]=reference(p/"rgb.npy");write_json(p/"manifest.json",gm)
  hyp=read_json(fixture/"hypothesis.json");hyp["geometry_manifest_sha256"]=reference(p/"manifest.json")["sha256"]
  if case_masks is not None:
   extra=copy.deepcopy(hyp["entities"][1]);extra["id"]="far_wall";hyp["entities"].append(extra)
  write_json(p/"hypothesis.json",hyp)
  with np.load(fixture/"masks.npz") as z:masks=z["masks"].copy();ids=z["ids"].copy()
  if case_masks is not None:masks=case_masks;ids=np.array(["ground","wall","vehicle","far_wall"])
  if case=="noisy_depth_dropout":masks[2,:3]=False
  np.savez_compressed(p/"masks.npz",masks=masks,ids=ids)
  md=read_json(fixture/"masks_manifest.json");md.update(geometry_manifest=reference(p/"manifest.json"),hypothesis=reference(p/"hypothesis.json"),data=reference(p/"masks.npz"));write_json(p/"masks_manifest.json",md)
  initial=copy.deepcopy(truth);initial.update(geometry_manifest=reference(p/"manifest.json"),hypothesis=reference(p/"hypothesis.json"),masks_manifest=reference(p/"masks_manifest.json"))
  e=next(e for e in initial["entities"] if e["id"]=="vehicle")
  pp=np.asarray(e["entity_to_world"]);pp[:,:3,3]+=[.18,0.,-.20]
  e["entity_to_world"]=pp.tolist()
  if case=="proxy_mismatch":
   mm=np.asarray(e["parts"][0]["canonical_to_entity"]);mm[:3,:3]*=.7;e["parts"][0]["canonical_to_entity"]=mm.tolist()
  write_json(p/"initial_scene.json",initial)
  valid=np.ones(uv.shape[:2],bool)
  if case=="noisy_depth_dropout":valid[:3,:6]=False
  if case=="partial_occlusion":
   xy=np.rint(uv).astype(int)
   for t in range(9):valid[t]=masks[2,t,xy[t,:,1],xy[t,:,0]]
  obs=uv.copy()
  if case=="noisy_depth_dropout":obs+=rng.normal(0,.3,obs.shape)
  if case=="track_outliers":obs[1:8:2,:3]+=np.array([10.,-8.])
  anchors=np.array([np.flatnonzero(valid[:,j])[len(np.flatnonzero(valid[:,j]))//2] if valid[:,j].any() else 4 for j in range(len(q))])
  np.savez_compressed(p/"tracks.npz",uv=obs,valid=valid,anchor_frame=anchors,fb_error=np.zeros(valid.shape),photo_error=np.zeros(valid.shape))
  tm={"geometry_manifest":reference(p/"manifest.json"),"entity_id":"vehicle","data":reference(p/"tracks.npz")};write_json(p/"tracks_manifest.json",tm)
  review={"actor":"synthetic_ground_truth_correspondence","decision":"accept_candidate_evidence","tracks_manifest":reference(p/"tracks_manifest.json")};write_json(p/"review.json",review)
  for variant,opts in variants.items():
   with (p/(variant+".log")).open("x") as log,contextlib.redirect_stdout(log):
    rep=fit(p/"initial_scene.json",p/"tracks_manifest.json",p/"review.json",p/variant,steps=steps,options=opts)
   row={"case":case,"variant":variant,"options":opts,"fit":reference(p/variant/"report.json"),
     "reprojection_px":rep["median_reprojection_px"],"outside_proxy_fraction":rep["outside_proxy_fraction"],
     **evaluate(p/variant/"scene.json",truth,data,masks[2])}
   results.append(row);print(json.dumps(row),flush=True)
   if case=="offset_clean" and variant=="baseline":
    with (p/"old_equivalence.log").open("x") as log,contextlib.redirect_stdout(log):
     oldfit(p/"initial_scene.json",p/"tracks_manifest.json",p/"review.json",p/"old_equivalence",steps=steps)
    old=read_json(p/"old_equivalence/scene.json");new=read_json(p/variant/"scene.json")
    oo=next(e for e in old["entities"] if e["id"]=="vehicle");nn=next(e for e in new["entities"] if e["id"]=="vehicle")
    equivalence=float(np.max(np.abs(np.array(oo["entity_to_world"])-nn["entity_to_world"])))
    if equivalence>1e-6:raise AssertionError("Disabled V4 differs from V2")
  write_json(p/"complete.json",{"status":"complete","variants":list(variants)})
 write_json(root/"report.json",{"steps_per_variant":steps,"baseline_equivalence_max_pose_difference":equivalence,
  "rows":results,"scope":"controlled development; proxy mismatch is an intentional non-recoverable fixed-shape case",
  "acceptance":"No automatic best-config choice from training loss. Inspect volume, silhouette and motion tradeoffs."})
 return results
if __name__=="__main__":
 ap=argparse.ArgumentParser();ap.add_argument("--out",required=True);ap.add_argument("--steps",type=int,default=400);ap.add_argument("--config");ap.add_argument("--cases",nargs="+");a=ap.parse_args();run(a.out,a.steps,read_json(a.config) if a.config else None,a.cases)
