"""Fixed numerical pipeline. Semantic plan is locked before any fitting; no auto-retuning."""
import argparse,json,numpy as np,copy
from scipy.spatial.transform import Rotation,Slerp
from pathlib import Path
from scene_baseline.storage import read_json,write_json,reference,verify,new_run
from scene_baseline.scene import initialize
from scene_baseline.masks import import_masks
from scene_baseline.fit_scene import fit
from scene_baseline.persistence import apply,validate_policy
from scene_baseline.delivery import render
from scene_baseline.persistence_view import export
from scene_baseline.physical_audit import audit
from scene_baseline.contracts import load_bundle
def run(plan_path,policy_path,out,freeze_path):
 freeze=read_json(freeze_path)
 for ref in freeze["sources"]:verify(ref)
 assert freeze["policy"]==reference(policy_path)
 plan=read_json(plan_path);policy=validate_policy(read_json(policy_path))
 p=new_run(out);m,a=load_bundle(verify(plan["geometry_manifest"]))
 hyp={"schema":"scene_baseline.hypothesis.v1","actor":plan["actor"],"geometry_manifest_sha256":plan["geometry_manifest"]["sha256"],
 "entities":[],"relations":plan["relations"],"tool_requests":[],"planning_source":reference(plan_path)}
 masks=[]
 for e in plan["entities"]:
  source=read_json(verify(e["sam_manifest"]))
  assert source["geometry_manifest"]==plan["geometry_manifest"]
  with np.load(verify(source["data"]),allow_pickle=False) as z:
   index=z["ids"].tolist().index(e["sam_id"]);mask=z["masks"][index]
  if np.count_nonzero(mask.any((1,2)))<policy["minimum_observations"]:raise ValueError("Insufficient SAM temporal support "+e["id"])
  masks.append(mask)
  hyp["entities"].append({"id":e["id"],"label":e["label"],"motion":e["motion"],
   "segmentation_prompt":e["prompt"],"confidence":.8,
   "parts":[{"id":e["id"]+"_proxy","kind":"plane" if e["id"]==plan["ground_entity"] else "box","role":"ground" if e["id"]==plan["ground_entity"] else "coarse whole entity"}]})
 write_json(p/"hypothesis.json",hyp);masks=np.stack(masks);ids=np.array([e["id"] for e in plan["entities"]])
 np.savez_compressed(p/"masks.npz",masks=masks,ids=ids)
 manifest=verify(plan["geometry_manifest"])
 import_masks(manifest,p/"hypothesis.json",p/"masks.npz",p/"masks")
 initialize(manifest,p/"hypothesis.json",p/"masks/manifest.json",p/"init")
 fit(p/"init/scene.json",p/"fit",policy["fit_steps"],policy["fit_resolution"],"cuda",policy["fit_lr"])
 apply(p/"fit/scene.json",policy_path,p/"persistent")
 render(p/"fit/scene.json",p/"raw_delivery",width=policy["render_width"])
 render(p/"persistent/scene.json",p/"persistent_delivery",width=policy["render_width"])
 export(p/"fit/scene.json",p/"persistent/scene.json",p/"view",width=policy["render_width"])
 write_json(p/"physical_conflicts.json",audit(read_json(p/"persistent/scene.json")))
 # Genuine withheld downstream masks: initialize+fit again from training masks, not full fitted poses.
 held=masks.copy();heldframes={}
 for i,e in enumerate(plan["entities"]):
  if e["motion"]!="rigid":continue
  visible=np.flatnonzero(masks[i].any((1,2)));start=int(len(visible)*policy["holdout_start_fraction"])
  count=max(1,int(len(visible)*policy["holdout_fraction"]));frames=visible[start:start+count]
  held[i,frames]=False;heldframes[e["id"]]=frames.tolist()
 np.savez_compressed(p/"held_masks.npz",masks=held,ids=ids)
 import_masks(manifest,p/"hypothesis.json",p/"held_masks.npz",p/"held_masks")
 initialize(manifest,p/"hypothesis.json",p/"held_masks/manifest.json",p/"held_init")
 fit(p/"held_init/scene.json",p/"held_fit",policy["fit_steps"],policy["fit_resolution"],"cuda",policy["fit_lr"])
 apply(p/"held_fit/scene.json",policy_path,p/"held_persistent")
 linear=read_json(p/"held_fit/scene.json")
 for i,e in enumerate(linear["entities"]):
  if e["motion"]!="rigid":continue
  good=np.flatnonzero(held[i].any((1,2)));po=np.array(e["entity_to_world"]);t=a["times_s"];source=po[good].copy()
  for k in range(3):po[:,k,3]=np.interp(t,t[good],source[:,k,3])
  po[:,:3,:3]=Slerp(t[good],Rotation.from_matrix(source[:,:3,:3]))(np.clip(t,t[good][0],t[good][-1])).as_matrix()
  e["entity_to_world"]=po.tolist();e["renderable"]=[True]*len(t)
 write_json(p/"held_linear_scene.json",linear)
 render(p/"held_linear_scene.json",p/"held_linear_delivery",width=policy["render_width"],evaluation_masks_manifest=p/"masks/manifest.json")
 render(p/"held_fit/scene.json",p/"held_raw_delivery",width=policy["render_width"],evaluation_masks_manifest=p/"masks/manifest.json")
 render(p/"held_persistent/scene.json",p/"held_delivery",width=policy["render_width"],evaluation_masks_manifest=p/"masks/manifest.json")
 rawmetrics=read_json(p/"raw_delivery/metrics.json");newmetrics=read_json(p/"persistent_delivery/metrics.json")
 hr=read_json(p/"held_raw_delivery/metrics.json");hn=read_json(p/"held_delivery/metrics.json");hl=read_json(p/"held_linear_delivery/metrics.json")
 def maskscore(m,e,frames):
  v=[m["frames"][t]["entities"][e]["visible_mask_iou"] for t in frames]
  return float(np.mean([x for x in v if x is not None])) if any(x is not None for x in v) else None
 summary={}
 for e in plan["entities"]:
  if e["motion"]!="rigid":continue
  i=ids.tolist().index(e["id"]);visible=np.flatnonzero(masks[i].any((1,2))).tolist()
  summary[e["id"]]={"visible_frames":visible,"observed_iou_raw":maskscore(rawmetrics,e["id"],visible),
   "observed_iou_persistent":maskscore(newmetrics,e["id"],visible),
   "withheld_frames":heldframes[e["id"]],"withheld_iou_observation_gated":maskscore(hr,e["id"],heldframes[e["id"]]),
   "withheld_iou_persistent":maskscore(hn,e["id"],heldframes[e["id"]]),"withheld_iou_linear_baseline":maskscore(hl,e["id"],heldframes[e["id"]])}
 write_json(p/"case_report.json",{"plan":reference(plan_path),"freeze":reference(freeze_path),"policy":reference(policy_path),
 "status":"frozen_pipeline_completed","entity_metrics":summary,"physical":reference(p/"physical_conflicts.json"),
 "holdout_scope":"downstream dynamic masks withheld before initialization and geometry fitting; VGGT RGB/depth/camera still conditioned on full video; not end-to-end visual holdout",
 "no_instance_feedback_retuning":True,"assess_with":"persistence/report.json + raw/persistent delivery + physical_conflicts.json"})
 for ref in freeze["sources"]:verify(ref)
 return str(p/"case_report.json")
if __name__=="__main__":
 ap=argparse.ArgumentParser();ap.add_argument("--plan",required=True);ap.add_argument("--policy",required=True);ap.add_argument("--out",required=True);ap.add_argument("--freeze",required=True)
 args=ap.parse_args();print(run(args.plan,args.policy,args.out,args.freeze))
