"""Prepare/resume orchestration. Handoffs target the current-chat agent, not the user."""
import argparse
import numpy as np
from .storage import read_json,write_json,new_run,reference,verify
from .tracking_v2 import run as track
from .surfaces_v2 import prepare as surfaces
from .sparse_fit_v2 import fit as sparse_fit
from .agent_recovery_v2 import supplement
from .scene import load_scene
from .recovery_view_v2 import export

def prepare(config_path,out):
 c=read_json(config_path);p=new_run(out)
 geometry=verify(c["geometry"]);scene=verify(c["initial_scene"])
 s,_,_=load_scene(scene)
 if s["geometry_manifest"]!=c["geometry"]:raise ValueError("Scene and stage mismatch")
 if c.get("reuse_tracks"):
  trpath=verify(c["reuse_tracks"]);tr=read_json(trpath)
  if tr["geometry_manifest"]!=c["geometry"]:raise ValueError("Reused tracks from another stage")
 else:
  track(geometry,verify(s["masks_manifest"]),c["entity_id"],p/"tracks")
  trpath=p/"tracks/manifest.json"
 state={"schema":"scene_baseline.recovery_prepared.v2","config":reference(config_path),"tracks_manifest":reference(trpath),
 "agent_request":{"actor":"current_chat_codex","action":"Review candidate identity; accept, reject or supplement observations",
 "automatic_tool_api":False,"status":"pending_current_chat_agent"}}
 write_json(p/"prepared.json",state);return state

def resume(prepared_path,review_path,out):
 prepared=read_json(prepared_path);c=read_json(verify(prepared["config"]));r=read_json(review_path)
 if r["tracks_manifest"]!=prepared["tracks_manifest"]:raise ValueError("Review is not for prepared tracks")
 p=new_run(out);track_path=verify(prepared["tracks_manifest"]);geometry=verify(c["geometry"])
 if r["decision"]=="supplement_observations":
  supplement(geometry,track_path,review_path,p/"supplement")
  track_path=p/"supplement/manifest.json"
  accept={"actor":"current_chat_codex","decision":"accept_candidate_evidence","tracks_manifest":reference(track_path),
   "derived_from_explicit_supplement_review":reference(review_path),"scope":"fit candidate observations, not final acceptance"}
  write_json(p/"supplement_fit_review.json",accept);review_path=p/"supplement_fit_review.json"
 elif r["decision"]!="accept_candidate_evidence":raise ValueError("Agent did not authorize candidate-evidence fitting")
 source=verify(c["initial_scene"])
 if c.get("surface_plan"):
  surfaces(source,verify(c["surface_plan"]),p/"surfaces");source=p/"surfaces/scene.json"
 sparse_fit(source,track_path,review_path,p/"sparse_fit",steps=c.get("sparse_steps",1200),device=c.get("device","cpu"))
 final=p/"sparse_fit/scene.json"
 if c.get("joint_fit",False):
  from .fit_scene_v2 import fit as joint_fit
  s=read_json(final);lm=np.load(p/"sparse_fit/landmark_fit.npz")
  ent=next(e for e in s["entities"] if e["id"]==c["entity_id"])
  ent["landmarks"]={"entity_points":lm["entity_points"].tolist(),"image_points":lm["image_points"].tolist(),
   "valid":lm["valid"].tolist(),"weight":8.}
  for e in s["entities"]:
   e["fit_constraints"]={"lock_shape":True}
   if e["motion"]=="static":e["fit_constraints"]["lock_yaw"]=True
  s["relations"]+=c.get("extra_relations",[])
  write_json(p/"joint_input.json",s)
  joint_fit(p/"joint_input.json",p/"joint_fit",steps=c.get("joint_steps",300),resolution=72,device=c.get("device","cpu"),
   lr=.003,entity_ids=c.get("optimize_entities",[c["entity_id"]]))
  final=p/"joint_fit/scene.json"
 report=export(verify(c["comparison_scene"]),final,p/"sparse_fit/landmark_fit.npz",p/"view",width=c.get("render_width",300))
 queue={"actor":"current_chat_codex","status":"pending_render_review","final_scene":reference(final),
 "view_report":reference(p/"view/report.json"),"coarse_collision_events":len(report["collision_after"]["events"]),
 "median_sparse_residual_px":report["sparse_residual_median_px"],
 "automatic_acceptance":False,"note":"Inspect source and global views. Good tracked-point fit does not prove full geometry."}
 write_json(p/"validation_queue.json",queue);return queue

if __name__=="__main__":
 ap=argparse.ArgumentParser();sp=ap.add_subparsers(dest="mode",required=True)
 pr=sp.add_parser("prepare");pr.add_argument("--config",required=True);pr.add_argument("--out",required=True)
 re=sp.add_parser("resume");re.add_argument("--prepared",required=True);re.add_argument("--review",required=True);re.add_argument("--out",required=True)
 a=ap.parse_args()
 print(prepare(a.config,a.out) if a.mode=="prepare" else resume(a.prepared,a.review,a.out))
