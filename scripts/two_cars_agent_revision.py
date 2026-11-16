"""Agent round1: test pickup composition after one-box source-view diagnosis."""
import numpy as np
from pathlib import Path
from scene_baseline.storage import read_json,write_json,reference
from scene_baseline.revision import apply
for st in [20,50]:
 parent=f"runs/two_cars_s{st}_fit_r0/scene.json"
 scene=read_json(parent);initial=read_json(f"runs/two_cars_s{st}_init/scene.json")
 actions=[];rationale=[]
 for eid in ["near_truck","far_truck"]:
  e=next(x for x in scene["entities"] if x["id"]==eid)
  e0=next(x for x in initial["entities"] if x["id"]==eid)
  size=np.linalg.norm(np.array(e["parts"][0]["canonical_to_entity"])[:3,:3],axis=0)
  # Round0 collapsed height to explain lower body; restore data-initialized full visible height
  # before proposing cabin and wheels. No metric-size or cross-stage source used.
  size[1]=np.linalg.norm(np.array(e0["parts"][0]["canonical_to_entity"])[:3,1])
  L,H,D=size;parts=[]
  def add(name,kind,center,linear,role):
   mat=np.eye(4);mat[:3,:3]=linear;mat[:3,3]=center
   parts.append({"id":eid+"_"+name,"kind":kind,"role":role,"canonical_to_entity":mat.tolist()})
  add("body","box",[0,-.08*H,0],np.diag([L,.46*H,.85*D]),"lower pickup body")
  add("cabin","box",[0,.20*H,0],np.diag([.44*L,.60*H,.78*D]),"central cabin; front-back offset uncertain")
  rot=np.array([[1.,0,0],[0,0,-1],[0,1,0]])
  for ix,x in enumerate([-.32*L,.32*L]):
   for iz,z in enumerate([-.46*D,.46*D]):
    add(f"wheel{ix}{iz}","cylinder",[x,-.32*H,z],rot@np.diag([.36*H,.18*D,.36*H]),"tire proxy; hidden-side wheel inferred by symmetric prior")
  actions.append({"type":"replace_parts","entity_id":eid,"parts":parts})
  rationale.append({"entity":eid,"extent_before_shape_optimization":size.tolist(),
     "inference":"body/cabin/wheel topology from RGB semantics; symmetric hidden-side wheels are a prior, not observed geometry"})
 request={"schema":"scene_baseline.revision.v1","parent_scene":reference(parent),
  "actor":"Codex agent; visual inspection of round0 frames0,12,32 RGB/white plus per-instance IoU",
  "reason":"Round0 boxes preserve two identities and depth order but miss cabin and tire silhouette; test shared multipart pickup geometry",
  "actions":actions,"assumptions":rationale,"camera_change":False,
  "acceptance":"Review separate visible-frame IoU for both trucks and source/global appearance; reject regressions rather than silently replace round0",
  "remaining_unknowns":["front-back cabin offset","near truck clipped extent","hidden trajectories","true camera accuracy"]}
 out=Path(f"runs/two_cars_s{st}_revision1_request");out.mkdir();write_json(out/"request.json",request)
 apply(parent,out/"request.json",f"runs/two_cars_s{st}_revised1")
 print(st,"multipart revision authored",flush=True)
