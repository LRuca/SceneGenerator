"""Explicit agent-chosen room topology, SAM person, measured wall-floor boundary."""
import argparse,copy
import numpy as np
from scene_baseline.storage import BASE,new_run,read_json,write_json,verify,reference
from scene_baseline.contracts import load_bundle
from scene_baseline.scene import initialize,points_for_mask,validate_scene
from scene_baseline.cage_v4 import make_state
from scene_baseline.modeling_policy_v4 import apply
def run(stage):
 root=new_run(BASE/f"runs/v4_person_s{stage}_prepare_r1")
 manifest=BASE/f"runs/v4_person_s{stage}_vggt_r2/manifest.json"
 hyp=BASE/f"runs/v4_person_s{stage}_agent_r1/hypothesis.json"
 m,a=load_bundle(manifest);sm=read_json(BASE/f"runs/v4_person_s{stage}_sam_r1/manifest.json")
 with np.load(verify(sm["data"])) as z:
  if len(z["ids"])!=1:raise ValueError("Person identity requires agent review")
  person=z["masks"][0]
 n,H,W=person.shape;walls=[];floors=[];lines=[]
 # Agent identified single rear wall/floor topology. Locate appearance boundary from unoccluded side strips.
 for t in range(n):
  gray=a["rgb"][t].mean(-1);edge=np.abs(np.diff(gray,axis=0))
  valid=~person[t,1:]&~person[t,:-1]
  side=np.zeros((H-1,W),bool);side[:,:W//5]=True;side[:,-W//5:]=True
  score=np.where(valid&side,edge,0).sum(1)/np.maximum((valid&side).sum(1),1)
  lo,hi=int(.55*H),int(.86*H);line=lo+int(np.argmax(score[lo:hi]))+1
  lines.append(line);yy=np.arange(H)[:,None]
  floors.append(np.broadcast_to(yy>line+2,(H,W))&~person[t])
  walls.append(np.broadcast_to(yy<line-2,(H,W))&~person[t])
 masks=np.stack([person,np.array(floors),np.array(walls)])
 np.savez_compressed(root/"masks.npz",masks=masks,ids=np.array(["person","ground","wall"]))
 write_json(root/"masks_manifest.json",{"schema":"scene_baseline.masks.v1","geometry_manifest":reference(manifest),"hypothesis":reference(hyp),"data":reference(root/"masks.npz"),
  "sources":{"person":reference(BASE/f"runs/v4_person_s{stage}_sam_r1/manifest.json"),"background":"agent-selected simple wall/floor topology; strongest side-strip boundary; exclude person and boundary band"},"floor_wall_boundary_rows":lines})
 scene=initialize(manifest,hyp,root/"masks_manifest.json",root/"initial_rigid")
 e=next(e for e in scene["entities"] if e["id"]=="person")
 up=np.array(scene["ground_normal"]);right=a["c2w"][0,:3,0].copy();right-=up*(up@right);right/=np.linalg.norm(right)
 basis=np.stack([right,up,np.cross(right,up)],1)
 pts,bytime=points_for_mask(a,person)
 height=float(np.quantile(pts@up,.98)-np.quantile(pts@up,.02))
 size=np.array([height*.28,height,height*.20])
 poses=np.asarray(e["entity_to_world"]);poses[:,:3,:3]=basis
 # Initialize from observed point extents, with an explicit hidden-thickness prior; not certified object origins.
 for t,p in enumerate(bytime):
  if not len(p):continue
  q=p@basis;center=(np.quantile(q,.02,axis=0)+np.quantile(q,.98,axis=0))/2
  poses[t,:3,3]=center@basis.T+a["c2w"][t,:3,2]*size[2]*.5
 e.update(motion="deformable",entity_to_world=poses.tolist(),renderable=[True]*n)
 e["deformation"]=make_state(size,np.zeros((n,5,2)),["upright_human_coarse_prior"]*n)
 e["initialization"]="explicit upright tube with human aspect ratio prior; observed extent and hidden-depth correction; not true center"
 scene["stage"]="agent_explicit_coarse_cage_conversion";validate_scene(scene,n);write_json(root/"cage_scene.json",scene)
 decisions=[]
 for x in scene["entities"]:
  eid=x["id"];is_person=eid=="person"
  decisions.append(dict(id=eid,narrative_role="key_actor" if is_person else "incidental",geometric_duties=[] if is_person else ["support","contact"] if eid=="ground" else ["layout"],
   representation="deformable_cage" if is_person else "thin_surface",motion=x["motion"],confidence=.85,part_budget=1,
   identity_status="single adult reviewed in same-stage RGB" if is_person else "agent simple room topology",
   boundary_status="SAM candidate" if is_person else "measured appearance boundary, uncertain behind person",
   evidence=[{"frames":[0,6,12,18,24],"source":m["source_rgb"],"finding":"one adult bends then straightens; floor and rear wall"}]))
 policy={"schema":"scene_baseline.modeling_policy.v4","actor":"current_chat_codex","source_scene":reference(root/"cage_scene.json"),"entities":decisions,"relations":[dict(subject="person",object="ground",relation="supported_by",evidence=["feet at visible floor throughout"])],"revision_proposals":[]}
 write_json(root/"policy.json",policy);apply(root/"cage_scene.json",root/"policy.json",root/"prioritized",total_steps=1200)
 return root
if __name__=="__main__":
 ap=argparse.ArgumentParser();ap.add_argument("--stage",type=int,required=True);run(ap.parse_args().stage)
