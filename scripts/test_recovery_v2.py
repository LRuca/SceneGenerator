"""Behavioral checks for recovery failure modes, using known synthetic evidence."""
from pathlib import Path
import json,copy
import numpy as np
import cv2
from scene_baseline.storage import new_run,reference,write_json,read_json,verify
from scene_baseline.tracking_v2 import track_points,run
from scene_baseline.agent_recovery_v2 import supplement
from scene_baseline.fixtures import create
from scene_baseline.sparse_fit_v2 import fit
from scene_baseline.scene import load_scene
from scene_baseline.surfaces_v2 import vertical_surface

import sys
root=new_run(sys.argv[1] if len(sys.argv)>1 else "runs/pv2_recovery_component_tests")
checks=[]
def check(name,condition,details=None):
 if not condition:raise AssertionError(name+str(details))
 checks.append({"name":name,"passed":True,"details":details})
# Known translation: masked-out early frames do not stop optical flow.
rng=np.random.default_rng(7)
base=np.zeros((128,128),np.uint8);base[35:90,30:85]=rng.integers(30,255,(55,55),dtype=np.uint8)
gray=[cv2.warpAffine(base,np.array([[1,0,t*2],[0,1,t]],np.float32),(128,128)) for t in range(8)]
anchor=7
mask=np.zeros((8,128,128),bool);mask[4:]=True
seed=cv2.goodFeaturesToTrack(gray[anchor],maxCorners=16,qualityLevel=.05,minDistance=6)[:,0]
u,v,fb,photo=track_points(gray,anchor,seed,mask)
truth=seed[None]+np.array([[2*(t-anchor),t-anchor] for t in range(8)])[:,None]
err=np.linalg.norm(u-truth,axis=-1)
check("LK recovers known translation despite empty early masks",v[0].sum()>=10 and np.median(err[v])<.15,
 {"valid_at_first":int(v[0].sum()),"median_error_px":float(np.median(err[v]))})
# Blank imagery produces an explicit review artifact.
fixture=create(root/"fixture",n=5,size=96)
md=read_json(fixture/"masks_manifest.json")
blank=np.zeros((3,5,96,96),bool)
np.savez_compressed(root/"blank_masks.npz",masks=blank,ids=np.array(["ground","wall","vehicle"]))
md["data"]=reference(root/"blank_masks.npz");write_json(root/"blank_masks_manifest.json",md)
r=run(fixture/"geometry_manifest.json",root/"blank_masks_manifest.json","vehicle",root/"zero_track_run")
req=read_json(root/"zero_track_run/review_request.json")
check("Zero-track output requests agent review instead of inventing state",r["point_count"]==0 and req["triggers"]["insufficient_track_support"]==[[0,4]])
# A guessed point is never converted into a measured observation.
review={"actor":"current_chat_codex","decision":"supplement_observations","tracks_manifest":reference(root/"zero_track_run/manifest.json"),
 "test_source":"synthetic schema/measurement-separation test, not actual visual annotation",
 "add_points":[{"label":"body_corner","anchor_frame":0,"propagate":False,"observations":[
 {"frame":0,"uv":[40,50],"evidence":"visual_annotation"},
 {"frame":2,"uv":[42,51],"evidence":"inferred_hypothesis"},
 {"frame":4,"uv":[44,52],"evidence":"visual_annotation"}]}]}
write_json(root/"supplement_review.json",review)
s=supplement(fixture/"geometry_manifest.json",root/"zero_track_run/manifest.json",root/"supplement_review.json",root/"supplement")
with np.load(verify(s["data"])) as z:
 check("Manual visual observations and inferred guesses remain distinct",z["valid"][:,0].tolist()==[True,False,False,False,True] and z["source_kind"][0,0]==2)
bad=copy.deepcopy(review);bad["tracks_manifest"]["sha256"]="bad"
write_json(root/"bad_review.json",bad)
try:supplement(fixture/"geometry_manifest.json",root/"zero_track_run/manifest.json",root/"bad_review.json",root/"invalid")
except ValueError:checks.append({"name":"Cross-artifact review rejected","passed":True})
else:raise AssertionError("Accepted mismatched review")
# A second review must not relabel previous manual observations as automatic.
review2={"actor":"current_chat_codex","decision":"supplement_observations",
 "tracks_manifest":reference(root/"supplement/manifest.json"),"add_points":[],"reject_points":[]}
write_json(root/"second_review.json",review2)
s2=supplement(fixture/"geometry_manifest.json",root/"supplement/manifest.json",root/"second_review.json",root/"second_supplement")
with np.load(verify(s2["data"])) as z:
 check("Repeated agent reviews preserve manual provenance",z["source_kind"][0,0]==2 and z["source_kind"][4,0]==2)
# Known 3D body landmarks with a camera moving: sparse fit must fix a perturbed trajectory even with no SAM masks.
scene,m,a=load_scene(fixture/"truth_scene.json")
e=next(x for x in scene["entities"] if x["id"]=="vehicle");pose=np.array(e["entity_to_world"])
q=np.array([[-.4,-.3,.31],[-.3,.3,.31],[.3,-.3,.31],[.4,.3,.31],[-.2,0,.31],[.2,0,.31],[0,.2,.31],[0,-.2,.31]])
world=np.einsum("nij,pj->npi",pose[:,:3,:3],q)+pose[:,:3,3][:,None,:]
wc=np.linalg.inv(a["c2w"]);cam=np.einsum("nij,npj->npi",wc[:,:3,:3],world)+wc[:,:3,3][:,None,:]
pix=np.einsum("nij,npj->npi",a["K"],cam);uv=pix[...,:2]/cam[...,2:]
# Place landmark-consistent depth/confidence at sampled points, with exact fixture truth elsewhere.
gg={k:a[k].copy() for k in a}
for t in range(5):
 for j in range(len(q)):
  x,y=np.rint(uv[t,j]).astype(int);gg["depth"][t,y,x]=cam[t,j,2]
np.savez_compressed(root/"exact_geometry.npz",**gg)
gm=read_json(fixture/"geometry_manifest.json");gm["geometry"]=reference(root/"exact_geometry.npz")
write_json(root/"exact_manifest.json",gm)
md2=copy.deepcopy(md);md2["geometry_manifest"]=reference(root/"exact_manifest.json")
write_json(root/"exact_masks_manifest.json",md2)
scene["geometry_manifest"]=reference(root/"exact_manifest.json");scene["masks_manifest"]=reference(root/"exact_masks_manifest.json")
# Keep reference anchor exact; perturb other times to test real recovery.
pe=np.array(e["entity_to_world"]);pe[1:,:3,3]+=[.12,0,.1];e["entity_to_world"]=pe.tolist();e["observed"]=[False]*5
write_json(root/"sparse_initial.json",scene)
valid=np.ones((5,len(q)),bool);valid[2,0]=False
bad_photo=np.zeros_like(valid,float);bad_photo[2,0]=-1e30
np.savez_compressed(root/"known_tracks.npz",uv=uv.astype("float32"),valid=valid,
 fb_error=np.zeros_like(valid,float),photo_error=bad_photo,anchor_frame=np.zeros(len(q),int))
tm={"geometry_manifest":reference(root/"exact_manifest.json"),"entity_id":"vehicle","data":reference(root/"known_tracks.npz")}
write_json(root/"known_tracks_manifest.json",tm)
write_json(root/"known_tracks_review.json",{"tracks_manifest":reference(root/"known_tracks_manifest.json"),"decision":"accept_candidate_evidence","actor":"synthetic_ground_truth_test"})
rep=fit(root/"sparse_initial.json",root/"known_tracks_manifest.json",root/"known_tracks_review.json",root/"known_fit",steps=500)
fitted,_,_=load_scene(root/"known_fit/scene.json");fe=next(x for x in fitted["entities"] if x["id"]=="vehicle")
err=np.linalg.norm(np.array(fe["entity_to_world"])[:,:3,3]-pose[:,:3,3],axis=1)
check("Sparse evidence corrects known perturbed pose without SAM",rep["median_reprojection_px"]<1 and np.mean(err)<.08,
 {"median_reprojection_px":rep["median_reprojection_px"],"mean_position_error":float(np.mean(err))})
check("Invalid LK errors cannot poison sparse fit weights",np.isfinite(rep["median_reprojection_px"]))
# Surface fit retains explicit small thickness instead of using the depth range as occupied volume.
wall=np.zeros((5,96,96),bool);wall[:,:55,:]=True
pp,mm,info=vertical_surface(a,wall,np.array([0,-1,0]),.01)
check("Surface thickness comes from an explicit limited prior",np.isclose(info["thickness"]/info["width"],.01,atol=.003),info)
write_json(root/"report.json",{"checks":checks,"status":"passed","scope":"component behavior, not real-scene generalization"})
print(json.dumps(checks,indent=2))
