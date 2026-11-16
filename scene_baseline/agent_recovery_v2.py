"""Explicit current-chat agent fallback; manual observations never become silent truth."""
import argparse,copy
import numpy as np
from .storage import read_json,write_json,reference,verify,new_run
from .contracts import load_bundle
from .tracking_v2 import track_points,intervals

def supplement(geometry,tracks_path,review_path,out):
 import cv2
 tr=read_json(tracks_path);r=read_json(review_path);m,a=load_bundle(geometry)
 if tr["geometry_manifest"]!=reference(geometry) or r["tracks_manifest"]!=reference(tracks_path):
  raise ValueError("Review/track/geometry binding mismatch")
 if r.get("actor")!="current_chat_codex":raise ValueError("Explicit review actor required")
 if r.get("decision")!="supplement_observations":raise ValueError("Review action is not observation supplement")
 with np.load(verify(tr["data"])) as z:
  uv=z["uv"].copy();valid=z["valid"].copy();fb=z["fb_error"].copy();photo=z["photo_error"].copy();anchors=z["anchor_frame"].copy()
  existing_source=z["source_kind"].copy() if "source_kind" in z else np.where(valid,1,0).astype("uint8")
 n,H,W=a["depth"].shape;P=uv.shape[1]
 provenance=existing_source
 for reject in r.get("reject_points",[]):
  j=int(reject["point_id"])
  if not 0<=j<P:raise ValueError("Unknown point to reject")
  frames=reject.get("frames",list(range(n)))
  valid[frames,j]=False;provenance[frames,j]=0
 gray=None;added=0
 for point in r.get("add_points",[]):
  obs=point["observations"];anchor=int(point["anchor_frame"])
  measured=[o for o in obs if o["evidence"]=="visual_annotation"]
  anchor_obs=next((o for o in measured if int(o["frame"])==anchor),None)
  if anchor_obs is None:raise ValueError("Body point requires a visually observed anchor")
  for o in obs:
   if o["evidence"] not in ["visual_annotation","inferred_hypothesis"]:raise ValueError("Unknown evidence class")
   t=int(o["frame"]);xy=np.asarray(o["uv"],float)
   if not 0<=t<n or xy.shape!=(2,) or not np.isfinite(xy).all() or not 0<=xy[0]<W or not 0<=xy[1]<H:raise ValueError("Invalid annotated point")
  seed=np.array(anchor_obs["uv"],np.float32)[None]
  if point.get("propagate",False):
   if gray is None:gray=[cv2.cvtColor((x*255).astype("uint8"),cv2.COLOR_RGB2GRAY) for x in a["rgb"]]
   u,v,f,e=track_points(gray,anchor,seed,None)
   source=np.where(v,1,0).astype("uint8")
  else:
   u=np.zeros((n,1,2),np.float32);v=np.zeros((n,1),bool)
   f=np.full((n,1),1e4,np.float32);e=f.copy();source=np.zeros((n,1),np.uint8)
  for o in measured:
   t=int(o["frame"]);u[t,0]=o["uv"];v[t,0]=True;f[t,0]=0;e[t,0]=0;source[t,0]=2
  # Inferred coordinates are retained in review JSON, never inserted as measurements.
  uv=np.concatenate([uv,u],1);valid=np.concatenate([valid,v],1);fb=np.concatenate([fb,f],1);photo=np.concatenate([photo,e],1)
  provenance=np.concatenate([provenance,source],1);anchors=np.append(anchors,anchor);added+=1
 p=new_run(out)
 np.savez_compressed(p/"tracks.npz",uv=uv,valid=valid,fb_error=fb,photo_error=photo,anchor_frame=anchors,source_kind=provenance)
 new=copy.deepcopy(tr);new["data"]=reference(p/"tracks.npz");new["parent"]=reference(tracks_path)
 new["agent_review"]=reference(review_path);new["point_count"]=uv.shape[1];new["valid_points_per_frame"]=valid.sum(1).tolist()
 support=[]
 for t in range(n):
  pts=uv[t,valid[t]]
  area=cv2.contourArea(cv2.convexHull(pts)) if len(pts)>=3 else 0
  support.append(bool(len(pts)>=6 and area>=36))
 new["support_per_frame"]=support
 new["source_kind"]={"0":"unavailable","1":"automatic_candidate","2":"agent_visual_annotation"}
 new["status"]="supplemented_candidate_requires_render_validation";new["manual_points_added"]=added
 write_json(p/"manifest.json",new)
 write_json(p/"next_review.json",{"unsupported_intervals":intervals(~np.array(support)),
 "status":"fit_then_render_review" if all(support) else "agent_review_still_required",
 "note":"Agent judgments are not independent accuracy labels. No guessed pose or interpolated pixel is a measured observation."})
 return new

if __name__=="__main__":
 ap=argparse.ArgumentParser()
 for k in ["geometry","tracks","review","out"]:ap.add_argument("--"+k,required=True)
 x=ap.parse_args();print(supplement(x.geometry,x.tracks,x.review,x.out))
