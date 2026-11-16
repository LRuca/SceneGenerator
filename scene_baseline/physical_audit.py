"""Diagnose coarse interpenetration; do not pretend uncertain geometry is a simulator."""
import numpy as np
from .geometry import scene_parts

def box_overlap(A,B):
 # Oriented bound of each primitive. Cylinder/sphere bounds are conservative.
 A=np.asarray(A);B=np.asarray(B);aa=A[:3,:3];bb=B[:3,:3]
 ha=np.linalg.norm(aa,axis=0)/2;hb=np.linalg.norm(bb,axis=0)/2
 U=aa/(2*ha);V=bb/(2*hb);delta=B[:3,3]-A[:3,3]
 axes=[U[:,i] for i in range(3)]+[V[:,i] for i in range(3)]
 axes +=[np.cross(U[:,i],V[:,j]) for i in range(3) for j in range(3)]
 overlaps=[]
 for axis in axes:
  norm=np.linalg.norm(axis)
  if norm<1e-8:continue
  axis=axis/norm;ra=np.abs(axis@U)@ha;rb=np.abs(axis@V)@hb
  overlap=ra+rb-abs(axis@delta)
  if overlap<=0:return 0.
  overlaps.append(overlap)
 return float(min(overlaps))

def audit(scene):
 n=len(scene["entities"][0]["renderable"]);by={e["id"]:e for e in scene["entities"]};events=[]
 for t in range(n):
  parts=scene_parts(scene,t)
  for i,a in enumerate(parts):
   for b in parts[:i]:
    if a["entity_id"]==b["entity_id"]:continue
    if by[a["entity_id"]]["motion"]!="rigid" and by[b["entity_id"]]["motion"]!="rigid":continue
    if a["kind"]=="plane" or b["kind"]=="plane":continue
    overlap=box_overlap(a["canonical_to_world"],b["canonical_to_world"])
    if overlap>0:events.append({"frame":t,"a":a["entity_id"],"b":b["entity_id"],"overlap_world_bound":overlap})
 return {"events":events,"status":"conflicts_detected" if events else "no_coarse_bound_conflict_detected",
  "interpretation":"oriented primitive bounds are conservative; no rigid-body dynamics or collision correction; wrong inferred interactions must remain flagged"}
