"""Agent-authored semantic axes and surface/frame proposals; all assumptions recorded."""
import copy,numpy as np,cv2
from scipy.ndimage import median_filter
from pathlib import Path
from scene_baseline.storage import read_json,write_json,reference
from scene_baseline.scene import validate_scene
from scene_baseline.masks import import_masks

def plane_points(a,t,uv,up,point):
 d=np.c_[uv,np.ones(len(uv))]@np.linalg.inv(a["K"][t]).T@a["c2w"][t,:3,:3].T
 o=a["c2w"][t,:3,3];den=d@up;distance=(point-o)@up/den
 return o+d*distance[:,None],distance

def part(eid,name,kind,center,linear,role):
 M=np.eye(4);M[:3,:3]=linear;M[:3,3]=center
 return {"id":eid+"_"+name,"kind":kind,"role":role,"canonical_to_entity":M.tolist()}

for st in [20,50]:
 out=Path(f"runs/two_cars_s{st}_semantic_r3");out.mkdir()
 gm=f"runs/two_cars_s{st}_geometry/manifest.json"
 old=read_json(f"runs/two_cars_s{st}_fit_r2/scene.json");s=copy.deepcopy(old)
 a=np.load(f"runs/two_cars_s{st}_geometry/geometry.npz")
 z=np.load(f"runs/two_cars_s{st}_masks/masks.npz");ids=z["ids"].tolist();masks=z["masks"].copy()
 up=np.array(s["ground_normal"]);up/=np.linalg.norm(up)
 right=a["c2w"][0,:3,0].copy();right-=up*(right@up);right/=np.linalg.norm(right)
 basis=np.stack([right,up,np.cross(right,up)],1)
 ground=next(e for e in s["entities"] if e["id"]=="ground")
 gpoint=np.array(ground["entity_to_world"])[0,:3,3]
 wheel=np.load(f"runs/two_cars_s{st}_sam3_wheels/masks.npz")
 far=next(e for e in s["entities"] if e["id"]=="far_truck");n=len(a["depth"])
 obs=np.array(far["observed"]);valid=np.zeros((n,3),bool);uv=np.zeros((n,3,2));world=np.zeros((n,3,3))
 for t in range(n):
  for j,k in enumerate([4,5,6]):
   ma=wheel["masks"][k,t];yy,xx=np.nonzero(ma)
   if not len(xx):continue
   # Reject a wheel mask outside the corresponding far-vehicle region.
   overlap=(ma&masks[ids.index("far_truck"),t]).sum()/max(ma.sum(),1)
   if overlap<.4:continue
   y=float(np.quantile(yy,.98));x=float(np.median(xx[yy>=np.quantile(yy,.9)]))
   p,d=plane_points(a,t,np.array([[x,y]]),up,gpoint)
   if d[0]<=0:continue
   uv[t,j]=[x,y];world[t,j]=p[0];valid[t,j]=True
 good=np.flatnonzero(valid.all(1)&obs)
 assert len(good)>15
 fwd=world[good,1]-world[good,0];fwd-=np.outer(fwd@up,up);wb=np.linalg.norm(fwd,axis=1);fwd/=wb[:,None]
 yaw=np.unwrap(np.arctan2(-(fwd@basis[:,2]),fwd@basis[:,0]));yaw=median_filter(yaw,size=5,mode="nearest")
 all_yaw=np.interp(np.arange(n),good,yaw)
 poses=np.array(far["entity_to_world"]);R=[]
 for y in all_yaw:
  c,ss=np.cos(y),np.sin(y);R.append(basis@np.array([[c,0,ss],[0,1,0],[-ss,0,c]]))
 R=np.array(R)
 track=np.linalg.norm(world[good,2]-world[good,1],axis=1)
 init=read_json(f"runs/two_cars_s{st}_init/scene.json");fe=next(e for e in init["entities"] if e["id"]=="far_truck")
 H=float(np.linalg.norm(np.array(fe["parts"][0]["canonical_to_entity"])[:3,1]))
 L=float(np.median(wb)/.64);D=float(np.median(track)/.92)
 side=-np.sign(np.median(np.einsum("ij,ij->i",world[good,2]-world[good,1],R[good,:,2])))
 contacts=np.array([[-.32*L,-.5*H,side*.46*D],[.32*L,-.5*H,side*.46*D],[.32*L,-.5*H,-side*.46*D]])
 poses[:,:3,:3]=R
 for t in good:poses[t,:3,3]=(world[t]-contacts@R[t].T).mean(0)
 far["entity_to_world"]=poses.tolist()
 P=[part("far_truck","body","box",[0,-.08*H,0],np.diag([L,.46*H,.85*D]),"lower body"),
    part("far_truck","cabin","box",[-.06*L,.2*H,0],np.diag([.44*L,.60*H,.78*D]),"cabin behind hood"),
    part("far_truck","front_bumper","box",[.49*L,-.12*H,0],np.diag([.05*L,.20*H,.93*D]),"+X is semantic front")]
 rot=np.array([[1.,0,0],[0,0,-1],[0,1,0]])
 for ix,x in enumerate([-.32*L,.32*L]):
  for iz,zz in enumerate([-.46*D,.46*D]):
   P.append(part("far_truck",f"wheel{ix}{iz}","cylinder",[x,-.32*H,zz],rot@np.diag([.36*H,.18*D,.36*H]),"tire; hidden rear opposite wheel is a symmetry prior"))
 far["parts"]=P;far["fit_constraints"]={"lock_yaw":True}
 far["landmarks"]={"entity_points":contacts.tolist(),"image_points":uv.tolist(),"valid":valid.tolist(),"weight":8.,
  "meaning":["rear same-side tire contact","front same-side tire contact","front opposite tire contact"],
  "source":reference(f"runs/two_cars_s{st}_sam3_wheels/manifest.json")}
 oldR=np.array(next(e for e in old["entities"] if e["id"]=="far_truck")["entity_to_world"])[:,:3,:3]
 diff=np.degrees(np.arccos(np.clip(np.einsum("ij,ij->i",oldR[good,:,0],R[good,:,0]),-1,1)))
 notes={"actor":"Codex current-chat agent after user orientation/solid-background report",
  "wheel_binding":{"4":"rear same-side","5":"front same-side","6":"front opposite-side",
   "review_frames":[16,32,48],"excluded_other_ids":"contain foreground identities or semantic switches"},
  "orientation":{"front":"+X, rear wheel4 to front wheel5","ground_dependency":"contact rays intersect estimated ground using predicted K/c2w",
   "valid_three_contact_frames":good.tolist(),"early_rotation":"nearest known yaw only within already-observed vehicle interval",
   "median_change_deg":float(np.median(diff)),"initial_LHD":[L,H,D],
   "unresolved":"ground/camera/segmentation errors can bias metric heading"},
  "bookcases":{},"table_probe":{"source":reference(f"runs/two_cars_s{st}_sam3_tables/manifest.json"),
   "decision":"not adopted; Step20 no masks, Step50 partial fragments; no synthetic substitute"}}
 for eid in ["shelf_left","shelf_right"]:
  ei=ids.index(eid);ma=masks[ei];depthvals=a["depth"][16:][ma[16:]];med=float(np.median(depthvals))
  cleaned=ma&(a["depth"]>=med*.65)&(a["depth"]<=med*1.35)
  excluded=int(ma.sum()-cleaned.sum());masks[ei]=cleaned
  e=next(x for x in s["entities"] if x["id"]==eid)
  candidates=[]
  for t in range(16,n):
   mm=cleaned[t];ys,xs=np.nonzero(mm)
   if not len(xs):continue
   contacts2=[]
   for x in np.unique(xs):
    y=np.flatnonzero(mm[:,x])[-1]
    if y+5<518 and masks[ids.index("ground"),t,y+2:y+6,x].any():contacts2.append([x,y])
   if len(contacts2)>=30:
    pp,dd=plane_points(a,t,np.array(contacts2),up,gpoint)
    pp=pp[(dd>0)&(dd<med*3)]
    if len(pp)>=30:candidates.append((len(pp),t,pp))
  yaw_offset=0.;anchor=int(np.argmax(cleaned[16:].sum((1,2)))+16);method="front-facing vertical-plane prior; no reliable floor-contact line"
  if candidates:
   _,anchor,pp=max(candidates,key=lambda v:v[0])
   flat=pp@basis[:,[0,2]];_,vec=np.linalg.eigh(np.cov(flat.T));axis=vec[:,-1]
   if axis[0]<0:axis=-axis
   yaw_offset=float(np.clip(np.arctan2(-axis[1],axis[0]),-np.pi/4,np.pi/4))
   method="floor-contact line, constrained to visible front-facing bookcase hypothesis"
  c0,ss=np.cos(yaw_offset),np.sin(yaw_offset);Rb=basis@np.array([[c0,0,ss],[0,1,0],[-ss,0,c0]])
  mm=cv2.erode(cleaned[anchor].astype("uint8"),np.ones((3,3),np.uint8))>0
  yy,xx=np.nonzero(mm);step=max(1,len(xx)//2000);yy,xx=yy[::step],xx[::step]
  ray=np.c_[xx,yy,np.ones(len(xx))]@np.linalg.inv(a["K"][anchor]).T@a["c2w"][anchor,:3,:3].T;o=a["c2w"][anchor,:3,3]
  pp=o+ray*a["depth"][anchor,yy,xx,None]
  plane=float(np.median(pp@Rb[:,2]));dd=(plane-o@Rb[:,2])/(ray@Rb[:,2])
  goodp=(dd>0)&(dd<med*3);surf=o+ray[goodp]*dd[goodp,None]
  q=surf@Rb;lo,hi=np.percentile(q,[2,98],axis=0)
  W=float(hi[0]-lo[0]);bottom=float(gpoint@up);top=float(hi[1]);height=max(top-bottom,float(hi[1]-lo[1]))
  thick=max(min(.16*height,.12*W),.008*med)
  ctr=Rb@np.array([(lo[0]+hi[0])/2,bottom+height/2,plane-thick/2])
  B=[]
  for sidev in [-1,1]:
   B.append(part(eid,"upright"+str(sidev),"box",[sidev*.48*W,0,0],np.diag([.04*W,height,thick]),"bookcase upright"))
  for level in range(6):
   B.append(part(eid,"shelf"+str(level),"box",[0,(-.48+level*.192)*height,0],np.diag([W,.025*height,thick]),"shelf board; contents not modeled"))
  pose=np.eye(4);pose[:3,:3]=Rb;pose[:3,3]=ctr
  oldsize=np.linalg.norm(np.array(e["parts"][0]["canonical_to_entity"])[:3,:3],axis=0).tolist()
  e["parts"]=B;e["entity_to_world"]=[pose.tolist()];e["fit_constraints"]={"lock_yaw":True}
  notes["bookcases"][eid]={"surface_anchor_frame":anchor,"method":method,"depth_band":[med*.65,med*1.35],
   "excluded_pixels":excluded,"old_bound_dimensions":oldsize,"new_WHD":[W,height,thick],
   "prior":"vertical open frame plus six shelves; thickness bounded semantically, not recovered hidden solid volume",
   "unmodeled":"individual books/back panels","source_mask":reference(f"runs/two_cars_s{st}_masks/manifest.json")}
 hyp=read_json(f"runs/two_cars_s{st}_hypothesis/hypothesis.json")
 hyp["actor"]=notes["actor"];hyp["semantic_revision"]=notes
 for e in hyp["entities"]:
  if e["id"] in ["far_truck","shelf_left","shelf_right"]:
   target=next(x for x in s["entities"] if x["id"]==e["id"]);e["parts"]=[{"id":x["id"],"kind":x["kind"],"role":x["role"]} for x in target["parts"]]
 write_json(out/"hypothesis.json",hyp);np.savez_compressed(out/"masks.npz",ids=np.array(ids),masks=masks)
 import_masks(gm,out/"hypothesis.json",out/"masks.npz",out/"masks_imported")
 s["hypothesis"]=reference(out/"hypothesis.json");s["masks_manifest"]=reference(out/"masks_imported/manifest.json")
 s["stage"]="agent_semantic_revision_requires_fit";s["revision"]={"parent_scene":reference(f"runs/two_cars_s{st}_fit_r2/scene.json"),"request":reference(out/"hypothesis.json")}
 s["limitations"]+=["orientation from semantic wheel correspondences depends on estimated ground/cameras",
 "bookshelf open topology/thickness is a prior; books not individually reconstructed"]
 validate_scene(s,n);write_json(out/"scene.json",s);write_json(out/"agent_decision.json",notes)
 np.savez_compressed(out/"wheel_contacts.npz",uv=uv,valid=valid,ground_points=world,local_contacts=contacts)
 print(st,"heading correction median",np.median(diff),"bookcase proposals",notes["bookcases"],flush=True)
