"""Same RGB/masks, compare old/new semantic heading and wheel reprojections."""
from pathlib import Path
import numpy as np,cv2,imageio.v2 as imageio
from PIL import Image,ImageDraw
from scene_baseline.storage import read_json,write_json,reference
def project(a,t,p):
 cam=(p-a["c2w"][t,:3,3])@a["c2w"][t,:3,:3]
 pix=cam@a["K"][t].T
 return pix[:,:2]/pix[:,2:]
for st in [20,50]:
 a=np.load(f"runs/two_cars_s{st}_geometry/geometry.npz")
 z=np.load(f"runs/two_cars_s{st}_masks/masks.npz");v=z["masks"][list(z["ids"]).index("far_truck")]
 old=read_json(f"runs/two_cars_s{st}_fit_r2/scene.json");new=read_json(f"runs/two_cars_s{st}_fit_r3_bound/scene.json")
 oe=next(x for x in old["entities"] if x["id"]=="far_truck");ne=next(x for x in new["entities"] if x["id"]=="far_truck")
 oldpose=np.array(oe["entity_to_world"]);newpose=np.array(ne["entity_to_world"])
 lm=ne["landmarks"];uv=np.array(lm["image_points"]);valid=np.array(lm["valid"])
 local=np.array(lm["entity_points"])
 newerr=[];olderr_by_side={}
 oldpoints=[]
 for side,ids in [("negative",["wheel00","wheel10","wheel11"]),("positive",["wheel01","wheel11","wheel10"])]:
  pts=[]
  for pid in ids:
   M=np.array(next(p for p in oe["parts"] if p["id"]=="far_truck_"+pid)["canonical_to_entity"])
   pts.append((M@np.array([0,0,.5,1]))[:3])
  pts=np.array(pts);errs=[]
  for t in np.flatnonzero(valid.all(1)):
   projected=project(a,t,pts@oldpose[t,:3,:3].T+oldpose[t,:3,3])
   errs.extend(np.linalg.norm(projected-uv[t],axis=1).tolist())
  olderr_by_side[side]=float(np.mean(errs));oldpoints.append(pts)
 for t in np.flatnonzero(valid.all(1)):
  projected=project(a,t,local@newpose[t,:3,:3].T+newpose[t,:3,3])
  newerr.extend(np.linalg.norm(projected-uv[t],axis=1).tolist())
 p=Path(f"runs/two_cars_s{st}_heading_review");p.mkdir()
 frames=[];caps=[]
 for rd in [2,3]:caps.append(cv2.VideoCapture(f"runs/two_cars_s{st}_delivery_r{rd}/white.mp4"))
 for t in range(16,49):
  yy,xx=np.nonzero(v[t]);margin=25;box=[max(0,int(xx.min())-margin),max(0,int(yy.min())-margin),min(518,int(xx.max())+margin),min(518,int(yy.max())+margin)]
  rgb=(a["rgb"][t]*255).astype("uint8")
  for j,color in enumerate([(255,50,210),(20,240,240),(240,210,20)]):
   if valid[t,j]:
    pt=tuple(np.rint(uv[t,j]).astype(int));cv2.circle(rgb,pt,4,color,-1);cv2.putText(rgb,["rear","front","opp"][j],(pt[0]-10,pt[1]+13),cv2.FONT_HERSHEY_SIMPLEX,.35,color,1,cv2.LINE_AA)
  panels=[rgb]
  for k,(cap,e,pose) in enumerate(zip(caps,[oe,ne],[oldpose,newpose])):
   cap.set(cv2.CAP_PROP_POS_FRAMES,t);ok,im=cap.read();im=cv2.cvtColor(im,cv2.COLOR_BGR2RGB);im=cv2.resize(im,(518,518))
   C=pose[t,:3,3];F=pose[t,:3,0];up=pose[t,:3,1]
   body=np.array(next(p for p in e["parts"] if p["id"]=="far_truck_body")["canonical_to_entity"])
   L=np.linalg.norm(body[:3,0]);H=np.linalg.norm(body[:3,1])/.46
   points=np.array([C+up*.65*H-F*.2*L,C+up*.65*H+F*.5*L])
   arrow=np.rint(project(a,t,points)).astype(int)
   cv2.arrowedLine(im,tuple(arrow[0]),tuple(arrow[1]),(255,100,50) if k==0 else (20,230,230),2,cv2.LINE_AA,tipLength=.25)
   panels.append(im)
  canvas=Image.new("RGB",(900,290),"#131925");dr=ImageDraw.Draw(canvas)
  for j,(im,title) in enumerate(zip(panels,["RGB + wheel correspondences","Old R2 semantic front +X","Revised R3 semantic front +X"])):
   crop=Image.fromarray(im).crop(box);crop.thumbnail((300,250));canvas.paste(crop,(j*300,34));dr.text((j*300+5,5),title,fill="white")
  frames.append(np.asarray(canvas))
 for cap in caps:cap.release()
 imageio.mimwrite(p/"heading_comparison.mp4",frames,fps=16,codec="libx264",macro_block_size=1,quality=8)
 Image.fromarray(frames[16]).save(p/"heading_comparison.jpg")
 oldm=read_json(f"runs/two_cars_s{st}_delivery_r2/metrics.json");newm=read_json(f"runs/two_cars_s{st}_delivery_r3/metrics.json")
 def ious(m):
  return {eid:float(np.mean([f["entities"][eid]["visible_mask_iou"] for f in m["frames"] if f["entities"][eid]["target_pixels"]>0])) for eid in ["near_truck","far_truck","shelf_left","shelf_right"]}
 nearold=next(x for x in old["entities"] if x["id"]=="near_truck");nearnew=next(x for x in new["entities"] if x["id"]=="near_truck")
 assert nearold==nearnew
 assert np.allclose(newpose[:,:3,:3],np.array(read_json(f"runs/two_cars_s{st}_semantic_r3/scene.json")["entities"][1]["entity_to_world"])[:,:3,:3],atol=1e-6)
 write_json(p/"report.json",{"old_scene":reference(f"runs/two_cars_s{st}_fit_r2/scene.json"),"new_scene":reference(f"runs/two_cars_s{st}_fit_r3_bound/scene.json"),
 "old_visible_iou":ious(oldm),"new_visible_iou":ious(newm),
 "wheel_contact_reprojection_mean_px":{"old_best_side_mapping":min(olderr_by_side.values()),"old_side_candidates":olderr_by_side,"new":float(np.mean(newerr))},
 "evaluation_mask_policy":"same original SAM masks for old and new; filtered masks only used for fit",
 "near_geometry_unchanged":True,"yaw_lock_verified":True,
 "video_frames":list(range(16,49)),
 "background_tradeoff":"open shelf frame omits books; reduced filled-mask coverage does not demonstrate a complete reconstruction",
 "interpretation":"semantic wheel-ground estimate and reprojection consistency, not independent true 3D pose accuracy"})
 print(st,"old/new vehicle IoU",ious(oldm)["far_truck"],ious(newm)["far_truck"],"wheel pixel error",min(olderr_by_side.values()),np.mean(newerr),flush=True)
