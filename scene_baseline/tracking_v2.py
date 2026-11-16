"""Sparse body evidence independent of dense masks; explicit review requests."""
import argparse
import cv2
import numpy as np
from PIL import Image,ImageDraw
import imageio.v2 as imageio
from .contracts import load_bundle
from .storage import new_run,reference,write_json,read_json,verify

def intervals(flags):
 out=[];start=None
 for t,b in enumerate(list(flags)+[False]):
  if b and start is None:start=t
  if not b and start is not None:out.append([start,t-1]);start=None
 return out

def track_points(gray,anchor,seed,masks=None,fb_max=1.5,photo_max=25.):
 n=len(gray);P=len(seed);uv=np.zeros((n,P,2),np.float32);valid=np.zeros((n,P),bool)
 fb=np.full((n,P),1e4,np.float32);photo=fb.copy()
 uv[anchor]=seed;valid[anchor]=True;fb[anchor]=0;photo[anchor]=0
 H,W=gray[0].shape
 for direction in [-1,1]:
  for t in range(anchor+direction,n if direction>0 else -1,direction):
   prev=t-direction;idx=np.flatnonzero(valid[prev])
   if not len(idx):break
   p=uv[prev,idx,None,:]
   a,st,err=cv2.calcOpticalFlowPyrLK(gray[prev],gray[t],p,None,winSize=(25,25),maxLevel=4,
      criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,40,.01))
   if a is None:break
   b,back,_=cv2.calcOpticalFlowPyrLK(gray[t],gray[prev],a,None,winSize=(25,25),maxLevel=4,
      criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,40,.01))
   if b is None:break
   a=a[:,0];delta=np.linalg.norm(b[:,0]-p[:,0],axis=1);ee=np.nan_to_num(err[:,0],nan=1e4,posinf=1e4,neginf=1e4)
   ok=(st[:,0]>0)&(back[:,0]>0)&(delta<=fb_max)&(ee>=0)&(ee<=photo_max)
   ok &= (a[:,0]>=3)&(a[:,0]<W-3)&(a[:,1]>=3)&(a[:,1]<H-3)
   # When available, a trusted mask can reject drift. Empty masks are NOT negative evidence.
   if masks is not None and masks[t].sum()>=24:
    dil=cv2.dilate(masks[t].astype("uint8"),np.ones((7,7),np.uint8))>0
    pix=np.rint(a).astype(int);pix[:,0]=pix[:,0].clip(0,W-1);pix[:,1]=pix[:,1].clip(0,H-1)
    ok &= dil[pix[:,1],pix[:,0]]
   uv[t,idx]=a;fb[t,idx]=delta;photo[t,idx]=ee;valid[t,idx]=ok
 return uv,valid,fb,photo

def run(geometry,masks_manifest,entity,out,max_points=32):
 m,a=load_bundle(geometry);md=read_json(masks_manifest)
 if md["geometry_manifest"]!=reference(geometry):raise ValueError("Mask geometry mismatch")
 with np.load(verify(md["data"])) as z:
  masks=z["masks"][z["ids"].tolist().index(entity)]
 rgb=(a["rgb"]*255).clip(0,255).astype("uint8")
 gray=[cv2.cvtColor(x,cv2.COLOR_RGB2GRAY) for x in rgb];n,H,W=masks.shape
 good=np.flatnonzero(masks.sum((1,2))>=24)
 p=new_run(out)
 scores=[];corners={}
 for t in good:
  interior=cv2.erode(masks[t].astype("uint8"),np.ones((5,5),np.uint8))
  pts=cv2.goodFeaturesToTrack(gray[t],maxCorners=max_points,qualityLevel=.015,minDistance=7,mask=interior,blockSize=5)
  if pts is not None:
   corners[int(t)]=pts[:,0];scores.append((len(pts),int(t)))

 anchors=sorted(set([min(corners),max(scores)[1],max(corners)])) if scores else []
 uu=[];vv=[];ff=[];ee=[];aa=[]
 for t in anchors:
  u,v,f,e=track_points(gray,t,corners[t],masks)
  keep=v.sum(0)>=4
  if keep.any():
   uu.append(u[:,keep]);vv.append(v[:,keep]);ff.append(f[:,keep]);ee.append(e[:,keep]);aa.extend([t]*int(keep.sum()))
 uv=np.concatenate(uu,1) if uu else np.zeros((n,0,2),np.float32)
 valid=np.concatenate(vv,1) if vv else np.zeros((n,0),bool)
 fb=np.concatenate(ff,1) if ff else np.zeros((n,0),np.float32)
 photo=np.concatenate(ee,1) if ee else np.zeros((n,0),np.float32)
 # Support includes spatial spread, not only point count. Tracking remains candidate evidence.
 count=valid.sum(1);spread=[]
 for t in range(n):
  pts=uv[t,valid[t]]
  spread.append(float(cv2.contourArea(cv2.convexHull(pts))) if len(pts)>=3 else 0.)
 support=(count>=6)&(np.asarray(spread)>=36)
 np.savez_compressed(p/"tracks.npz",uv=uv,valid=valid,fb_error=fb,photo_error=photo,anchor_frame=np.array(aa))
 request={"schema":"scene_baseline.agent_review_request.v2","actor_requested":"current_chat_codex",
  "geometry_manifest":reference(geometry),"entity_id":entity,
  "triggers":{"insufficient_track_support":intervals(~support),"sam_empty":intervals(masks.sum((1,2))<24)},
  "representative_frames":sorted(set([t for lo,hi in intervals((~support)|(masks.sum((1,2))<24)) for t in [lo,(lo+hi)//2,hi]])),
  "required_review":"Confirm identity and body-fixed correspondences; add only necessary region/points. Do not infer absence from SAM.",
  "status":"pending_agent_review","caution":"LK forward/back consistency cannot establish semantic identity or physical rigidity."}
 write_json(p/"review_request.json",request)
 manifest={"schema":"scene_baseline.sparse_tracks.v2","geometry_manifest":reference(geometry),
 "masks_manifest":reference(masks_manifest),"entity_id":entity,"method":"OpenCV pyramidal LK + per-pair forward/back and photometric tests; masks reject drift only where nonempty",
 "parameters":{"max_points_per_anchor":max_points,"fb_max_px":1.5,"photo_max":25.,"win_size":25,"pyramid_levels":4},
 "anchors":anchors,"point_count":len(aa),"valid_points_per_frame":count.tolist(),"support_per_frame":support.tolist(),
 "convex_hull_area_px":spread,"data":reference(p/"tracks.npz"),"automatic_identity_verified":False,
 "status":"candidate_evidence_requires_visual_review"}
 write_json(p/"manifest.json",manifest)
 frames=[]
 for t in range(n):
  im=Image.fromarray(rgb[t]);d=ImageDraw.Draw(im)
  for j in np.flatnonzero(valid[t]):
   x,y=uv[t,j];d.ellipse((x-2,y-2,x+2,y+2),fill=(0,255,100))
  d.rectangle((0,0,W,30),fill="black")
  d.text((8,8),f"frame {t} | LK candidate points {count[t]} | SAM {'yes' if masks[t].any() else 'EMPTY'}",fill="white")
  frames.append(np.asarray(im))
 imageio.mimwrite(p/"tracking.mp4",frames,fps=float(m.get("fps",16)),codec="libx264",macro_block_size=1,quality=8)
 ids=sorted(set([0,4,8,12,16,17,24,36,48])&set(range(n)))
 sheet=Image.new("RGB",(320*3,345*int(np.ceil(len(ids)/3))),"#20252c");d=ImageDraw.Draw(sheet)
 for k,t in enumerate(ids):
  im=Image.fromarray(frames[t]);im.thumbnail((320,320));sheet.paste(im,(k%3*320,k//3*345))
 sheet.save(p/"contact.jpg")
 return manifest

if __name__=="__main__":
 ap=argparse.ArgumentParser();ap.add_argument("--geometry",required=True);ap.add_argument("--masks",required=True);ap.add_argument("--entity",required=True);ap.add_argument("--out",required=True)
 x=ap.parse_args();print(run(x.geometry,x.masks,x.entity,x.out))
