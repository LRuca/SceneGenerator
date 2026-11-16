"""Same-camera and same-global-view comparisons, with sparse evidence diagnostics."""
import argparse
import numpy as np
import imageio.v2 as imageio
from PIL import Image,ImageDraw
from .scene import load_scene
from .geometry import scene_parts,mesh_preview,render_parts,unit_mesh,transform_mesh
from .global_views import look_at
from .physical_audit import audit
from .storage import new_run,read_json,write_json,reference,verify

def panel(rgb,title):
 im=Image.fromarray(rgb);d=ImageDraw.Draw(im);d.rectangle((0,0,im.width,24),fill="#101722");d.text((7,6),title,fill="white");return np.asarray(im)

def export(old_path,new_path,fit_path,out,width=300):
 old,m,a=load_scene(old_path);new,_,aa=load_scene(new_path)
 if old["geometry_manifest"]!=new["geometry_manifest"]:raise ValueError("Different video stages")
 p=new_run(out);n,H,W=a["depth"].shape;sc=width/W
 A=np.array([[sc,0,(sc-1)/2],[0,sc,(sc-1)/2],[0,0,1.]])
 lm=np.load(fit_path);uv=lm["image_points"];valid=lm["valid"]
 entity=next(e for e in new["entities"] if e["motion"]=="rigid")
 local=np.asarray(entity.get("landmarks",{}).get("entity_points",lm["entity_points"]))
 poses=np.asarray(entity["entity_to_world"])
 world=np.einsum("nij,pj->npi",poses[:,:3,:3],local)+poses[:,:3,3][:,None,:]
 wc=np.linalg.inv(a["c2w"])
 campts=np.einsum("nij,npj->npi",wc[:,:3,:3],world)+wc[:,:3,3][:,None,:]
 px=np.einsum("nij,npj->npi",a["K"],campts);pred=px[...,:2]/campts[...,2:]
 residual=np.linalg.norm(pred-uv,axis=-1)
 allv=[]
 for scene in [old,new]:
  for t in [0,n//2,n-1]:
   for part in scene_parts(scene,t):
    allv.append(transform_mesh(unit_mesh("box" if part["kind"]=="plane" else part["kind"]),part["canonical_to_world"])[0])
 xyz=np.concatenate(allv);ctr=(xyz.min(0)+xyz.max(0))/2;rad=np.linalg.norm(xyz-ctr,axis=1).max()
 up=np.asarray(new["ground_normal"]);up/=np.linalg.norm(up)
 right=a["c2w"][0,:3,0].copy();right-=up*(right@up);right/=np.linalg.norm(right)
 front=np.cross(right,up)
 gc=look_at(ctr+rad*(front*2.2+right*.75+up*.9),ctr,up)
 gK=np.array([[width*.9,0,(width-1)/2],[0,width*.9,(width-1)/2],[0,0,1.]])
 md=read_json(verify(new["masks_manifest"]))
 with np.load(verify(md["data"])) as z:masks=z["masks"];ids=z["ids"].tolist()
 metrics={};source=[];global_frames=[];white=[];overlays=[]
 for t in range(n):
  K=A@a["K"][t];cam=a["c2w"][t]
  rgb=np.array(Image.fromarray((a["rgb"][t]*255).astype("uint8")).resize((width,width)))
  op=scene_parts(old,t);np_=scene_parts(new,t)
  ow=mesh_preview(op,K,cam,width,width);nw=mesh_preview(np_,K,cam,width,width)
  support=entity.get("sparse_evidence",{}).get("support",[True]*n)[t]
  if not support:
   _,uncertain_labels=render_parts(np_,K,cam,width,width)
   for jj,part in enumerate(np_):
    if part["entity_id"]==entity["id"]:
     sel=uncertain_labels==jj;nw[sel]=(nw[sel]*.5+np.array([255,165,30])*.5).astype("uint8")
  _,lab=render_parts(np_,K,cam,width,width)
  boundary=((lab!=np.roll(lab,1,0))|(lab!=np.roll(lab,1,1)))&(lab>=0)
  overlay=rgb.copy();overlay[boundary]=[255,50,100]
  im=Image.fromarray(overlay);d=ImageDraw.Draw(im)
  for j in np.flatnonzero(valid[t]):
   x,y=(uv[t,j]+.5)*sc-.5;xx,yy=(pred[t,j]+.5)*sc-.5
   d.line((x,y,xx,yy),fill=(255,180,0),width=1)
   d.ellipse((x-1.5,y-1.5,x+1.5,y+1.5),fill=(0,255,170))
  ov=panel(np.asarray(im),f"V2 edge + sparse evidence | f{t}")
  source.append(np.concatenate([panel(rgb,f"RGB step | f{t}"),panel(ow,"V1 geometry / original VGGT camera"),panel(nw,"V2 pose UNCERTAIN" if not support else "V2 shared proxy / sparse pose"),ov],1))
  white.append(nw);overlays.append(ov)
  og=mesh_preview(op,gK,gc,width,width);ng=mesh_preview(np_,gK,gc,width,width)
  global_frames.append(np.concatenate([panel(og,f"V1 global | f{t}"),panel(ng,f"V2 global | f{t}"+(" | POSE UNCERTAIN" if not support else ""))],1))
  for eid in ids:
   target=masks[ids.index(eid),t]
   if target.sum()<24:continue
   predmask=np.zeros_like(lab,bool)
   for j,part in enumerate(np_):
    if part["entity_id"]==eid:predmask|=lab==j
   target=np.array(Image.fromarray(target).resize((width,width),Image.Resampling.NEAREST),bool)
   inter=(target&predmask).sum();union=(target|predmask).sum()
   metrics.setdefault(eid,[]).append({"frame":t,"iou":float(inter/max(union,1))})
 fps=m.get("fps",16)
 for name,frames in [("source_comparison",source),("global_comparison",global_frames),("white_model",white),("evidence_overlay",overlays)]:
  imageio.mimwrite(p/(name+".mp4"),frames,fps=fps,codec="libx264",macro_block_size=1,quality=8)
 sheet=Image.new("RGB",(width*4,width*5))
 for row,t in enumerate([0,4,12,24,n-1]):sheet.paste(Image.fromarray(source[t]),(0,row*width))
 sheet.save(p/"source_contact.jpg")
 Image.fromarray(global_frames[n//2]).save(p/"global_contact.jpg")
 report={"old_scene":reference(old_path),"new_scene":reference(new_path),
 "camera_comparison":"Original VGGT cameras used identically for both source renderings; frozen V1 geometry retained",
 "global_comparison":"same fixed view and scale; opaque geometry, no depth truncation",
 "observed_mask_iou":{k:{"mean":float(np.mean([r["iou"] for r in v])),"frames":v} for k,v in metrics.items()},
 "collision_before":audit(old),"collision_after":audit(new),
 "sparse_residual_median_px":float(np.median(residual[valid])),
 "sparse_residual_note":"fit diagnostic on tracked points; not independent accuracy or true 3D error"}
 np.savez_compressed(p/"actual_landmark_reprojection.npz",predicted_uv=pred,image_points=uv,valid=valid,reprojection_px=residual)
 write_json(p/"report.json",report)
 (p/"index.html").write_text("""<!doctype html><meta charset="utf-8"><title>Agent recovery V2</title><style>body{background:#121923;color:#e5e9f0;font:16px system-ui;max-width:1250px;margin:30px auto}video{width:100%}a{color:#80c8ff}img{max-width:100%}</style><h1>Agent recovery V2 — development comparison</h1><p>Same diffusion stage; original fixed VGGT cameras. Sparse points are candidate observations. Shared vehicle proxy remains coarse. Wall back-side thickness is an explicit prior. Orange source geometry / POSE UNCERTAIN means insufficient point support; these coordinates are hypotheses, not measured hidden trajectories.</p><h2>Source: RGB / V1 / V2 / evidence</h2><video src="source_comparison.mp4" controls loop></video><h2>Same global viewpoint</h2><video src="global_comparison.mp4" controls loop></video><h2>White model</h2><video src="white_model.mp4" controls loop></video><p><a href="report.json">Metrics and unresolved collisions</a> · <a href="source_contact.jpg">Source contact sheet</a></p>""")
 return report

if __name__=="__main__":
 ap=argparse.ArgumentParser()
 for k in ["old","new","fit","out"]:ap.add_argument("--"+k,required=True)
 ap.add_argument("--width",type=int,default=300);x=ap.parse_args()
 r=export(x.old,x.new,x.fit,x.out,x.width)
 print("collisions",len(r["collision_before"]["events"]),len(r["collision_after"]["events"]))
 print("mask_iou",{k:v["mean"] for k,v in r["observed_mask_iou"].items()})
