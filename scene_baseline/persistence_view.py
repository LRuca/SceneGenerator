"""Audit persistence without confusing a filled-in state with a measurement."""
import numpy as np,cv2,imageio.v2 as imageio
from PIL import Image,ImageDraw
from .scene import load_scene
from .geometry import scene_parts,render_parts,mesh_preview,unit_mesh,transform_mesh
from .global_views import look_at
from .storage import new_run,write_json,reference

def _colored(parts,K,c2w,w,h,entities,t):
 frame=mesh_preview(parts,K,c2w,w,h);_,lab=render_parts(parts,K,c2w,h,w)
 for k,part in enumerate(parts):
  e=entities[part["entity_id"]];used=e.get("observation_used",e.get("observed"))[t]
  color=np.array([30,210,230] if used else [255,166,45]) if e["motion"]=="rigid" else np.array([210,210,210])
  mask=lab==k;frame[mask]=(frame[mask]*.55+color*.45).astype("uint8")
 boundary=((lab!=np.roll(lab,1,0))|(lab!=np.roll(lab,1,1)))&(lab>=0);frame[boundary]=[22,22,22]
 return frame

def export(raw_scene,persistent_scene,out,width=256):
 raw,m,a=load_scene(raw_scene);scene,_,_=load_scene(persistent_scene);p=new_run(out);n,H,W=a["depth"].shape
 from .persistence import effective_cameras
 cameras=effective_cameras(scene,a)
 entities={e["id"]:e for e in scene["entities"]};dynamic=[e for e in scene["entities"] if e["motion"]=="rigid"]
 verts=[]
 for t in range(n):
  for part in scene_parts(scene,t):
   if part["kind"]=="plane":continue
   verts.append(transform_mesh(unit_mesh("box" if part["kind"]=="plane" else part["kind"]),part["canonical_to_world"])[0])
 points=np.concatenate(verts);center=(points.max(0)+points.min(0))/2;rad=max(np.linalg.norm(points-center,axis=1).max(),1e-3)
 up=np.array(scene["ground_normal"]);right=a["c2w"][0,:3,0].copy();right-=up*(right@up);right/=np.linalg.norm(right);front=np.cross(right,up)
 cam=look_at(center+rad*2.3*(right*.6+front*.9+up*.7),center,up)
 Kglobal=np.array([[width*.8,0,(width-1)/2],[0,width*.8,(width-1)/2],[0,0,1.]])
 sx=width/W;K=np.array([[sx,0,(sx-1)/2],[0,sx,(sx-1)/2],[0,0,1.]])@a["K"]
 videos=[];global_videos=[];diagnostics=[]
 for t in range(n):
  rgb=cv2.resize((a["rgb"][t]*255).astype("uint8"),(width,width))
  before=mesh_preview(scene_parts(raw,t),K[t],a["c2w"][t],width,width)
  after=_colored(scene_parts(scene,t),K[t],cameras[t],width,width,entities,t)
  panels=[rgb,before,after]
  canvas=Image.new("RGB",(width*3,width+96),"#131925");draw=ImageDraw.Draw(canvas)
  for j,(im,title) in enumerate(zip(panels,["RGB","Raw observation-gated","Persistent | evidence colors"])):
   canvas.paste(Image.fromarray(im),(j*width,30));draw.text((j*width+5,7),title,fill="white")
  draw.text((5,width+35),f"frame{t} | cyan=observation used | amber=inferred | gray=static",fill="white")
  for j,e in enumerate(dynamic[:3]):
   text=f'{e["id"]}: {e["state_source"][t]} / {e["visibility_state"][t]}'
   draw.text((5,width+52+j*13),text,fill="#ffbd65" if not e["observation_used"][t] else "#65e5ee")
  videos.append(np.asarray(canvas))
  b=mesh_preview(scene_parts(raw,t),Kglobal,cam,width,width)
  af=_colored(scene_parts(scene,t),Kglobal,cam,width,width,entities,t)
  gl=Image.new("RGB",(width*2,width+76),"#131925");dr=ImageDraw.Draw(gl)
  gl.paste(Image.fromarray(b),(0,30));gl.paste(Image.fromarray(af),(width,30))
  dr.text((5,7),"Old global: detection-gated",fill="white");dr.text((width+5,7),"Persistent global: inferred != measured",fill="white")
  dr.text((5,width+35),f"frame{t} / {n-1} | same global camera",fill="white")
  dr.text((5,width+53),"Missing mask does not remove an existing entity.",fill="white")
  global_videos.append(np.asarray(gl))
  diagnostics.append({"frame":int(a["frame_indices"][t]),"time_s":float(a["times_s"][t]),"entities":{
   e["id"]:{"exists":e["exists"][t],"observation_available":e["observation_available"][t],"observation_used":e["observation_used"][t],
     "state_source":e["state_source"][t],"visibility_state":e["visibility_state"][t],
     "uncertainty_position_proxy":e["uncertainty_proxy"]["position_radius"][t]} for e in scene["entities"]}})
 fps=float(1/np.median(np.diff(a["times_s"])))
 imageio.mimwrite(p/"source_evidence.mp4",videos,fps=fps,codec="libx264",macro_block_size=1,quality=8)
 imageio.mimwrite(p/"global_evidence.mp4",global_videos,fps=fps,codec="libx264",macro_block_size=1,quality=8)
 sheet=Image.new("RGB",(width*2*3,width+76),"#131925")
 for j,t in enumerate([0,n//2,n-1]):sheet.paste(Image.fromarray(global_videos[t]),(j*width*2,0))
 sheet.save(p/"global_contact.jpg");Image.fromarray(videos[n//2]).save(p/"source_contact.jpg")
 import matplotlib;matplotlib.use("Agg")
 import matplotlib.pyplot as plt
 fig,axes=plt.subplots(len(dynamic),3,figsize=(14,max(3,len(dynamic)*3)),squeeze=False)
 for i,e in enumerate(dynamic):
  old=next(x for x in raw["entities"] if x["id"]==e["id"]);op=np.array(old["entity_to_world"]);pp=np.array(e["entity_to_world"]);t=a["times_s"]
  for k in range(3):
   axes[i,0].plot(t,op[:,k,3],alpha=.4,ls="--");axes[i,0].plot(t,pp[:,k,3],label="XYZ"[k])
  axes[i,0].set_title(e["id"]+" world position");axes[i,0].legend()
  axes[i,1].plot(t,e["observation_available"],label="available");axes[i,1].plot(t,e["observation_used"],label="used",ls="--")
  axes[i,1].plot(t,e["exists"],label="exists",alpha=.5);axes[i,1].set_ylim(-.1,1.2);axes[i,1].legend();axes[i,1].set_title("existence / observation")
  axes[i,2].plot(t,e["uncertainty_proxy"]["position_radius"]);axes[i,2].set_title("uncertainty radius proxy (uncalibrated)")
 fig.tight_layout();fig.savefig(p/"state_curves.jpg",dpi=130);plt.close(fig)
 write_json(p/"states.json",{"scene":reference(persistent_scene),"frames":diagnostics,"color_policy":"cyan observation-supported state, amber inferred state; both fitted estimates, neither ground truth"})
 # Small immutable standalone viewer; experiment publisher can embed these videos.
 (p/"index.html").write_text('<meta charset="utf-8"><style>body{background:#131925;color:white;font:18px system-ui;margin:30px}video,img{max-width:100%}a{color:#8cf}</style><h1>Persistent state / observation and inference</h1><p>Cyan: observation-supported. Amber: inferred. Gray: static. Estimates are not measurements.</p><video controls src="source_evidence.mp4"></video><video controls src="global_evidence.mp4"></video><img src="state_curves.jpg"><p><a href="states.json">per-frame evidence state</a></p>')
 return str(p)
