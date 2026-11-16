"""Freeze a jointly visible instant for global orbit, not a middle-frame singleton."""
import numpy as np
from PIL import Image,ImageDraw
import imageio.v2 as imageio
from scene_baseline.scene import load_scene
from scene_baseline.geometry import scene_parts,transform_mesh,unit_mesh,mesh_preview,render_parts
from scene_baseline.global_views import look_at
from scene_baseline.storage import new_run,reference,write_json
for st in [20,50]:
 path=f"runs/two_cars_s{st}_fit_r3_bound/scene.json";s,m,a=load_scene(path)
 p=new_run(f"runs/two_cars_s{st}_overlap_r3");t=12;parts=scene_parts(s,t)
 up=np.array(s["ground_normal"]);up/=np.linalg.norm(up)
 right=a["c2w"][0,:3,0].copy();right-=up*(right@up);right/=np.linalg.norm(right);front=np.cross(right,up)
 vv=[]
 for part in parts:
  if part["entity_id"] in ["near_truck","far_truck"]:
   vv.append(transform_mesh(unit_mesh(part["kind"]),part["canonical_to_world"])[0])
 v=np.concatenate(vv);center=(v.min(0)+v.max(0))/2;rad=np.linalg.norm(v-center,axis=1).max()
 w=448;K=np.array([[w*.85,0,(w-1)/2],[0,w*.85,(w-1)/2],[0,0,1.]])
 def draw(angle):
  pose=look_at(center+rad*2.2*(right*np.cos(angle)+front*np.sin(angle)+up*.7),center,up)
  im=mesh_preview(parts,K,pose,w,w);_,lab=render_parts(parts,K,pose,w,w)
  b=((lab!=np.roll(lab,1,0))|(lab!=np.roll(lab,1,1)))&(lab>=0);im[b]=[20,20,20]
  return im
 frames=[draw(x) for x in np.linspace(np.pi/2,np.pi/2+np.pi*2,96,endpoint=False)]
 imageio.mimwrite(p/"orbit_both_cars.mp4",frames,fps=24,codec="libx264",macro_block_size=1,quality=8)
 c=Image.new("RGB",(w*3,w+30),"#141923");dr=ImageDraw.Draw(c)
 for j,angle in enumerate([np.pi/2-.7,np.pi/2+.7,np.pi/2+np.pi]):
  c.paste(Image.fromarray(draw(angle)),(j*w,30));dr.text((j*w+8,8),f"Two-car snapshot | frame 12 | view {j+1}",fill="white")
 c.save(p/"both_cars.jpg")
 offset=1;lines=["# Both vehicles visible; frozen frame12; arbitrary VGGT scale"]
 for part in parts:
  v,f=transform_mesh(unit_mesh("box" if part["kind"]=="plane" else part["kind"]),part["canonical_to_world"])[:2]
  lines.append("o "+part["entity_id"]);lines.extend("v %.8f %.8f %.8f"%tuple(q) for q in v)
  lines.extend("f "+" ".join(str(int(i)+offset) for i in face) for face in f);offset+=len(v)
 (p/"both_cars_frame12.obj").write_text("\n".join(lines)+"\n")
 write_json(p/"manifest.json",{"scene":reference(path),"frame_index":12,"time_s":float(a["times_s"][12]),
  "purpose":"orbit freezes a time with both vehicles observed; view framing emphasizes vehicles, all scene geometry retained",
  "rendering":"same scene parts and depths; no RGB backdrop, no far clipping"})
 print(st,"overlap orbit done",flush=True)
