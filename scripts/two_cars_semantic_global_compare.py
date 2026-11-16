import numpy as np
from pathlib import Path
from PIL import Image,ImageDraw
from scene_baseline.scene import load_scene
from scene_baseline.geometry import scene_parts,transform_mesh,unit_mesh,mesh_preview,render_parts
from scene_baseline.global_views import look_at
p=Path("runs/two_cars_semantic_comparison");p.mkdir()
for st in [20,50]:
 old,m,a=load_scene(f"runs/two_cars_s{st}_fit_r2/scene.json");new,_,_=load_scene(f"runs/two_cars_s{st}_semantic_final/scene.json")
 allpoints=[];t=12
 for s in [old,new]:
  for part in scene_parts(s,t):
   allpoints.append(transform_mesh(unit_mesh("box" if part["kind"]=="plane" else part["kind"]),part["canonical_to_world"])[0])
 points=np.concatenate(allpoints);center=(points.min(0)+points.max(0))/2;rad=np.linalg.norm(points-center,axis=1).max()
 up=np.array(new["ground_normal"]);right=a["c2w"][0,:3,0].copy();right-=up*(right@up);right/=np.linalg.norm(right);front=np.cross(right,up)
 w,h=640,480;K=np.array([[w*.9,0,(w-1)/2],[0,w*.9,(h-1)/2],[0,0,1.]])
 canvas=Image.new("RGB",(1280,530),"#131925");d=ImageDraw.Draw(canvas)
 pose=look_at(center+rad*2.35*(np.cos(np.pi/2+.6)*right+np.sin(np.pi/2+.6)*front+up*.6),center,up)
 colors={"shelf_left":[240,190,30],"shelf_right":[180,70,230],"far_truck":[30,210,230],"near_truck":[230,60,170]}
 for col,(s,title) in enumerate([(old,"Previous R2: PCA heading / solid bounds"),(new,"Revised: wheel heading / open shelf frame")]):
  parts=scene_parts(s,t);frame=mesh_preview(parts,K,pose,w,h);_,labels=render_parts(parts,K,pose,h,w)
  for k,part in enumerate(parts):
   if part["entity_id"] in colors:
    mask=labels==k;frame[mask]=(frame[mask]*.65+np.array(colors[part["entity_id"]])*.35).astype("uint8")
  canvas.paste(Image.fromarray(frame),(col*w,35));d.text((col*w+10,10),title,fill="white")
 d.text((10,515),"Same world camera | gold = left shelf, purple = right shelf, cyan = far truck | frame12",fill="white")
 canvas.save(p/f"global_before_after_s{st}.jpg")
print("same-camera global comparisons rendered")
