import numpy as np,json
from scene_baseline.storage import read_json
for st in [20,50]:
 s=read_json(f"runs/two_cars_s{st}_fit_r2/scene.json");a=np.load(f"runs/two_cars_s{st}_geometry/geometry.npz");z=np.load(f"runs/two_cars_s{st}_sam3_wheels/masks.npz")
 up=np.array(s["ground_normal"]);g=next(e for e in s["entities"] if e["id"]=="ground");gpoint=np.array(g["entity_to_world"])[0,:3,3]
 f=next(e for e in s["entities"] if e["id"]=="far_truck")
 for t in [16,32,48]:
  contacts=[];uv=[]
  for k in [4,5,6]:
   yy,xx=np.nonzero(z["masks"][k,t]);y=np.quantile(yy,.98);x=np.median(xx[yy>=np.quantile(yy,.9)])
   d=np.linalg.inv(a["K"][t])@np.array([x,y,1]);d=a["c2w"][t,:3,:3]@d;o=a["c2w"][t,:3,3]
   p=o+d*((gpoint-o)@up/(d@up));contacts.append(p);uv.append([x,y])
  contacts=np.array(contacts);forward=contacts[1]-contacts[0];forward/=np.linalg.norm(forward)
  pred=np.array(f["entity_to_world"])[t,:3,0];cforward=a["c2w"][t,:3,:3].T@forward
  print(st,t,"uv",np.round(uv,1).tolist(),"wheelbase",np.linalg.norm(contacts[1]-contacts[0]),"track",np.linalg.norm(contacts[2]-contacts[1]),"cam-forward",np.round(cforward,3),"yawdiff",np.degrees(np.arccos(np.clip(pred@forward,-1,1))))
