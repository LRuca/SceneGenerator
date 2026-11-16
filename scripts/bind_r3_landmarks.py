"""Bind exported landmark coordinates to the fitted entity scale; geometry unchanged."""
from pathlib import Path
import copy,numpy as np
from scene_baseline.storage import read_json,write_json,reference
from scene_baseline.scene import validate_scene
for st in [20,50]:
 src=f"runs/two_cars_s{st}_fit_r3/scene.json";s=read_json(src);initial=read_json(f"runs/two_cars_s{st}_semantic_r3/scene.json")
 e=next(x for x in s["entities"] if x["id"]=="far_truck");b=next(x for x in initial["entities"] if x["id"]=="far_truck")
 def body(x):return np.array(next(p for p in x["parts"] if p["id"]=="far_truck_body")["canonical_to_entity"])
 scale=np.linalg.norm(body(e)[:3,:3],axis=0)/np.linalg.norm(body(b)[:3,:3],axis=0)
 desired=np.array(b["landmarks"]["entity_points"])*scale
 changed=not np.allclose(e["landmarks"]["entity_points"],desired,atol=1e-7)
 e["landmarks"]["entity_points"]=desired.tolist()
 s["export_metadata_binding"]={"source":reference(src),"landmark_coordinate_rescaling":scale.tolist(),"geometry_changed":False}
 p=Path(f"runs/two_cars_s{st}_fit_r3_bound");p.mkdir();validate_scene(s,49);write_json(p/"scene.json",s)
 write_json(p/"report.json",{"source_fit":reference(f"runs/two_cars_s{st}_fit_r3/report.json"),"binding_changed":changed,"geometry_unchanged":True})
 print(st,"bound landmarks",scale.tolist(),flush=True)
