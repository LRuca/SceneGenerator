"""Known-geometry behavioral test for visible-front/back ambiguity."""
import copy,json
import numpy as np
from scene_baseline.fixtures import create
from scene_baseline.storage import new_run,read_json,write_json,reference
from scene_baseline.fit_scene_visibility_v3 import fit
import sys
p=new_run(sys.argv[1] if len(sys.argv)>1 else "runs/visibility_v3_synthetic")
f=create(p/"fixture",n=5,size=96)
truth=read_json(f/"truth_scene.json")
vehicle=next(e for e in truth["entities"] if e["id"]=="vehicle")
truepose=np.array(vehicle["entity_to_world"])
bad=copy.deepcopy(truth)
e=next(e for e in bad["entities"] if e["id"]=="vehicle")
pose=np.array(e["entity_to_world"]);pose[:,:,3][:,2]-=.55
e["entity_to_world"]=pose.tolist();e["fit_constraints"]={"lock_shape":True,"lock_yaw":True}
write_json(p/"bad_scene.json",bad)
weights={"depth":0.,"silhouette":0.,"free_space":0.,"shape_prior":0.,"motion":0.,"support":0.,"landmark":0.}
rows=[]
for mode in ["sdf","first_hit","visible_faces"]:
 fit(p/"bad_scene.json",p/mode,steps=300,resolution=64,lr=.003,entity_ids=["vehicle"],surface_mode=mode,term_weights=weights)
 ss=read_json(p/mode/"scene.json");ee=next(e for e in ss["entities"] if e["id"]=="vehicle")
 err=np.linalg.norm(np.array(ee["entity_to_world"])[:,:3,3]-truepose[:,:3,3],axis=1)
 rows.append({"mode":mode,"mean_position_error":float(err.mean()),"max_position_error":float(err.max())})
passed=rows[2]["mean_position_error"]<.08
write_json(p/"report.json",{"rows":rows,"visible_faces_recovery_passed":passed,
 "controlled":"Only the surface-correspondence objective changes across SDF, hard first-hit and visible-face methods; identical masks, camera, shape, initialization and optimization budget.",
 "scope":"Known synthetic geometry; other objective terms disabled to isolate correspondence ambiguity"})
print(json.dumps(rows))
if not passed:raise AssertionError("Visible faces did not recover the known geometry")
