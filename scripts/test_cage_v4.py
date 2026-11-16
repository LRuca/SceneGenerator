"""Known moving/deforming coarse person, renderer/state consistency and fit comparison."""
import copy,contextlib,json,argparse
import numpy as np
from scipy.spatial.transform import Rotation
from scene_baseline.fixtures import create
from scene_baseline.cage_v4 import make_state,world_mesh,local_vertices,torch_vertices,projector,validate
from scene_baseline.geometry import render_parts,scene_parts
from scene_baseline.fit_cage_v4 import fit
from scene_baseline.storage import new_run,read_json,write_json,reference
from scene_baseline.review_v4 import export

def run(out,steps=600):
 root=new_run(out);f=create(root/"fixture",n=17,size=112)
 with np.load(f/"geometry.npz") as z:a={k:z[k] for k in z.files}
 scene=read_json(f/"truth_scene.json");e=next(e for e in scene["entities"] if e["id"]=="vehicle")
 e.update(id="person",label="coarse person",motion="deformable")
 n=len(a["depth"]);tt=np.linspace(0,1,n);bend=np.zeros((n,5,2))
 profile=projector()@np.array([-.35,.10,.50,.10,-.35])
 bend[:,:,1]=np.sin(np.pi*tt)[:,None]*profile[None]*.17
 pose=np.tile(np.eye(4),(n,1,1));base=np.diag([1.,-1.,-1.])
 pose[:,:3,:3]=base@Rotation.from_euler("x",.20*np.sin(np.pi*tt)).as_matrix()
 pose[:,:3,3]=np.column_stack([-.15+.3*tt,np.full(n,.35),np.full(n,3.)])
 e["entity_to_world"]=pose.tolist();e["deformation"]=make_state([.60,1.8,.46],bend,["known_synthetic"]*n)
 e["parts"]=[{"id":"coarse_body","kind":"cylinder","canonical_to_entity":np.diag([.60,1.8,.46,1.]).tolist()}]
 scene["relations"]=[{"subject":"person","relation":"supported_by","object":"ground"}]
 depths=[];labels=[]
 for t in range(n):
  dd,ll=render_parts(scene_parts(scene,t),a["K"][t],a["c2w"][t],112,112);depths.append(dd);labels.append(ll)
 a["depth"]=np.asarray(depths,dtype="float32");labels=np.asarray(labels)
 palette=np.array([[.65,.6,.55],[.4,.45,.5],[.1,.3,.85]],np.float32);a["rgb"]=palette[labels]
 np.save(root/"rgb.npy",a["rgb"]);write_json(root/"script.json",{"prompt":"One coarse person bends at the waist while the camera slides.","source":"known synthetic coarse geometry, not real person reconstruction"})
 np.savez_compressed(root/"geometry.npz",**a)
 gm=read_json(f/"geometry_manifest.json");gm.update(geometry=reference(root/"geometry.npz"),source_rgb=reference(root/"rgb.npy"),script=reference(root/"script.json"));write_json(root/"manifest.json",gm)
 hyp=read_json(f/"hypothesis.json");hyp["geometry_manifest_sha256"]=reference(root/"manifest.json")["sha256"]
 he=next(x for x in hyp["entities"] if x["id"]=="vehicle");he.update(id="person",label="person",motion="deformable",segmentation_prompt="person");hyp["relations"]=scene["relations"];write_json(root/"hypothesis.json",hyp)
 masks=np.stack([labels==j for j in range(3)]);true_masks=masks.copy()
 # Missing middle observations test persistence, not measured invisible shape.
 masks[2,7:9]=False
 np.savez_compressed(root/"masks.npz",masks=masks,ids=np.array(["ground","wall","person"]))
 md=read_json(f/"masks_manifest.json");md.update(geometry_manifest=reference(root/"manifest.json"),hypothesis=reference(root/"hypothesis.json"),data=reference(root/"masks.npz"));write_json(root/"masks_manifest.json",md)
 scene.update(geometry_manifest=reference(root/"manifest.json"),hypothesis=reference(root/"hypothesis.json"),masks_manifest=reference(root/"masks_manifest.json"))
 write_json(root/"truth_scene.json",scene)
 initial=copy.deepcopy(scene);ie=next(x for x in initial["entities"] if x["id"]=="person")
 pp=np.asarray(ie["entity_to_world"]);pp[:,:3,3]+=[.035,.02,.06];ie["entity_to_world"]=pp.tolist()
 ie["deformation"]=make_state([.57,1.76,.44],np.zeros((n,5,2)),["initial_prior"]*n);write_json(root/"initial_scene.json",initial)
 import torch
 expected=local_vertices(e["deformation"],5);actual=torch_vertices(torch.tensor([.60,1.8,.46],dtype=torch.float64),torch.tensor(bend,dtype=torch.float64))[5].numpy()
 assert np.max(np.abs(expected-actual))<1e-10
 bad=copy.deepcopy(e["deformation"]);bad["bend_xz"][0][0][0]+=.1
 try:validate(bad,n)
 except ValueError:pass
 else:raise AssertionError("Noncanonical rigid component accepted as deformation")
 rows=[]
 for name,allow in [("rigid_tube",False),("deformable_tube",True)]:
  with (root/(name+".log")).open("x") as log,contextlib.redirect_stdout(log):
   rep=fit(root/"initial_scene.json","person",root/name,steps=steps,allow_bend=allow)
  fitted=read_json(root/name/"scene.json");fe=next(x for x in fitted["entities"] if x["id"]=="person")
  errors=[np.linalg.norm(world_mesh(fe,t)[0]-world_mesh(e,t)[0],axis=-1).mean() for t in range(n)]
  rows.append({"variant":name,"mean_world_vertex_error":float(np.mean(errors)),"max_bend":rep["max_bend_height_fraction"],
    "all_frames_renderable":all(fe["renderable"]),"missing_state_labels":fe["deformation"]["state_source"][7:9]})
 write_json(root/"report.json",{"rows":rows,"torch_numpy_geometry_max_difference":float(np.max(abs(expected-actual))),
   "scope":"known synthetic coarse tube; two masks missing; not anatomical shape recovery",
   "deformation_improves":rows[1]["mean_world_vertex_error"]<rows[0]["mean_world_vertex_error"]})
 export(root/"rigid_tube/scene.json",root/"deformable_tube/scene.json",root/"view",width=192)
 print(json.dumps(rows),flush=True)
if __name__=="__main__":
 ap=argparse.ArgumentParser();ap.add_argument("--out",required=True);ap.add_argument("--steps",type=int,default=600);a=ap.parse_args();run(a.out,a.steps)
