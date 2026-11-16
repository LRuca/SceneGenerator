"""Known-truth synthetic experiment, including withheld spans and boundary extrapolation."""
import numpy as np,copy
from pathlib import Path
from scene_baseline.fixtures import create
from scene_baseline.storage import read_json,write_json,reference
from scene_baseline.persistence import apply,sample_entity,DEFAULT_POLICY
from scipy.spatial.transform import Rotation
out=Path("runs/persistence_synthetic_benchmark");out.mkdir()
f=create(out/"fixture",n=49,size=48);s=read_json(f/"truth_scene.json")
z=np.load(f/"masks.npz");masks=z["masks"].copy();masks[-1,:6]=False;masks[-1,19:29]=False;masks[-1,-7:]=False
np.savez_compressed(out/"masked.npz",masks=masks,ids=z["ids"])
md=read_json(f/"masks_manifest.json");md["data"]=reference(out/"masked.npz");write_json(out/"masks_manifest.json",md)
s["masks_manifest"]=reference(out/"masks_manifest.json")
truth=np.array(s["entities"][-1]["entity_to_world"]);rng=np.random.default_rng(803)
raw=truth.copy();raw[:,:3,3]+=rng.normal(0,.028,(49,3))
base=np.diag([1.,-1.,-1.]);yaw=np.linspace(-.12,.12,49)+rng.normal(0,.1,49)
raw[:,:3,:3]=base@Rotation.from_euler("y",yaw).as_matrix()
s["entities"][-1]["entity_to_world"]=raw.tolist();s["entities"][-1]["renderable"]=masks[-1].any((1,2)).tolist()
write_json(out/"source.json",s)
apply(out/"source.json","experiments/persistence_v1/policy.json",out/"persistent")
new=read_json(out/"persistent/scene.json");e=new["entities"][-1];pred=np.array(e["entity_to_world"]);t=np.arange(49)/8
observed=masks[-1].any((1,2));hidden=~observed
raw_error=float(np.sqrt(np.mean((raw[:,:3,3]-truth[:,:3,3])**2)))
hidden_error=float(np.sqrt(np.mean((pred[hidden,:3,3]-truth[hidden,:3,3])**2)))
visible_error=float(np.sqrt(np.mean((pred[observed,:3,3]-truth[observed,:3,3])**2)))
report=read_json(out/"persistent/report.json")
write_json(out/"benchmark.json",{"known_truth":reference(f/"truth_scene.json"),"source":reference(out/"source.json"),
 "noise_seed":803,"mask_hidden_frames":np.flatnonzero(hidden).tolist(),"raw_position_RMSE":raw_error,
 "persistent_observed_position_RMSE":visible_error,"persistent_hidden_position_RMSE":hidden_error,
 "motion":report["entities"]["vehicle"],"no_visibility_teleport":all(e["exists"]),
 "heldout_protocol":"hidden positions contain synthetic noise in source file but are excluded by masks before continuous fitting; no train on hidden samples",
 "physical_truth":"known synthetic trajectory and shared shape; independent from heldout generated RGB cases"})
assert all(e["exists"]) and hidden_error<.08
assert report["entities"]["vehicle"]["continuous_dynamics"]["acceleration_rms"]<report["entities"]["vehicle"]["raw_dynamics"]["acceleration_rms"]
print("Synthetic benchmark",hidden_error,visible_error,flush=True)
