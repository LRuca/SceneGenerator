"""Audit correspondence ambiguity, frozen-variable ablations and point/proxy gauge."""
from pathlib import Path
import numpy as np
import matplotlib;matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import binary_erosion
from scene_baseline.storage import new_run,read_json,write_json,reference,verify,utc
from scene_baseline.physical_audit import audit
p=new_run("runs/visibility_audit_v3")
summary=[];fig,axes=plt.subplots(2,2,figsize=(12,8))
def samples(a,mask,e,t):
 pose=np.array(e["entity_to_world"])[0]
 mat=pose@np.array(e["parts"][0]["canonical_to_entity"]);inv=np.linalg.inv(mat)
 v,u=np.nonzero(binary_erosion(mask[t],iterations=2));v,u=v[::3],u[::3]
 ray=np.column_stack([u,v,np.ones(len(u))])@np.linalg.inv(a["K"][t]).T@a["c2w"][t,:3,:3].T
 org=a["c2w"][t,:3,3];o=inv[:3,:3]@org+inv[:3,3];d=ray@inv[:3,:3].T
 safe=np.where(np.abs(d)>1e-10,d,1e-10)
 aa=(-.5-o)/safe;bb=(.5-o)/safe
 near=np.minimum(aa,bb).max(1);far=np.maximum(aa,bb).min(1)
 dep=a["depth"][t,v,u];hit=(far>near)&(near>0)
 return near,far,dep,hit,np.column_stack([u,v])
for row,st in enumerate([20,50]):
 old_path=f"runs/pv2_courtyard_s{st}_pipeline_r2/joint_fit/scene.json"
 old=read_json(old_path);a=np.load(f"runs/pv1_courtyard_occlusion_s{st}_geometry/geometry.npz")
 mz=np.load(f"runs/pv1_courtyard_occlusion_s{st}_experiment/masks/masks.npz")
 mask=mz["masks"][mz["ids"].tolist().index("parked_truck")]
 case={"step":st,"variants":[]}
 original={e["id"]:e for e in old["entities"]}
 variants=[("before",old_path),("sdf_restart",f"runs/va3_courtyard_s{st}_blue_sdf_r2/scene.json"),
 ("first_hit",f"runs/va3_courtyard_s{st}_blue_entry_r2/scene.json"),
 ("visible_faces",f"runs/va3_courtyard_s{st}_blue_visible/scene.json")]
 for label,path in variants:
  s=read_json(path);by={e["id"]:e for e in s["entities"]};e=by["parked_truck"]
  assert s["geometry_manifest"]==old["geometry_manifest"]
  assert all(by[eid]==e0 for eid,e0 in original.items() if eid!="parked_truck"),"Uncontrolled dynamic geometry change"
  assert np.allclose(e["parts"][0]["canonical_to_entity"],original["parked_truck"]["parts"][0]["canonical_to_entity"],atol=1e-7)
  assert np.allclose(np.array(e["entity_to_world"])[0,:3,:3],np.array(original["parked_truck"]["entity_to_world"])[0,:3,:3],atol=1e-7)
  entries=[];exits=[];allhit=[]
  for t in range(0,len(mask),2):
   near,far,dep,hit,_=samples(a,mask,e,t)
   allhit.extend(hit.tolist());entries.extend(np.abs(dep[hit]-near[hit]).tolist());exits.extend(np.abs(dep[hit]-far[hit]).tolist())
  entries=np.array(entries);exits=np.array(exits)
  physical=audit(s);world_med=float(np.median(a["depth"]))
  r={"label":label,"scene":reference(path),"collision_frames":len(set(x["frame"] for x in physical["events"])),
   "max_overlap_bound":max([x["overlap_world_bound"] for x in physical["events"]],default=0),
   "remaining_frames":sorted(set(x["frame"] for x in physical["events"])),
   "ray_hit_fraction_on_target_mask":float(np.mean(allhit)),
   "median_entry_depth_error":float(np.median(entries)),"median_exit_depth_error":float(np.median(exits)),
   "median_entry_error_over_scene_depth":float(np.median(entries)/world_med),
   "exit_is_closer_fraction":float(np.mean(exits<entries)),
   "nonblue_entities_exactly_unchanged":True,"shape_and_orientation_unchanged":True}
  case["variants"].append(r)
 # Energy scan: same inputs, single controlled camera-Z displacement.
 for name,folder,col in [("Unsigned SDF",f"runs/va3_courtyard_s{st}_blue_sdf_r2","#ce5362"),
                         ("First hit",f"runs/va3_courtyard_s{st}_blue_entry_r2","#c99224"),
                         ("Visible faces",f"runs/va3_courtyard_s{st}_blue_visible","#278db0")]:
  ar=read_json(folder+"/audit.json");scan=ar["scan"]
  xx=np.array([x["offset_world"] for x in scan]);yy=np.array([x["loss"] for x in scan])
  axes[row,0].plot(xx,yy-yy.min(),label=name,color=col)
 axes[row,0].axvline(0,color="grey",ls=":");axes[row,0].set(title=f"Step {st}: fixed-shape depth scan",xlabel="Blue proxy displacement along camera Z (arbitrary units)",ylabel="Objective minus its own minimum")
 axes[row,0].legend(fontsize=8);axes[row,0].grid(alpha=.2)
 # One representative source pixel selected by median observed depth among rays hit by both variants.
 eb=original["parked_truck"];new=read_json(f"runs/va3_courtyard_s{st}_blue_visible/scene.json")
 en=next(e for e in new["entities"] if e["id"]=="parked_truck")
 bn,bf,dep,bh,uv=samples(a,mask,eb,12);nn,nf,_,nh,_=samples(a,mask,en,12)
 common=bh&nh;ids=np.flatnonzero(common);idx=ids[np.argmin(np.abs(dep[ids]-np.median(dep[ids])))]
 ax=axes[row,1]
 ax.plot([bn[idx],bf[idx]],[1,1],lw=12,solid_capstyle="butt",color="#ce5362",label="Before: occupied proxy")
 ax.plot([nn[idx],nf[idx]],[0,0],lw=12,solid_capstyle="butt",color="#278db0",label="After: occupied proxy")
 ax.axvline(dep[idx],color="#222222",ls="--",label="Observed camera-Z depth")
 ax.set(yticks=[0,1],yticklabels=["Visible-face fit","Before fix"],ylim=(-.8,1.8),
  xlabel="Depth along one camera ray (arbitrary units)",title=f"Step {st}, frame12, pixel {uv[idx].tolist()}")
 ax.legend(fontsize=8,loc="upper left");ax.grid(axis="x",alpha=.2)
 case["representative_ray"]={"frame":12,"uv":uv[idx].tolist(),"depth":float(dep[idx]),"before_entry":float(bn[idx]),"before_exit":float(bf[idx]),"after_entry":float(nn[idx]),"after_exit":float(nf[idx])}
 # Shared-body feature origin is not guaranteed to be the proxy geometry origin.
 lm=np.load(f"runs/pv2_courtyard_s{st}_pipeline_r2/sparse_fit/landmark_fit.npz")
 body=original["forklift"];q=lm["entity_points"];mat=np.array(body["parts"][0]["canonical_to_entity"]);inv=np.linalg.inv(mat)
 local=q@inv[:3,:3].T+inv[:3,3];outside=(np.abs(local)>.53).any(1)
 pose=np.array(body["entity_to_world"]);world=np.einsum("nij,pj->npi",pose[:,:3,:3],q)+pose[:,:3,3][:,None,:]
 delta=np.array([0,0,float(np.linalg.norm(mat[:3,2]))*.25])
 shifted_q=q-delta;shifted_c=pose[:,:3,3]+np.einsum("nij,j->ni",pose[:,:3,:3],delta)
 shifted_world=np.einsum("nij,pj->npi",pose[:,:3,:3],shifted_q)+shifted_c[:,None,:]
 case["point_proxy_consistency"]={"outside_proxy_fraction_at_3pct_tolerance":float(outside.mean()),
  "canonical_point_quantiles":np.percentile(local,[5,95],axis=0).tolist(),
  "gauge_test_delta":delta.tolist(),"max_world_point_change":float(np.abs(shifted_world-world).max()),
  "interpretation":"A shared local point shift and compensating pose shift preserve projected/3D feature points while moving the proxy. The existing anchor prior only weakly resolves this; no explicit point-to-proxy attachment is enforced."}
 summary.append(case)
fig.tight_layout();fig.savefig(p/"diagnostics.png",dpi=150);plt.close(fig)
tests=read_json("runs/visibility_v3_synthetic_r2/report.json")
# Frozen sources must still match; new audit work is outside both frozen baselines.
for freeze in ["experiments/persistence_v1/FREEZE.json","experiments/agent_mimic_v2/FREEZE_r2.json"]:
 for ref in read_json(freeze)["sources"]:verify(ref)
write_json(p/"report.json",{"created_at":utc(),"cases":summary,"synthetic":tests,
 "originals_preserved":True,"root_cause":"Unsigned nearest-surface loss permits visible measurements to fit the exit/back face. Instance surface loss can overwhelm globally averaged depth loss and trap optimization in a wrong near-camera minimum.",
 "first_hit_limit":"The naive hard ray-miss penalty caused optimizer failure in the controlled synthetic recovery test; it is retained as an ablation, not selected as the final correction.",
 "chosen_correction":"Closest camera-visible bounded face distance for convex box/slab proxies; same robust scaling and coefficient as the original surface term.",
 "scope_limit":"Visible-face correction is for boxes/slabs; compound self-occlusion and other primitive types need their own verified visibility treatment. No claim of fully solved dynamic 3D geometry.",
 "secondary_issue":"Sparse feature/pose fitting is not explicitly attached to proxy shape; low reprojection error is insufficient to validate coarse world occupancy.",
 "evidence_scale":"VGGT arbitrary units; cross-stage scalar geometry is not metric ground truth."})
print("SUMMARY",[(x["step"],[(r["label"],r["collision_frames"],round(r["median_entry_depth_error"],5),round(r["exit_is_closer_fraction"],3)) for r in x["variants"]]) for x in summary])
