"""Continuous-shot entity persistence with robust C2 motion and explicit inference."""
import copy
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation
from scipy.ndimage import distance_transform_edt
from .scene import load_scene,validate_scene
from .storage import read_json,write_json,reference,verify,new_run
from .geometry import unit_mesh,transform_mesh,scene_parts,render_parts

DEFAULT_POLICY={
 "schema":"scene_baseline.persistence_policy.v1",
 "position_lambda_s4":.0025,"yaw_lambda_s4":.005,
 "camera_position_lambda_s4":.001,"camera_rotation_lambda_s4":.001,"smooth_camera":True,
 "irls_iterations":5,"huber_scale_mad":2.5,
 "observation_min_pixels":24,"mask_erosion_pixels":2,
 "boundary_weight":.35,"minimum_observations":4,
 "process_acceleration_shape_s2":.5,"orientation_growth_rad_s":.35,
 "holdout_start_fraction":.4,"holdout_fraction":.2,
 "render_width":256,"fit_steps":200,"fit_resolution":64,"fit_lr":.01,
 "static_orientation":"depth_PCA_unresolved_semantic_axis",
 "dynamic_shape":"one_box_no_instance_specific_topology_revision",
 "visibility_resolution":96,
 "assumptions":["continuous shot: rigid/static entities persist absent explicit lifespan events",
 "default boundary motion is constant-velocity continuation, not semantic certainty",
 "yaw-only ground-oriented rigid motion; optional support prior projects lowest surface to supporting plane",
 "fixed original VGGT camera estimates; camera jitter reported separately",
 "one latent track per reviewed SAM identity; no automatic merge or semantic direction claims"]}

def validate_policy(p):
 if p.get("schema")!=DEFAULT_POLICY["schema"] or set(p)!=set(DEFAULT_POLICY):raise ValueError("Persistence policy keys/schema changed")
 for k in ["position_lambda_s4","yaw_lambda_s4","camera_position_lambda_s4","camera_rotation_lambda_s4","huber_scale_mad","process_acceleration_shape_s2","orientation_growth_rad_s"]:
  if not np.isfinite(p[k]) or p[k]<=0:raise ValueError("Invalid policy "+k)
 if not 1<=p["irls_iterations"]<=20 or not 2<=p["minimum_observations"]<=50:raise ValueError("Invalid policy iterations/support")
 return p

def fit_curve(times,values,weights,lam,iterations=5,huber=2.5):
 """Natural cubic smoothing spline; constant-velocity continuation is C2 at boundaries."""
 t=np.asarray(times,float);y=np.asarray(values,float);w=np.asarray(weights,float)
 if t.ndim!=1 or len(t)<2 or np.any(np.diff(t)<=0) or not np.isfinite(t).all():raise ValueError("Curve times")
 if y.ndim==1:y=y[:,None]
 if y.shape[0]!=len(t) or w.shape!=(len(t),) or not np.isfinite(y).all() or not np.isfinite(w).all() or np.any(w<=0):raise ValueError("Curve observations")
 n=len(t);basis=CubicSpline(t,np.eye(n),bc_type="natural")
 # Exact Gaussian quadrature of squared linear acceleration on every interval.
 roots=np.array([-.5773502691896258,.5773502691896258])
 sample=((t[:-1]+t[1:])[:,None]/2+np.diff(t)[:,None]/2*roots).ravel()
 qw=np.repeat(np.diff(t)/2,2)
 B=basis(sample,2);Q=B.T@(qw[:,None]*B)/(t[-1]-t[0])
 base=w/w.sum();robust=np.ones(n)
 for _ in range(iterations):
  ww=base*robust
  A=np.diag(ww)+lam*Q+np.eye(n)*1e-12
  smooth=np.linalg.solve(A,ww[:,None]*y)
  residual=np.linalg.norm(smooth-y,axis=1)
  # Nonzero floor supports exact straight-line inputs.
  sigma=max(float(np.median(np.abs(residual-np.median(residual)))*1.4826),float(np.median(residual))*.25,1e-6)
  robust=np.minimum(1.,huber*sigma/np.maximum(residual,1e-12))
 return {"times":t.tolist(),"values":smooth.tolist(),"bc":"natural","boundary":"linear_C2","dimension":y.shape[1],
  "robust_weights":robust.tolist(),"lambda_s4":lam}

def evaluate_curve(curve,times,derivative=0):
 t=np.asarray(times,float);knots=np.asarray(curve["times"]);values=np.asarray(curve["values"])
 if derivative not in (0,1,2,3):raise ValueError("Derivative order")
 cs=CubicSpline(knots,values,bc_type="natural")
 x=np.clip(t,knots[0],knots[-1]);out=cs(x,derivative)
 for edge,mask in [(0,t<knots[0]),(-1,t>knots[-1])]:
  if not np.any(mask):continue
  if derivative==0:out[mask]=values[edge]+(t[mask]-knots[edge])[:,None]*cs(knots[edge],1)
  elif derivative==1:out[mask]=cs(knots[edge],1)
  else:out[mask]=0.
 return out

def effective_cameras(scene,a):
    if "continuous_camera" not in scene:return a["c2w"]
    m=scene["continuous_camera"];t=a["times_s"]
    p=evaluate_curve(m["position"],t)
    r=evaluate_curve(m["rotation_log"],t)
    R=Rotation.from_rotvec(r).as_matrix()@np.asarray(m["reference_rotation"])
    c=np.tile(np.eye(4),(len(t),1,1));c[:,:3,:3]=R;c[:,:3,3]=p
    return c


def _frame_basis(scene,a):
 up=np.asarray(scene["ground_normal"],float);up/=np.linalg.norm(up)
 right=a["c2w"][0,:3,0].copy();right-=up*(right@up);right/=np.linalg.norm(right)
 return np.stack([right,up,np.cross(right,up)],axis=1)

def sample_entity(e,times):
 """Evaluate authoritative continuous state; support changes geometry, never masks."""
 if "continuous_motion" not in e:raise ValueError("No continuous motion")
 m=e["continuous_motion"];t=np.asarray(times,float);pos=evaluate_curve(m["position"],t)
 yaw=evaluate_curve(m["yaw"],t)[:,0];basis=np.asarray(m["basis"])
 rotation=basis@Rotation.from_euler("y",yaw).as_matrix()
 support=m.get("support")
 if support:
  up=np.asarray(support["normal"]);target=float(support["plane_offset"])
  local=np.asarray(m["support_vertices"])
  world=np.einsum("nij,pj->npi",rotation,local)
  bottom=np.min(world@up,axis=1)
  pos+=((target-bottom-pos@up)[:,None]*up)
 pose=np.tile(np.eye(4),(len(t),1,1));pose[:,:3,:3]=rotation;pose[:,:3,3]=pos
 return pose

def _proxy_vertices(e):
 result=[]
 for p in e["parts"]:
  result.append(transform_mesh(unit_mesh("box" if p["kind"]=="plane" else p["kind"]),p["canonical_to_entity"])[0])
 return np.concatenate(result)

def dynamics(times,pose):
 t=np.asarray(times);p=np.asarray(pose)
 if len(t)<3:return {"acceleration_rms":0.,"angular_acceleration_rms":0.}
 velocity=np.diff(p[:,:3,3],axis=0)/np.diff(t)[:,None]
 acc=np.diff(velocity,axis=0)/((np.diff(t)[:-1]+np.diff(t)[1:])/2)[:,None]
 # Rotation increments as vectors in a fixed world basis.
 omega=Rotation.from_matrix(p[1:,:3,:3]@p[:-1,:3,:3].transpose(0,2,1)).as_rotvec()/np.diff(t)[:,None]
 aa=np.diff(omega,axis=0)/((np.diff(t)[:-1]+np.diff(t)[1:])/2)[:,None]
 return {"acceleration_rms":float(np.sqrt(np.mean(acc**2))),"angular_acceleration_rms":float(np.sqrt(np.mean(aa**2))),
  "max_speed":float(np.linalg.norm(velocity,axis=1).max()),"max_angular_speed":float(np.linalg.norm(omega,axis=1).max())}

def _visibility(scene,a,e,index,resolution):
 n=len(a["depth"]);h,w=a["depth"].shape[1:];scale=resolution/w
 camera=effective_cameras(scene,a)
 A=np.array([[scale,0,(scale-1)/2],[0,scale,(scale-1)/2],[0,0,1.]])
 states=[];ratios=[]
 for t in range(n):
  if not e["exists"][t]:states.append("outside_lifespan");ratios.append(0.);continue
  parts=scene_parts(scene,t);K=A@a["K"][t]
  own=[p for p in parts if p["entity_id"]==e["id"]]
  solo,_=render_parts(own,K,camera[t],resolution,resolution)
  whole,lab=render_parts(parts,K,camera[t],resolution,resolution)
  amodal=np.isfinite(solo);count=int(amodal.sum())
  win=np.zeros_like(amodal)
  for j,p in enumerate(parts):
   if p["entity_id"]==e["id"]:win|=lab==j
  ratio=float(win.sum()/count) if count else 0.
  ratios.append(ratio)
  if not count:state="out_of_view_estimated"
  elif ratio<.1:state="occluded_estimated"
  elif e["observation_available"][t]:state="observed"
  else:state="visible_without_detection"
  states.append(state)
 return states,ratios

def apply(scene_path,policy_path,out,withheld=None):
 """No code path selects parameters based on a validation-case result."""
 policy=validate_policy(read_json(policy_path));scene,m,a=load_scene(scene_path)
 result=copy.deepcopy(scene);p=new_run(out);n=len(a["depth"]);times=a["times_s"]
 maskdoc=read_json(verify(scene["masks_manifest"]))
 with np.load(verify(maskdoc["data"]),allow_pickle=False) as z:masks=z["masks"];ids=z["ids"].tolist()
 basis=_frame_basis(scene,a);up=basis[:,1]
 if policy["smooth_camera"]:
  reference_R=a["c2w"][0,:3,:3]
  log=Rotation.from_matrix(a["c2w"][:,:3,:3]@reference_R.T).as_rotvec()
  if np.max(np.linalg.norm(log,axis=1))>2.8:raise ValueError("Camera rotation crosses log-map range; unsupported in frozen protocol")
  result["continuous_camera"]={
   "position":fit_curve(times,a["c2w"][:,:3,3],np.ones(n),policy["camera_position_lambda_s4"],policy["irls_iterations"],policy["huber_scale_mad"]),
   "rotation_log":fit_curve(times,log,np.ones(n),policy["camera_rotation_lambda_s4"],policy["irls_iterations"],policy["huber_scale_mad"]),
   "reference_rotation":reference_R.tolist(),"source":"smoothed estimate of original VGGT cameras; original observation bundle unchanged"}

 byid={e["id"]:e for e in result["entities"]}
 hold=np.zeros(n,bool) if withheld is None else np.asarray(withheld,bool)
 if hold.shape!=(n,):raise ValueError("Holdout timeline")
 reports={};heldout=[]
 for e in result["entities"]:
  ma=masks[ids.index(e["id"])];area=ma.sum((1,2));observed=area>=policy["observation_min_pixels"]
  train=observed&~hold if e["motion"]=="rigid" else observed.copy()
  events=e.get("lifespan_events",{})
  start=float(events.get("birth_s",times[0]));end=float(events.get("death_s",times[-1]))
  if end<start:raise ValueError("Invalid lifespan")
  exists=(times>=start)&(times<=end)
  train&=exists
  e["exists"]=exists.tolist();e["renderable"]=exists.tolist()
  e["observation_available"]=observed.tolist();e["observation_used"]=train.tolist()
  e["existence_source"]="explicit_lifespan" if events else "continuous_shot_persistence_prior"
  e["observation_geometry_source"]=scene["geometry_manifest"]
  if e["motion"]=="static":
   e["state_source"]=["static_shared_estimate" if o else "inferred_static_persistence" for o in observed]
   e["uncertainty_proxy"]={"position_radius":np.full(n,.01).tolist(),"units":"world arbitrary; heuristic not calibrated"}
   continue
  good=np.flatnonzero(train&exists)
  if len(good)<policy["minimum_observations"]:raise ValueError("Insufficient reliable observations for "+e["id"])
  original=np.asarray(e["entity_to_world"]);raw=original[good,:3,3]
  # Quality weights use mask support and predicted confidence, not disappearance gates.
  confidence=np.array([np.median(a["confidence"][t][ma[t]]) for t in good])
  weights=np.sqrt(area[good]/max(np.median(area[good]),1))*np.sqrt(confidence/max(np.median(confidence),1e-6))
  H,W=ma.shape[1:]
  boundary=np.array([ma[t,0].any() or ma[t,-1].any() or ma[t,:,0].any() or ma[t,:,-1].any() for t in good])
  weights*=np.where(boundary,policy["boundary_weight"],1.)
  weights=np.clip(weights,.05,4.)
  # PCA boxes have pi ambiguity; semantic axes, if supplied, retain 2pi direction.
  rel=basis.T@original[good,:3,:3]
  y=np.arctan2(rel[:,0,2],rel[:,0,0])
  semantic="landmarks" in e or e.get("semantic_front") is not None
  y=np.unwrap(y if semantic else 2*y)/(1 if semantic else 2)
  pos=fit_curve(times[good],raw,weights,policy["position_lambda_s4"],policy["irls_iterations"],policy["huber_scale_mad"])
  yaw=fit_curve(times[good],y,weights,policy["yaw_lambda_s4"],policy["irls_iterations"],policy["huber_scale_mad"])
  motion={"position":pos,"yaw":yaw,"basis":basis.tolist(),"time_unit":"seconds","support_vertices":_proxy_vertices(e).tolist()}
  for r in scene.get("relations",[]):
   if r.get("relation")=="supported_by" and r.get("subject")==e["id"] and r.get("object") in byid:
    parent=byid[r["object"]]
    if parent["motion"]=="static" and all(x["kind"]=="plane" for x in parent["parts"]):
     plane=np.asarray(parent["entity_to_world"])[0,:3,3]
     motion["support"]={"normal":up.tolist(),"plane_offset":float(plane@up),"entity_id":parent["id"],"source":"agent_declared_support_prior"}
  e["continuous_motion"]=motion;e["entity_to_world"]=sample_entity(e,times).tolist()
  source=[]
  for t in range(n):
   if not exists[t]:s="outside_lifespan"
   elif train[t]:s="observation_supported_smoothed"
   elif good[0]<t<good[-1]:s="inferred_interpolation"
   elif hold[t] and observed[t]:s="inferred_withheld_extrapolation"
   else:s="inferred_extrapolation"
   source.append(s)
  e["state_source"]=source
  pred=np.asarray(e["entity_to_world"]);res=np.linalg.norm(pred[good,:3,3]-raw,axis=1)
  verts=_proxy_vertices(e);extent=max(float(np.linalg.norm(verts.max(0)-verts.min(0))),1e-4)
  base=max(float(np.median(res)),extent*.03)
  distance=np.min(np.abs(times[:,None]-times[good][None]),axis=1)
  radius=base+policy["process_acceleration_shape_s2"]*extent*distance**2/2
  # Uncertainty is heuristic and explicitly grows away from evidence.
  e["uncertainty_proxy"]={"position_radius":radius.tolist(),"yaw_radius_rad":(.03+policy["orientation_growth_rad_s"]*distance).tolist(),
    "distance_to_evidence_s":distance.tolist(),"kind":"heuristic_radius_not_calibrated_probability"}
  e["evidence_summary"]={"used_frames":good.tolist(),"available_frames":np.flatnonzero(observed).tolist(),
    "boundary_prior":"constant velocity; alternatives possible","semantic_front_resolved":semantic}
  dense=np.linspace(times[0],times[-1],(n-1)*8+1);densepose=sample_entity(e,dense)
  support_error=None
  if motion.get("support"):
   offset=motion["support"]["plane_offset"]
   world=np.einsum("nij,pj->npi",densepose[:,:3,:3],verts)+densepose[:,:3,3][:,None,:]
   support_error=float(np.max(np.abs(np.min(world@up,axis=1)-offset)))
  withheld_good=np.flatnonzero(hold&observed)
  report={"raw_dynamics":dynamics(times[good],original[good]),"continuous_dynamics":dynamics(times[good],pred[good]),
    "dense_dynamics":dynamics(dense,densepose),"support_max_abs_error":support_error,
    "observation_fit_median_position_error":float(np.median(res)),"shape_extent":extent,
    "lifespan_frames":int(exists.sum()),"observed_frames":int(observed.sum()),"used_frames":len(good)}
  if len(withheld_good):
   # Pseudo-holdout only if input scene was already jointly fit; must be labeled in report.
   report["withheld_reference_position_error"]=float(np.mean(np.linalg.norm(pred[withheld_good,:3,3]-original[withheld_good,:3,3],axis=1)))
  reports[e["id"]]=report
 result["state_semantics"]="persistent_scene.v1"
 result["persistence"]={"source_scene":reference(scene_path),"policy":reference(policy_path),
  "shot_scope":"one continuous shot","hidden_state":"inferred not measured","withheld_frames":np.flatnonzero(hold).tolist()}
 # Geometric occlusion/out-of-view diagnosis is separated from semantic existence.
 for i,e in enumerate(result["entities"]):
  e["visibility_state"],e["visible_fraction_estimate"]=_visibility(result,a,e,i,policy["visibility_resolution"])
 validate_scene(result,n);write_json(p/"scene.json",result)
 write_json(p/"report.json",{"entities":reports,"policy":reference(policy_path),"source_scene":reference(scene_path),
  "camera":{"observation_bundle":"original VGGT K/c2w unchanged","output_estimate":"separate continuous camera state" if policy["smooth_camera"] else "original","raw_dynamics":dynamics(times,a["c2w"]),"output_dynamics":dynamics(times,effective_cameras(result,a))},"visibility":"coarse current geometry estimate, not occlusion truth",
  "uncertainty":"heuristic, uncalibrated; can be too small for a wrong model",
  "withheld_warning":"If source geometry/scene used withheld frames this is not independent holdout; use masked-scene initialization for genuine downstream holdout"})
 return result
