"""Single-camera dynamic scene contracts and depth/mask-based initialization."""
import numpy as np
from scipy.spatial.transform import Rotation
from .storage import read_json,reference,verify,write_json,new_run
from .contracts import load_bundle,validate_hypothesis,validate_masks

def validate_scene(scene,n):
    if scene.get("schema")!="scene_baseline.scene.v1":raise ValueError("Scene schema")
    ids=[]
    for e in scene["entities"]:
        ids.append(e["id"])
        if e["motion"] not in ("static","rigid","deformable"):raise ValueError("Unsupported scene motion")
        if e["motion"]=="deformable":
            from .cage_v4 import validate
            validate(e["deformation"],n)
        pose=np.asarray(e["entity_to_world"],float)
        if pose.shape!=((1 if e["motion"]=="static" else n),4,4):raise ValueError("Scene pose timeline")
        if not np.isfinite(pose).all() or not np.allclose(pose[:,3,:],[0,0,0,1]):raise ValueError("Scene pose")
        r=pose[:,:3,:3]
        if not np.allclose(r.swapaxes(1,2)@r,np.eye(3),atol=1e-4) or not np.allclose(np.linalg.det(r),1,atol=1e-4):raise ValueError("Entity transform must be rigid")
        if len(e["renderable"])!=n:raise ValueError("Renderable timeline")
        if "state_source" in e:
            for key in ["exists","observation_available","observation_used","state_source"]:
                if len(e[key])!=n:raise ValueError("Persistent state timeline: "+key)
            for key in ["exists","observation_available","observation_used"]:
                if any(type(x) is not bool for x in e[key]):raise ValueError("Persistent state flags")
            if np.any(np.asarray(e["observation_used"])&~np.asarray(e["observation_available"])):raise ValueError("Used nonexistent observation")
            if e["renderable"]!=e["exists"]:raise ValueError("Persistence render gates must follow existence, not observation")
            if "continuous_motion" in e:
                from .persistence import sample_entity
                # Scene pose samples must be bound to authoritative continuous curves.
                knots=e["continuous_motion"]["position"]["times"]
                if not np.isfinite(knots).all() or np.any(np.diff(knots)<=0):raise ValueError("Invalid continuous state knots")

        if "fit_constraints" in e and type(e["fit_constraints"].get("lock_yaw",False)) is not bool:raise ValueError("Yaw lock must be boolean")
        if "landmarks" in e:
            lm=e["landmarks"];lp=np.asarray(lm["entity_points"],float);uv=np.asarray(lm["image_points"],float);v=np.asarray(lm["valid"])
            if lp.ndim!=2 or lp.shape[1]!=3 or not 1<=len(lp)<=128:raise ValueError("Landmark local shape")
            if uv.shape!=(n,len(lp),2) or v.shape!=(n,len(lp)) or v.dtype!=np.bool_:raise ValueError("Landmark observation shape")
            if not np.isfinite(lp).all() or not np.isfinite(uv).all():raise ValueError("Nonfinite landmark")
            if not 0<float(lm.get("weight",8.))<=100:raise ValueError("Landmark weight")

        for p in e["parts"]:
            m=np.asarray(p["canonical_to_entity"],float)
            if p["kind"] not in ("box","cylinder","sphere","cone","plane"):raise ValueError("Primitive kind")
            if m.shape!=(4,4) or not np.isfinite(m).all() or np.linalg.det(m[:3,:3])<=0:raise ValueError("Part transform")
            if not np.allclose(m[3],[0,0,0,1]):raise ValueError("Part homogeneous row")
    if len(ids)!=len(set(ids)):raise ValueError("Duplicate scene entity")
    return scene

def load_observations(manifest,hypothesis,masks_manifest):
    m,a=load_bundle(manifest);h=validate_hypothesis(read_json(hypothesis));mask_doc=read_json(masks_manifest)
    if h["geometry_manifest_sha256"]!=reference(manifest)["sha256"]:raise ValueError("Hypothesis geometry mismatch")
    if mask_doc["geometry_manifest"]!=reference(manifest) or mask_doc["hypothesis"]!=reference(hypothesis):raise ValueError("Masks from another geometry/hypothesis")
    with np.load(verify(mask_doc["data"]),allow_pickle=False) as z:masks=z["masks"];ids=z["ids"].tolist()
    validate_masks(masks,ids,a)
    if not ids:raise ValueError("No reviewed instance masks supplied")
    if set(ids)-{e["id"] for e in h["entities"]}:raise ValueError("Unreviewed mask candidate IDs")
    return m,a,h,mask_doc,masks,ids

def points_for_mask(a,mask,max_points=1200):
    result=[];per_time=[]
    for t in range(len(mask)):
        v,u=np.nonzero(mask[t])
        if not len(u):per_time.append(np.empty((0,3)));continue
        step=max(1,int(np.ceil(len(u)/max_points)));v,u=v[::step],u[::step]
        local=np.column_stack([u,v,np.ones(len(u))])@np.linalg.inv(a["K"][t]).T
        p=local*a["depth"][t,v,u,None]
        p=p@a["c2w"][t,:3,:3].T+a["c2w"][t,:3,3]
        per_time.append(p);result.append(p)
    if not result:raise ValueError("Entity has no observed depth")
    return np.concatenate(result),per_time

def initialize(manifest,hypothesis,masks_manifest,out):
    m,a,h,md,masks,ids=load_observations(manifest,hypothesis,masks_manifest)
    n=len(a["depth"]);up=-a["c2w"][:,:3,1].mean(0);up/=np.linalg.norm(up)
    entities={e["id"]:e for e in h["entities"]}
    floor_ids=[i for i in ids if any(p["kind"]=="plane" and p.get("role")=="ground" for p in entities[i]["parts"])]
    if floor_ids:
        points,_=points_for_mask(a,masks[ids.index(floor_ids[0])])
        centered=points-np.median(points,axis=0)
        # Iteratively trim plane residuals; retain a diagnostic rather than certify a ground plane.
        for _ in range(3):
            _,_,vt=np.linalg.svd(centered,full_matrices=False);normal=vt[-1]
            residual=np.abs(centered@normal)
            centered=centered[residual<=np.quantile(residual,.85)]
        up=normal*np.sign(normal@up or 1)
    right=a["c2w"][0,:3,0].copy();right-=up*(right@up);right/=np.linalg.norm(right)
    basis=np.stack([right,up,np.cross(right,up)],axis=1)
    output=[]
    for entity_id,mask in zip(ids,masks):
        e=entities[entity_id]
        if e["motion"] not in ("static","rigid"):raise ValueError("Unsupported motion needs agent revision: "+entity_id)
        if e["motion"]=="static" and len(e["parts"])==1 and e["parts"][0]["kind"]=="plane" and e["parts"][0].get("role") in {"wall","vertical_surface"}:
            from .surfaces_v2 import vertical_surface
            pose,matrix,report=vertical_surface(a,mask,up,thickness_ratio=.01)
            output.append({"id":entity_id,"label":e["label"],"motion":"static",
              "parts":[{"id":e["parts"][0]["id"],"kind":"box","role":"vertical_surface","canonical_to_entity":matrix.tolist()}],
              "entity_to_world":[pose.tolist()],"observed":mask.any(axis=(1,2)).tolist(),"renderable":[True]*n,
              "initialization":"role-aware measured vertical surface with explicit thickness prior","surface_prior":report})
            continue
        points,by_time=points_for_mask(a,mask);observed=mask.any(axis=(1,2))
        boxes=[];centers=[];yaws=[]
        times=[0] if e["motion"]=="static" else list(range(n))
        for t in times:
            p=points if e["motion"]=="static" else by_time[t]
            if not len(p):centers.append(None);boxes.append(None);yaws.append(None);continue
            local=p@basis
            if e["parts"][0]["kind"]=="plane":yaw=0.
            else:
                flat=local[:,[0,2]];_,vec=np.linalg.eigh(np.cov(flat.T))
                axis=vec[:,-1];yaw=float(np.arctan2(-axis[1],axis[0]))
                if yaws and yaws[-1] is not None:
                    yaw+=round((yaws[-1]-yaw)/np.pi)*np.pi
            R=basis@Rotation.from_euler("y",yaw).as_matrix()
            q=p@R;lo,hi=np.percentile(q,[2,98],axis=0)
            size=np.maximum(hi-lo,max(float(np.max(hi-lo))*.08,1e-3))
            if e["parts"][0]["kind"]=="plane":size[1]=max(float(np.max(size))*.002,1e-3)
            centers.append(((lo+hi)/2)@R.T);boxes.append(size);yaws.append(yaw)
        good=np.array([i for i,v in enumerate(centers) if v is not None])
        if not len(good):raise ValueError("No initialization support")
        ctr=np.stack([centers[i] for i in good]);yaw_values=np.unwrap(np.array([yaws[i] for i in good]))
        count=len(times)
        ctr=np.stack([np.interp(np.arange(count),good,ctr[:,k]) for k in range(3)],axis=1)
        yaw_values=np.interp(np.arange(count),good,yaw_values)
        size=np.median(np.stack([boxes[i] for i in good]),axis=0)
        parts=[]
        for part in e["parts"]:
            if len(e["parts"])>1 and ("relative_center" not in part or "relative_size" not in part):raise ValueError("Multi-part hypothesis needs relative_center/relative_size")
            rel_c=np.asarray(part.get("relative_center",[0,0,0]));rel_s=np.asarray(part.get("relative_size",[1,1,1]))
            if rel_c.shape!=(3,) or rel_s.shape!=(3,) or np.any(rel_s<=0):raise ValueError("Invalid part proportions")
            matrix=np.eye(4);matrix[:3,:3]=np.diag(size*rel_s);matrix[:3,3]=size*rel_c
            parts.append({"id":part["id"],"kind":part["kind"],"role":part.get("role"),"canonical_to_entity":matrix.tolist()})
        poses=np.tile(np.eye(4),(count,1,1))
        for t in range(count):poses[t,:3,:3]=basis@Rotation.from_euler("y",yaw_values[t]).as_matrix()
        poses[:,:3,3]=ctr
        renderable=np.ones(n,bool) if e["motion"]=="static" else observed.copy()
        if e["motion"]=="rigid":
            # Only fill bounded internal gaps; no extrapolated existence at clip boundaries.
            keys=np.flatnonzero(observed)
            for aa,bb in zip(keys[:-1],keys[1:]):
                if a["times_s"][bb]-a["times_s"][aa]<=.5:renderable[aa:bb+1]=True
        output.append({"id":entity_id,"label":e["label"],"motion":e["motion"],"parts":parts,
            "entity_to_world":poses.tolist(),"observed":observed.tolist(),"renderable":renderable.tolist(),
            "initialization":"masked_depth_quantiles_with_ground_normal; hidden shape is a prior"})
    scene={"schema":"scene_baseline.scene.v1","geometry_manifest":reference(manifest),"hypothesis":reference(hypothesis),
        "masks_manifest":reference(masks_manifest),"coordinate_frame":"VGGT world; OpenCV cameras; arbitrary shared scale",
        "ground_normal":up.tolist(),"entities":output,"relations":h.get("relations",[]),
        "stage":"initialization_not_validated","limitations":["ground normal is a fitted candidate","single-view hidden extent unobserved","no metric scale"]}
    validate_scene(scene,n);p=new_run(out);write_json(p/"scene.json",scene);return scene

def load_scene(path):
    scene=read_json(path);m,a=load_bundle(verify(scene["geometry_manifest"]))
    validate_scene(scene,len(a["depth"]))
    for e in scene["entities"]:
        if "continuous_motion" in e:
            from .persistence import sample_entity
            if not np.allclose(e["entity_to_world"],sample_entity(e,a["times_s"]),atol=1e-6):raise ValueError("Stale pose samples differ from continuous state")
    verify(scene["hypothesis"]);verify(scene["masks_manifest"])
    return scene,m,a
