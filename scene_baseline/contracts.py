"""Explicit contracts for camera-Z geometry and agent-selected scene hypotheses."""
import numpy as np
from .storage import read_json,verify

def validate_geometry(a):
    for k in ("rgb","depth","confidence","K","c2w","times_s","original_to_model"):
        if k not in a: raise ValueError("Missing geometry array: "+k)
        if not np.isfinite(a[k]).all(): raise ValueError("Nonfinite "+k)
    rgb=a["rgb"];d=a["depth"];n,h,w=d.shape
    if rgb.shape!=(n,h,w,3) or a["confidence"].shape!=d.shape: raise ValueError("RGB/depth dimensions disagree")
    if np.any(d<=0) or np.any(a["confidence"]<0) or rgb.min()<0 or rgb.max()>1: raise ValueError("Invalid observation range")
    if a["K"].shape!=(n,3,3) or a["c2w"].shape!=(n,4,4): raise ValueError("Camera shape")
    if a["original_to_model"].shape!=(n,3,3): raise ValueError("Pixel transform shape")
    if a["times_s"].shape!=(n,) or (n>1 and np.any(np.diff(a["times_s"])<=0)): raise ValueError("Nonmonotonic times")
    R=a["c2w"][:,:3,:3]
    if not np.allclose(np.swapaxes(R,1,2)@R,np.eye(3),atol=2e-3) or not np.allclose(np.linalg.det(R),1,atol=2e-3): raise ValueError("Camera rotation not SO3")
    if not np.allclose(a["c2w"][:,3,:],[0,0,0,1],atol=1e-6): raise ValueError("Invalid c2w homogeneous row")
    if np.any(a["K"][:,0,0]<=0) or np.any(a["K"][:,1,1]<=0): raise ValueError("Nonpositive focal length")
    return n,h,w

def load_bundle(path):
    m=read_json(path)
    if m.get("schema")!="scene_baseline.geometry.v1": raise ValueError("Unknown geometry schema")
    p=verify(m["geometry"]);verify(m["source_rgb"]);verify(m["script"])
    with np.load(p,allow_pickle=False) as z:a={k:z[k] for k in z.files}
    validate_geometry(a)
    return m,a

def validate_hypothesis(doc):
    if doc.get("schema")!="scene_baseline.hypothesis.v1": raise ValueError("Hypothesis schema")
    if not doc.get("actor"): raise ValueError("Missing hypothesis actor")
    if not doc.get("geometry_manifest_sha256"): raise ValueError("Missing geometry binding")
    ids=[]
    for e in doc["entities"]:
        ids.append(e["id"])
        if e["motion"] not in ("static","rigid","deformable","unknown"): raise ValueError("Motion type")
        if not e.get("label") or not e.get("segmentation_prompt"): raise ValueError("Missing semantic label")
        if not 0<=e["confidence"]<=1: raise ValueError("Confidence")
        for p in e["parts"]:
            if p["kind"] not in ("box","cylinder","sphere","cone","plane"): raise ValueError("Primitive type")
    if len(ids)!=len(set(ids)): raise ValueError("Duplicate identity")
    return doc

def validate_masks(masks,ids,geometry):
    if masks.dtype!=np.bool_ or masks.shape!=(len(ids),*geometry["depth"].shape): raise ValueError("Masks must be boolean E,T,H,W on model pixels")
    if len(set(ids))!=len(ids): raise ValueError("Duplicate mask identity")
