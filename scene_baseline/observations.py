"""Explicit adoption of derived depth, preserving original evidence and camera provenance."""
import copy
import numpy as np
from .contracts import load_bundle,validate_geometry
from .storage import read_json,verify,reference,new_run,write_json,utc

def derive_depth(manifest,tool_result,out):
    m,a=load_bundle(manifest);result=read_json(tool_result)
    if result.get("schema")!="scene_baseline.tool_result.v1" or result.get("modality")!="depth":raise ValueError("Expected depth tool result")
    if result.get("usage")!="geometry_candidate":raise ValueError("Preview-only tool result cannot become geometry implicitly")
    if result.get("geometry_binding")!=reference(manifest):raise ValueError("Tool belongs to another geometry bundle")
    parent=np.load(verify(result["parent"]),allow_pickle=False)
    if parent.shape!=a["depth"].shape or not np.array_equal(parent,a["depth"]):raise ValueError("Tool input is not this bundle's depth")
    derived=np.load(verify(result["output"]),allow_pickle=False)
    if derived.shape!=a["depth"].shape:raise ValueError("Depth tool changed pixel/time coordinates")
    data={k:v.copy() for k,v in a.items()}
    data["raw_confidence"]=data["confidence"]
    data["depth"]=derived.astype("float32")
    # Raw confidence is not calibrated for a filtered depth map.
    data["confidence"]=np.ones_like(data["depth"])
    validate_geometry(data);p=new_run(out)
    np.savez_compressed(p/"geometry.npz",**data);np.save(p/"depth.npy",data["depth"]);np.save(p/"rgb.npy",data["rgb"])
    doc=copy.deepcopy(m);doc["geometry"]=reference(p/"geometry.npz");doc["created_at"]=utc()
    doc["derived_from"]=reference(manifest);doc["derivation"]=reference(tool_result)
    doc["method"]="explicit derived depth; K/c2w/time/pixels inherited without modification"
    doc["confidence_policy"]="uniform fit weights; original confidence retained as raw_confidence"
    write_json(p/"manifest.json",doc);return doc
