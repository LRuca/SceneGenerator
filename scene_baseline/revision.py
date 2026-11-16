"""Bounded agent corrections as new scene versions with explicit parent binding."""
import copy
import numpy as np
from .scene import load_scene,validate_scene
from .storage import read_json,reference,verify,new_run,write_json

def apply(scene_path,request_path,out):
    scene,m,a=load_scene(scene_path);request=read_json(request_path)
    if request.get("schema")!="scene_baseline.revision.v1":raise ValueError("Revision schema")
    if request["parent_scene"]!=reference(scene_path):raise ValueError("Revision parent changed")
    if not request.get("actor") or not request.get("reason"):raise ValueError("Revision actor/reason required")
    result=copy.deepcopy(scene);by_id={e["id"]:e for e in result["entities"]}
    for action in request["actions"]:
        e=by_id[action["entity_id"]]
        if action["type"]=="replace_parts":
            # New parts retain the shared entity motion; useful for agent topology refinement.
            e["parts"]=action["parts"]
        elif action["type"]=="set_renderable":
            flags=action["values"]
            if len(flags)!=len(a["depth"]) or any(type(x) is not bool for x in flags):raise ValueError("Visibility timeline")
            e["renderable"]=flags
        else:raise ValueError("Unsupported revision action")
    result["revision"]={"parent_scene":reference(scene_path),"request":reference(request_path)}
    result["stage"]="agent_revised_candidate_requires_refit"
    validate_scene(result,len(a["depth"]))
    p=new_run(out);write_json(p/"scene.json",result);write_json(p/"revision.json",request)
    return result
