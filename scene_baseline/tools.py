"""Allowlisted processing tools. Every call creates a provenance-bound new artifact."""
from dataclasses import dataclass,asdict
import numpy as np,cv2,time,platform
from scipy.ndimage import median_filter
from .storage import reference,verify,write_json,new_run,utc,read_json

@dataclass(frozen=True)
class ToolSpec:
    id:str
    modality:str
    description:str
    parameters:dict
    geometry_effect:str
    caution:str=""
    version:str="1"
    output_type:str="array"
    coordinate_effect:str="identity"

_REGISTRY={}
def register(spec,fn):
    if spec.id in _REGISTRY:raise ValueError("Duplicate tool")
    _REGISTRY[spec.id]=(spec,fn)

def catalog():
    return [asdict(s) for s,_ in _REGISTRY.values()]

register(ToolSpec("rgb.bilateral","rgb","Spatial edge-preserving denoise",
    {"sigma_color":{"default":18.,"min":0.1,"max":80.},"sigma_space":{"default":3.,"min":0.1,"max":10.}},
    "requires_new_vggt_if_used_as_geometry_input"),
    lambda a,p:np.stack([cv2.bilateralFilter((x*255).round().astype("uint8"),5,p["sigma_color"],p["sigma_space"])/255. for x in a]).astype("float32"))
register(ToolSpec("rgb.unsharp","rgb","Unsharp masking",
    {"amount":{"default":0.7,"min":0.,"max":2.},"sigma":{"default":1.4,"min":0.1,"max":4.}},
    "requires_new_vggt_if_used_as_geometry_input","Can amplify invented edges; sharpness is not accuracy"),
    lambda a,p:np.stack([np.clip(x*(1+p["amount"])-cv2.GaussianBlur(x,(0,0),p["sigma"])*p["amount"],0,1) for x in a]))
register(ToolSpec("rgb.temporal_median","rgb","Uncompensated 3-frame temporal median",{},
    "requires_new_vggt_if_used_as_geometry_input","Experimental: camera/object motion can produce ghosts. Never selected automatically"),
    lambda a,p:median_filter(a,size=(3,1,1,1),mode="nearest"))
register(ToolSpec("depth.spatial_log_median","depth","3x3 spatial median in log depth",{},
    "derived_depth_fixed_camera_geometry_changed","Can erase thin structures and cross instance boundaries; retain raw evidence"),
    lambda a,p:np.exp(median_filter(np.log(a),size=(1,3,3),mode="nearest")).astype("float32"))

def invoke(request_path,out):
    request=read_json(request_path);spec,fn=_REGISTRY[request["tool"]]
    if request["modality"]!=spec.modality:raise ValueError("Tool modality mismatch")
    if not request.get("reason") or not request.get("actor"):raise ValueError("Tool request needs actor and reason")
    params=request.get("parameters",{})
    if set(params)-set(spec.parameters):raise ValueError("Unknown tool parameter")
    values={}
    for k,v in spec.parameters.items():
        x=float(params.get(k,v["default"]))
        if not np.isfinite(x) or not v["min"]<=x<=v["max"]:raise ValueError("Tool parameter out of range")
        values[k]=x
    usage=request.get("usage","agent_preview")
    if usage not in ("agent_preview","geometry_candidate"):raise ValueError("Unsupported observation role")
    if request.get("geometry_manifest"):verify(request["geometry_manifest"])
    inp=verify(request["input"]);a=np.load(inp,allow_pickle=False)
    shape=a.shape
    if not np.isfinite(a).all():raise ValueError("Nonfinite tool input")
    if spec.modality=="rgb" and (a.ndim!=4 or a.shape[-1]!=3 or a.min()<0 or a.max()>1):raise ValueError("Expected T,H,W,3 RGB in [0,1]")
    if spec.modality=="depth" and (a.ndim!=3 or a.min()<=0):raise ValueError("Expected positive camera-Z depth T,H,W")
    start=time.perf_counter();result=fn(a,values)
    if spec.output_type=="array" and (result.shape!=shape or not np.isfinite(result).all()):raise ValueError("Tool violated output contract")
    dest=new_run(out)
    if spec.output_type=="array":
        output=dest/"array.npy";np.save(output,result.astype("float32"),allow_pickle=False)
    else:
        output=dest/"report.json";write_json(output,result)
    record={"schema":"scene_baseline.tool_result.v1","tool":asdict(spec),"parameters":values,
        "request":reference(request_path),"parent":request["input"],"output":reference(output),
        "actor":request["actor"],"reason":request["reason"],"created_at":utc(),
        "coordinate_effect":spec.coordinate_effect,"modality":spec.modality,"shape":list(shape),
        "usage":usage,"geometry_binding":request.get("geometry_manifest"),
        "runtime_seconds":time.perf_counter()-start,"software":{"python":platform.python_version(),"numpy":np.__version__,"opencv":cv2.__version__},
        "geometry_effect":spec.geometry_effect}
    write_json(dest/"result.json",record)
    return record

def quality(array,modality):
    if modality=="rgb":
        g=np.stack([cv2.cvtColor(x.astype("float32"),cv2.COLOR_RGB2GRAY) for x in array])
        sharp=[float(cv2.Laplacian(x,cv2.CV_32F).var()) for x in g]
        return {"laplacian_variance":sharp,"mean_absolute_adjacent_frame_difference":float(np.abs(np.diff(g,axis=0)).mean()) if len(g)>1 else None,
            "interpretation":"Sharpness and temporal difference are diagnostics, not fidelity/flicker scores; motion also causes differences."}
    if modality!="depth":raise ValueError("Unknown modality")
    return {"positive_fraction":float((array>0).mean()),"finite_fraction":float(np.isfinite(array).mean()),
        "quantiles":np.percentile(array,[0,1,50,99,100]).tolist(),"per_frame_median":np.median(array,axis=(1,2)).tolist(),
        "interpretation":"Temporal changes can reflect camera/object motion; do not automatically normalize them away."}

register(ToolSpec("rgb.inspect_quality","rgb","Inspect sharpness and temporal-change proxies",{},"no_geometry_change",output_type="report"),lambda a,p:quality(a,"rgb"))
register(ToolSpec("depth.inspect_quality","depth","Inspect depth validity and distribution",{},"no_geometry_change",output_type="report"),lambda a,p:quality(a,"depth"))
