"""Known synthetic moving-object/static-background data for integration tests."""
import numpy as np
from scipy.spatial.transform import Rotation
from .storage import new_run,reference,write_json
from .geometry import render_parts,scene_parts
from .scene import validate_scene

def create(out,n=4,size=48):
    p=new_run(out);K=np.tile(np.array([[size*1.1,0,size/2],[0,size*1.1,size/2],[0,0,1.]]),(n,1,1))
    cameras=np.tile(np.eye(4),(n,1,1));cameras[:,0,3]=np.linspace(-.12,.12,n)
    base=np.diag([1.,-1.,-1.])
    def part(kind,size):
        M=np.eye(4);M[:3,:3]=np.diag(size);return {"id":"body","kind":kind,"canonical_to_entity":M.tolist()}
    def entity(eid,motion,center,parts):
        N=1 if motion=="static" else n;poses=np.tile(np.eye(4),(N,1,1))
        poses[:,:3,:3]=base;poses[:,:3,3]=center
        if motion=="rigid":
            poses[:,0,3]=np.linspace(-.25,.25,n)
            for t,y in enumerate(np.linspace(-.12,.12,n)):poses[t,:3,:3]=base@Rotation.from_euler("y",y).as_matrix()
        return {"id":eid,"label":eid,"motion":motion,"parts":parts,"entity_to_world":poses.tolist(),"observed":[True]*n,"renderable":[True]*n}
    entities=[entity("ground","static",[0,1.3,3],[part("plane",[12,.01,12])]),
        entity("wall","static",[0,0,6],[part("box",[12,12,.12])]),
        entity("vehicle","rigid",[0,.85,3],[part("box",[1.1,.9,.6])])]
    minimal={"entities":entities}
    depths=[];labels=[]
    for t in range(n):
        d,l=render_parts(scene_parts(minimal,t),K[t],cameras[t],size,size)
        if not np.isfinite(d).all():raise ValueError("Synthetic background must cover image")
        depths.append(d);labels.append(l)
    depth=np.stack(depths).astype("float32");labels=np.stack(labels)
    palette=np.array([[.6,.5,.4],[.25,.4,.6],[.9,.12,.08]],dtype="float32")
    rgb=palette[labels];np.save(p/"rgb.npy",rgb)
    write_json(p/"script.json",{"prompt":"One rigid red vehicle moves while a camera slides; ground and wall are stationary.","source":"known synthetic test"})
    np.savez_compressed(p/"geometry.npz",rgb=rgb,depth=depth,confidence=np.ones_like(depth)*2,K=K,c2w=cameras,
        times_s=np.arange(n)/8.,frame_indices=np.arange(n),original_to_model=np.tile(np.eye(3),(n,1,1)))
    manifest={"schema":"scene_baseline.geometry.v1","geometry":reference(p/"geometry.npz"),"source_rgb":reference(p/"rgb.npy"),
        "script":reference(p/"script.json"),"fps":8,"model":"known_synthetic_fixture","camera_policy":"known_synthetic_fixture"}
    write_json(p/"geometry_manifest.json",manifest)
    hyp={"schema":"scene_baseline.hypothesis.v1","actor":"synthetic_fixture_ground_truth",
        "geometry_manifest_sha256":reference(p/"geometry_manifest.json")["sha256"],"entities":[
            {"id":e["id"],"label":e["label"],"motion":e["motion"],"segmentation_prompt":e["label"],"confidence":1.,
                "parts":[{"id":"body","kind":e["parts"][0]["kind"],"role":"ground" if e["id"]=="ground" else "body"}]} for e in entities],
        "relations":[{"subject":"vehicle","relation":"supported_by","object":"ground"}],"tool_requests":[]}
    write_json(p/"hypothesis.json",hyp)
    masks=np.stack([labels==i for i in range(len(entities))]);np.savez_compressed(p/"masks.npz",masks=masks,ids=np.array([e["id"] for e in entities]))
    md={"schema":"scene_baseline.masks.v1","geometry_manifest":reference(p/"geometry_manifest.json"),"hypothesis":reference(p/"hypothesis.json"),
        "data":reference(p/"masks.npz")}
    write_json(p/"masks_manifest.json",md)
    scene={"schema":"scene_baseline.scene.v1","geometry_manifest":reference(p/"geometry_manifest.json"),"hypothesis":reference(p/"hypothesis.json"),
        "masks_manifest":reference(p/"masks_manifest.json"),"entities":entities,"ground_normal":[0,-1,0],
        "relations":hyp["relations"],"stage":"synthetic_known_scene","limitations":[]}
    validate_scene(scene,n);write_json(p/"truth_scene.json",scene)
    import copy
    perturbed=copy.deepcopy(scene)
    for e in perturbed["entities"]:
        if e["id"]=="vehicle":
            poses=np.asarray(e["entity_to_world"]);poses[:,:3,3]+=[.09,.04,.12];e["entity_to_world"]=poses.tolist()
    perturbed["stage"]="perturbed_synthetic_initialization";write_json(p/"initial_scene.json",perturbed)
    return p
