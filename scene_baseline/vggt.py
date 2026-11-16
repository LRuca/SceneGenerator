"""One joint aggregation for depth, confidence and cameras; no point-head dependency."""
import os,sys,time
import numpy as np
from .storage import ROOT,BASE,inside,new_run,reference,write_json,utc
from .contracts import validate_geometry

def export(rgb_path,script_path,out,fps=16.,frame_count=49,resolution=518):
    import torch
    import torch.nn.functional as F
    gf_source=inside(os.environ.get("SCENEGEN_GF_SOURCE", BASE/"dependencies/GeometryForcing"))
    weights=inside(os.environ.get("SCENEGEN_VGGT_WEIGHTS", BASE/"weights/VGGT-1B"))
    sys.path.insert(0,str(gf_source))
    from external.vggt import VGGT
    from external.vggt.utils.pose_enc import pose_encoding_to_extri_intri
    torch.set_num_threads(8);torch.backends.cudnn.enabled=False
    src=inside(rgb_path);script_path=inside(script_path)
    source_refs={"source_rgb":reference(src),"script":reference(script_path)}
    a=np.load(src,allow_pickle=False)
    if a.ndim!=4 or a.shape[-1]!=3 or not np.isfinite(a).all() or a.min()<0 or a.max()>1:raise ValueError("Invalid RGB")
    if fps<=0 or not 1<=frame_count<=len(a) or resolution%14:raise ValueError("Invalid sample settings")
    idx=np.unique(np.linspace(0,len(a)-1,frame_count).round().astype(int))
    h0,w0=a.shape[1:3];h=resolution;w=round(w0/h0*h/14)*14
    x=F.interpolate(torch.from_numpy(a[idx]).permute(0,3,1,2).cuda(),size=(h,w),mode="bilinear",align_corners=False)[None]
    model=VGGT.from_pretrained(str(weights)).cuda().eval()
    torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
    with torch.inference_mode(),torch.autocast("cuda",dtype=torch.bfloat16):
        tokens,offset=model.shortcut_forward(x)
        with torch.autocast("cuda",enabled=False):
            pose=model.camera_head(tokens)[-1]
            depth,confidence=model.depth_head(tokens,images=x,patch_start_idx=offset)
            extr,K=pose_encoding_to_extri_intri(pose,(h,w))
    data={"rgb":x[0].permute(0,2,3,1).float().cpu().numpy(),"depth":depth[0,...,0].float().cpu().numpy(),
        "confidence":confidence[0].float().cpu().numpy(),"K":K[0].float().cpu().numpy(),"frame_indices":idx,
        "times_s":idx.astype(float)/fps}
    e=extr[0].float().cpu().numpy()
    w2c=np.tile(np.eye(4),(len(idx),1,1));w2c[:,:3,:4]=e
    data["c2w"]=np.linalg.inv(w2c)
    sx,sy=w/w0,h/h0
    A=np.array([[sx,0,(sx-1)/2],[0,sy,(sy-1)/2],[0,0,1.]])
    data["original_to_model"]=np.tile(A,(len(idx),1,1))
    validate_geometry(data)
    dest=new_run(out);np.savez_compressed(dest/"geometry.npz",**data)
    np.save(dest/"rgb.npy",data["rgb"]);np.save(dest/"depth.npy",data["depth"])
    manifest={"schema":"scene_baseline.geometry.v1","created_at":utc(),"geometry":reference(dest/"geometry.npz"),
        **source_refs,"model":"facebook/VGGT-1B","method":"same joint tokens, camera_head + depth_head",
        "model_weights":reference(weights/"model.safetensors"),
        "coordinate_convention":"OpenCV c2w, camera-Z depth, arbitrary common scale",
        "seconds":time.perf_counter()-start,"frames":len(idx),"fps":fps,"resolution_hw":[h,w],
        "peak_gpu_GiB":torch.cuda.max_memory_allocated()/2**30,
        "source_frame_indices":idx.tolist(),"camera_policy":"predicted_initialization_not_validated"}
    write_json(dest/"manifest.json",manifest)
    return manifest
