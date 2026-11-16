"""Isolated SAM3 worker. Real output identities remain hypotheses, never force-merged."""
import sys,time
from pathlib import Path
import numpy as np
from PIL import Image
from .storage import ROOT,inside,new_run,reference,write_json
from .contracts import load_bundle,validate_hypothesis
from .storage import read_json

def track(manifest,hypothesis,entity,checkpoint,out,max_frames=None):
    m,a=load_bundle(manifest);hyp=validate_hypothesis(read_json(hypothesis))
    if hyp["geometry_manifest_sha256"]!=reference(manifest)["sha256"]:raise ValueError("Hypothesis bundle mismatch")
    item=next(e for e in hyp["entities"] if e["id"]==entity)
    checkpoint=inside(checkpoint)
    if not checkpoint.is_file():raise FileNotFoundError("SAM3 local checkpoint missing")
    import torch
    from sam3.model_builder import build_sam3_video_predictor
    torch.set_num_threads(8)
    p=new_run(out);frames=p/"frames";frames.mkdir()
    n=len(a["depth"]) if max_frames is None else min(max_frames,len(a["depth"]))
    for t in range(n):Image.fromarray((a["rgb"][t]*255).astype("uint8")).save(frames/f"{t:05d}.jpg",quality=95)
    start=time.perf_counter();torch.cuda.reset_peak_memory_stats()
    predictor=build_sam3_video_predictor(checkpoint_path=str(checkpoint),gpus_to_use=[0],compile=False)
    session=None;instances={}
    try:
        session=predictor.handle_request({"type":"start_session","resource_path":str(frames)})["session_id"]
        predictor.handle_request({"type":"add_prompt","session_id":session,"frame_index":0,"text":item["segmentation_prompt"]})
        for response in predictor.handle_stream_request({"type":"propagate_in_video","session_id":session}):
            t=response["frame_index"];output=response["outputs"]
            ids=output["out_obj_ids"].tolist();masks=output["out_binary_masks"]
            if torch.is_tensor(masks):masks=masks.cpu().numpy()
            for k,oid in enumerate(ids):
                rec=instances.setdefault(int(oid),np.zeros(a["depth"].shape,dtype=bool))
                mask=np.asarray(masks[k],bool).squeeze()
                if mask.shape!=a["depth"].shape[1:]:raise ValueError("SAM mask changed model pixel dimensions")
                rec[t]=mask
            print("SAM3 frame",t,"instances",len(ids),flush=True)
    finally:
        if session is not None:predictor.handle_request({"type":"close_session","session_id":session})
        if hasattr(predictor,"shutdown"):predictor.shutdown()
    keys=sorted(instances)
    ids=[entity if len(keys)==1 else entity+":candidate_"+str(k) for k in keys]
    masks=np.stack([instances[k] for k in keys]) if keys else np.empty((0,*a["depth"].shape),bool)
    np.savez_compressed(p/"masks.npz",masks=masks,ids=np.asarray(ids,dtype="U80"),tested_frame_indices=np.arange(n))
    report={"schema":"scene_baseline.sam3_worker.v1","geometry_manifest":reference(manifest),"hypothesis":reference(hypothesis),
        "checkpoint":reference(checkpoint),"data":reference(p/"masks.npz"),"requested_entity":entity,"candidate_ids":ids,
        "tested_frames":n,"total_frames":len(a["depth"]),"seconds":time.perf_counter()-start,
        "peak_gpu_GiB":torch.cuda.max_memory_allocated()/2**30,
        "status":"full_clip_inference_completed_not_identity_verified" if n==len(a["depth"]) else "partial_clip_smoke_only",
        "identity_policy":"one SAM object may bind to requested entity; multiple SAM objects remain separate candidates",
        "positive_pixels_per_frame":masks.sum(axis=(2,3)).tolist()}
    write_json(p/"manifest.json",report)
    return report
