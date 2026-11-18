"""Shared conservative envelope calibration, with explicit coverage/precision tradeoffs."""
import copy
import numpy as np
import cv2
from .geometry import scene_parts,part_mesh
from .scene import load_scene,validate_scene
from .storage import read_json,verify,new_run,write_json,reference
def projected_mask(parts,K,c2w,H,W,eid):
    out=np.zeros((H,W),np.uint8)
    for p in parts:
        if p["entity_id"]!=eid:continue
        v,faces,_=part_mesh(p);v=(v-c2w[:3,3])@c2w[:3,:3]
        pix=v@K.T;uv=np.rint(pix[:,:2]/pix[:,2:].clip(1e-6)).astype(np.int32)
        valid=(v[faces,2]>0).all(-1)
        for face in np.asarray(faces)[valid]:cv2.fillConvexPoly(out,uv[face],1)
    return out.astype(bool)
def calibrate(scene_path,eid,out,target_coverage=.90,width=140):
    s,m,a=load_scene(scene_path);p=new_run(out)
    md=read_json(verify(s["masks_manifest"]))
    with np.load(verify(md["data"])) as z:masks=z["masks"][z["ids"].tolist().index(eid)]
    n,H,W=masks.shape;h=round(H/W*width);A=np.diag([width/W,h/H,1.])
    target=np.stack([cv2.resize(mm.astype(np.uint8),(width,h),interpolation=cv2.INTER_NEAREST)>0 for mm in masks])
    e=next(x for x in s["entities"] if x["id"]==eid);base=np.asarray(e["deformation"]["shared_size"])
    rows=[];best=None
    # Shared inflation only: never per-frame width pumping or guessed trajectories.
    for factor in [1.,1.15,1.3,1.5,1.75,2.]:
        for hr in [.08,.095,.11]:
            e["deformation"]["shared_size"]=(base*np.array([factor,1,factor])).tolist()
            e["deformation"]["head_radius_height_fraction"]=hr
            coverage=[];ious=[];precision=[]
            for t in range(n):
                pred=projected_mask(scene_parts(s,t),A@a["K"][t],a["c2w"][t],h,width,eid)
                hit=(pred&target[t]).sum();coverage.append(hit/max(target[t].sum(),1));precision.append(hit/max(pred.sum(),1));ious.append(hit/max((pred|target[t]).sum(),1))
            row=dict(width_depth_factor=factor,head_radius_height_fraction=hr,mean_coverage=float(np.mean(coverage)),min_coverage=float(min(coverage)),mean_precision=float(np.mean(precision)),mean_iou=float(np.mean(ious)))
            rows.append(row)
            # Meet explicit average coverage with least inflation; otherwise retain best achieved coverage.
            score=(0 if row["mean_coverage"]>=target_coverage else 1, factor**2*(hr/.08)**.5 if row["mean_coverage"]>=target_coverage else -row["mean_coverage"],-row["mean_iou"])
            if best is None or score<best[0]:best=(score,copy.deepcopy(e["deformation"]),row)
    e["deformation"]=best[1];e["envelope_calibration"]={"source_scene":reference(scene_path),"target_mean_silhouette_coverage":target_coverage,"selected":best[2],"scope":"same-stage 2D silhouette envelope, not amodal 3D truth; hidden thickness can be overestimated"}
    validate_scene(s,n);write_json(p/"scene.json",s);write_json(p/"report.json",dict(source=reference(scene_path),target_mean_coverage=target_coverage,selected=best[2],candidates=rows,all_frames_same_dimensions=True,guarantee="projected union coverage only; actual occlusion-aware raster metrics reported separately"))
    return s
