"""Visible-mask/3D ordering diagnostics and explicit agent round acceptance."""
from pathlib import Path
import numpy as np
from scene_baseline.storage import read_json,write_json,reference
from scene_baseline.scene import load_scene
from scene_baseline.geometry import scene_parts,render_parts
records={}
for st in [20,50]:
 history={}
 for rd in [0,1,2]:
  m=read_json(f"runs/two_cars_s{st}_delivery_r{rd}/metrics.json")
  d={"full_image":m["summary"]}
  for eid in ["near_truck","far_truck"]:
   rows=[f["entities"][eid] for f in m["frames"] if f["entities"][eid]["target_pixels"]>0]
   d[eid]={"mean_visible_frame_iou":float(np.mean([r["visible_mask_iou"] for r in rows])),
    "mean_depth_error_on_target":float(np.mean([r["depth_on_target"]["capped_relative_error_with_misses"] for r in rows]))}
  history[str(rd)]=d
 scene,m,a=load_scene(f"runs/two_cars_s{st}_fit_r2/scene.json")
 maskdoc=read_json(f"runs/two_cars_s{st}_masks/manifest.json");z=np.load(maskdoc["data"]["path"])
 masks={eid:ma for eid,ma in zip(z["ids"],z["masks"])}
 both=np.flatnonzero(masks["near_truck"].any((1,2))&masks["far_truck"].any((1,2)))
 ordering=[]
 for t in both:
  target={eid:float(np.median(a["depth"][t][masks[eid][t]])) for eid in ["near_truck","far_truck"]}
  width=128;scale=width/a["depth"].shape[2];A=np.array([[scale,0,(scale-1)/2],[0,scale,(scale-1)/2],[0,0,1]])
  K=A@a["K"][t];parts=scene_parts(scene,int(t))
  dn,_=render_parts([x for x in parts if x["entity_id"]=="near_truck"],K,a["c2w"][t],width,width)
  df,_=render_parts([x for x in parts if x["entity_id"]=="far_truck"],K,a["c2w"][t],width,width)
  overlap=np.isfinite(dn)&np.isfinite(df);count=int(overlap.sum())
  ordering.append({"frame":int(t),"median_input_depth":target,"near_input_depth_is_smaller":target["near_truck"]<target["far_truck"],
   "projected_amodal_overlap_pixels":count,"near_geometry_wins_fraction_on_overlap":float(np.mean(dn[overlap]<df[overlap])) if count else None})
 # Actual acceptance of local-refit result, no silent global switch.
 assert history["2"]["near_truck"]["mean_visible_frame_iou"]>=history["0"]["near_truck"]["mean_visible_frame_iou"]-.005
 assert history["2"]["far_truck"]["mean_visible_frame_iou"]>history["0"]["far_truck"]["mean_visible_frame_iou"]
 assert history["2"]["far_truck"]["mean_depth_error_on_target"]<history["0"]["far_truck"]["mean_depth_error_on_target"]
 records[str(st)]={"rounds":history,"chosen_round":2,
  "agent_decisions":["round0 two independent vehicle boxes",
    "round1 shared body/cabin/four tires tested for both",
    "round1 near geometry rejected due visible-mask and depth regression",
    "round2 near and all background restored/frozen at round0; optimize far only",
    "round2 accepted: preserves near silhouette, improves far silhouette and predicted-depth agreement"],
  "joint_visible_frames":both.tolist(),"ordering_diagnostics":ordering,
  "scene":reference(f"runs/two_cars_s{st}_fit_r2/scene.json"),
  "accepted_by":"Codex agent in current conversation after measured and visual review",
  "acceptance_scope":"local improvement versus one-box baseline, not independently validated 3D truth"}
out=Path("runs/two_cars_comparison");out.mkdir()
write_json(out/"report.json",{"schema":"scene_baseline.two_car_agent_experiment.v1",
 "actor":"current conversation Codex agent; no external VLM API configured or called",
 "case":"existing wan_joint generated clip, two red pickups despite singular script prompt",
 "inputs":"independent Step20 and Step50 x0 RGB readouts from existing 50-step sample; 49frames/16fps",
 "models":{"video":"Wan2.1-T2V-1.3B","geometry":"facebook/VGGT-1B","segmentation":"SAM3 local ModelScope checkpoint"},
 "stages":records,
 "no_silent_fallback":True,"manual_actions":"entity interpretation, SAM identity binding, topology proposals and accept/reject decisions; no pixel mask edits",
 "camera":"fixed stage-specific same-pass VGGT K/c2w, not optimized or independently validated",
 "unknown_trajectory":"no reconstruction before first or after last observation; global appearance/disappearance is observation gating, not physical creation/destruction",
 "limitations":["near vehicle remains one box; clipped full shape uncertain","rear tires/hidden side rely on symmetric shape prior",
 "unmodeled windows/ceiling/table; missing render areas not filled with RGB","static bookshelf shapes still coarse and may be distorted",
 "all frames fitted; depth and mask agreement are not independent truth","not a new early-stopped video generation budget benchmark"]})
print(records)
