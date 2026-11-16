"""Codex agent decisions after RGB/depth and independent instance-mask review."""
from pathlib import Path
import json,numpy as np
from scene_baseline.storage import read_json,write_json,reference
from scene_baseline.agent import accept
from scene_baseline.masks import import_masks
from scene_baseline.scene import initialize
import imageio.v2 as imageio
from PIL import Image,ImageDraw
for st in [20,50]:
 out=Path(f"runs/two_cars_s{st}_reviewed");out.mkdir()
 gm=f"runs/two_cars_s{st}_geometry/manifest.json"
 a=np.load(f"runs/two_cars_s{st}_geometry/geometry.npz")
 entities=[];masks=[];sources=[];bindings={}
 def add(eid,label,motion,kind,role,mask,prompt,source):
  entities.append({"id":eid,"label":label,"motion":motion,"segmentation_prompt":prompt,"confidence":.85,
   "parts":[{"id":eid+"_proxy","kind":kind,"role":role}],
   "uncertainties":["unseen extent is weakly constrained"]})
  masks.append(mask);bindings[eid]={"SAM_id":source,"observed_frame_indices":np.flatnonzero(mask.any((1,2))).tolist()}
 z=np.load(f"runs/two_cars_s{st}_sam3_vehicles/masks.npz")
 # Explicit reviewed identities, never inferred by area alone.
 assert z["ids"].tolist()==["vehicles:candidate_0","vehicles:candidate_1"]
 add("near_truck","foreground red pickup exiting right","rigid","box","coarse whole vehicle",z["masks"][0],"red pickup truck","vehicles:candidate_0")
 add("far_truck","smaller rear red pickup revealed behind foreground pickup","rigid","box","coarse whole vehicle",z["masks"][1],"red pickup truck","vehicles:candidate_1")
 for en,label,kind,role in [("ground","tiled floor","plane","ground"),("column","white pillar","box","architectural support")]:
  zz=np.load(f"runs/two_cars_s{st}_sam3_{en}/masks.npz");assert len(zz["ids"])==1
  add(en,label,"static",kind,role,zz["masks"][0],label,str(zz["ids"][0]))
 zz=np.load(f"runs/two_cars_s{st}_sam3_shelves/masks.npz")
 assert len(zz["ids"])==2
 for k in range(2):
  add("shelf_left" if k==0 else "shelf_right","left rear bookshelf" if k==0 else "right bookshelf",
      "static","box","coarse static bookcase",zz["masks"][k],"bookshelf",str(zz["ids"][k]))
 sources=[reference(f"runs/two_cars_s{st}_sam3_{en}/manifest.json") for en in ["vehicles","ground","column","shelves"]]
 h={"schema":"scene_baseline.hypothesis.v1","geometry_manifest_sha256":reference(gm)["sha256"],
 "actor":"Codex agent in current conversation; separate stage RGB/depth and SAM candidate visual inspection",
 "entities":entities,"relations":[{"subject":x,"relation":"supported_by","object":"ground","status":"semantic_prior"} for x in ["near_truck","far_truck","column","shelf_left","shelf_right"]],
 "tool_requests":[],"status":"reviewed_identity_hypothesis",
 "occlusion":[{"front":"near_truck","back":"far_truck","evidence":"both visible during overlapping frames; near pickup leaves right to expose far pickup",
 "status":"visual_qualitative_not_a_complete_amodal_mask"}],
 "reasoning":["Two real visual identities; script single-truck wording is contradicted",
 "SAM candidate0 stays foreground and exits; candidate1 emerges behind and persists",
 "Two bookshelf masks belong to separate left/right static furniture; keep separate geometry",
 "Do not reconstruct a hidden trajectory before first observation or after last observation"],
 "limitations":["near truck clipped throughout much of visible interval; full length uncertain",
 "global disappearance means no reconstructed pose outside observed interval, not physical nonexistence",
 "window/ceiling/table geometry not yet modeled","predicted camera is fixed and not independently validated"]}
 write_json(out/"hypothesis_response.json",h)
 accept(f"runs/two_cars_s{st}_agent/request.json",out/"hypothesis_response.json",f"runs/two_cars_s{st}_hypothesis")
 hp=f"runs/two_cars_s{st}_hypothesis/hypothesis.json"
 np.savez_compressed(out/"masks.npz",masks=np.stack(masks),ids=np.array([e["id"] for e in entities]))
 write_json(out/"review.json",{"actor":h["actor"],"sources":sources,"bindings":bindings,
 "visual_review_frames":[0,8,12,16,32,48],"cross_stage_transfer":False,"manual_pixel_editing":False,"dynamic_identity_merge":False})
 import_masks(gm,hp,out/"masks.npz",f"runs/two_cars_s{st}_masks")
 initialize(gm,hp,f"runs/two_cars_s{st}_masks/manifest.json",f"runs/two_cars_s{st}_init")
 colors=np.array([[255,50,220],[40,230,240],[90,235,70],[80,150,255],[245,190,30],[180,60,240]])
 frames=[]
 for t in range(len(a["depth"])):
  rgb=a["rgb"][t]*255
  for ma,co in zip(masks,colors):rgb[ma[t]]=rgb[ma[t]]*.5+co*.5
  im=Image.fromarray(rgb.astype("uint8")).resize((384,384));draw=ImageDraw.Draw(im)
  draw.rectangle([0,0,384,22],fill="black");draw.text((5,5),f"Step{st} frame{t} | near magenta / far cyan",fill="white")
  frames.append(np.asarray(im))
 imageio.mimwrite(out/"masks.mp4",frames,fps=16,codec="libx264",macro_block_size=1)
 print(st,"reviewed identities and initial scene written",flush=True)
