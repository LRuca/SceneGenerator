"""Reviewed SAM masks; static shelf regions grouped as scenery, no dynamic merge."""
import numpy as np
from pathlib import Path
from scene_baseline.storage import read_json,write_json,reference
from scene_baseline.masks import import_masks
from scene_baseline.scene import initialize
from PIL import Image,ImageDraw
import imageio.v2 as imageio
for stage in [20,50]:
 out=Path(f"runs/stage{stage}_reviewed");out.mkdir()
 hyp=read_json(f"runs/stage{stage}_hypothesis/hypothesis.json")
 hyp["status"]="visually_reviewed_for_coarse_baseline_not_ground_truth"
 hyp["mask_review"]={
  "actor":"Codex inspection of frames 0,16,32,48 RGB+mask contact sheet",
  "vehicle":"one SAM identity throughout; no dynamic identity merging",
  "shelves":"reviewed stationary bookshelf regions grouped as one coarse background scenery block; not one physical object",
  "missing_shelves_frames":list(range(10)) if stage==20 else [],
  "cross_stage_transfer":False}
 write_json(out/"hypothesis.json",hyp)
 allm=[];sources=[]
 for entity in ["vehicle","ground","column","shelves"]:
  p=Path(f"runs/stage{stage}_sam3_{entity}/masks.npz")
  z=np.load(p);mask=z["masks"]
  if entity!="shelves" and len(mask)!=1:raise ValueError("Dynamic or architectural identity needs new review")
  allm.append(mask.any(0));sources.append(reference(p.parent/"manifest.json"))
 masks=np.stack(allm);ids=np.array(["vehicle","ground","column","shelves"])
 np.savez_compressed(out/"masks.npz",masks=masks,ids=ids)
 write_json(out/"review.json",{"sources":sources,"policy":hyp["mask_review"]})
 import_masks(f"runs/stage{stage}_geometry/manifest.json",out/"hypothesis.json",out/"masks.npz",f"runs/stage{stage}_masks")
 initialize(f"runs/stage{stage}_geometry/manifest.json",out/"hypothesis.json",f"runs/stage{stage}_masks/manifest.json",f"runs/stage{stage}_init")
 a=np.load(f"runs/stage{stage}_geometry/geometry.npz");colors=np.array([[255,70,70],[80,255,80],[60,180,255],[245,190,40]],float)
 frames=[]
 for t in range(len(a["depth"])):
  rgb=a["rgb"][t]*255
  for m,c in zip(masks,colors):rgb[m[t]]=rgb[m[t]]*.55+c*.45
  frames.append(np.asarray(Image.fromarray(rgb.astype("uint8")).resize((384,384))))
 imageio.mimwrite(out/"masks.mp4",frames,fps=16,codec="libx264",macro_block_size=1)
 print(stage,[(e["id"],np.diag(e["parts"][0]["canonical_to_entity"])[:3].tolist()) for e in read_json(f"runs/stage{stage}_init/scene.json")["entities"]],flush=True)
