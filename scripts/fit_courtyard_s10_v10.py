from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from scene_baseline.storage import read_json,write_json,reference,verify,new_run
from scene_baseline.masks import import_masks
from scene_baseline.scene import initialize,load_scene
from scene_baseline.fit_scene import fit
from scene_baseline.tracking_v2 import run as track
R=Path('/home/tione/notebook/home/keithanwang/vlm-vla/gafu');D=R/'sceneblocker/SceneGenerator';p=new_run(D/'runs/v10_s10_fit_r1');sam=D/'runs/v10_s10_sam_r1';gm=D/'runs/v10_courtyard_s10_geometry_r1/manifest.json';hyp=sam/'hypothesis.json';h=read_json(hyp)
# Explicit visual candidate selection is supplied by the current-chat agent before this script runs.
review=read_json(D/'runs/v10_s10_candidate_review.json');masks=[];ids=[]
for e in h['entities']:
 md=read_json(sam/e['id']/'manifest.json')
 with np.load(verify(md['data'])) as z:
  idx=z['ids'].tolist().index(review['selected'][e['id']]);mask=z['masks'][idx]
  if mask.sum()==0:raise ValueError('No observations '+e['id'])
  masks.append(mask);ids.append(e['id'])
# Exclusive-instance ownership only where masks conflict: actor > kiosk > wall > ground.
ordered=['forklift','parked_truck','gray_obstacle','wall','ground'];owned=np.zeros_like(masks[0]);clean={}
for eid in ordered:
 mm=masks[ids.index(eid)] & ~owned;clean[eid]=mm;owned|=mm
np.savez_compressed(p/'masks.npz',masks=np.stack([clean[e] for e in ids]),ids=np.array(ids))
import_masks(gm,hyp,p/'masks.npz',p/'masks');initialize(gm,hyp,p/'masks/manifest.json',p/'init')
# Same shared-shape fitter; same-stage visible surface roles avoid inflated wall/kiosk depth extent.
fit(p/'init/scene.json',p/'fit',steps=160,resolution=48,device='cuda',lr=.015)
track(gm,p/'masks/manifest.json','forklift',p/'tracks')
print('FIT AND TRACKS READY',flush=True)
