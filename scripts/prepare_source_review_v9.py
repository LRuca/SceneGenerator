from pathlib import Path
import json
import numpy as np
import imageio.v2 as iio
from PIL import Image,ImageDraw
from scipy.ndimage import binary_erosion
from scene_baseline.scene import load_scene
from scene_baseline.storage import verify,read_json,reference
D=Path('/home/tione/notebook/home/keithanwang/vlm-vla/gafu/sceneblocker/SceneGenerator')
p=D/'runs/v9_s20_fiveway_r1';p.mkdir(exist_ok=False)
s,m,a=load_scene(D/'runs/v4_courtyard_r1/step20_visible_soft_containment_free/scene.json')
raw=np.load(verify(m['source_rgb']));md=read_json(verify(s['masks_manifest']))
with np.load(verify(md['data'])) as z:masks=z['masks'];ids=z['ids'].tolist()
with np.load(D/'runs/v4_courtyard_r1/step20_visible_soft_containment_free/landmark_fit.npz') as z:uv=z['image_points'];valid=z['valid'];pred=z['predicted_uv']
sel=np.linspace(0,uv.shape[1]-1,min(32,uv.shape[1])).round().astype(int)
colors={'forklift':(255,195,35),'parked_truck':(55,125,255),'gray_obstacle':(220,100,225),'wall':(75,200,160),'ground':(230,165,100)}
frames=[]
for t in range(len(raw)):
 rgb=(a['rgb'][t]*255).astype('uint8');im=Image.fromarray(rgb);d=ImageDraw.Draw(im)
 for eid,mask in zip(ids,masks[:,t]):
  boundary=mask & ~binary_erosion(mask,iterations=1);yy,xx=np.nonzero(boundary)
  col=colors.get(eid,(220,220,220))
  for y,x in zip(yy,xx):d.point((int(x),int(y)),fill=col)
  yy,xx=np.nonzero(mask)
  if len(yy):
   text=eid;x=int(np.clip(np.median(xx)-30,0,410));y=int(np.clip(np.percentile(yy,10)-14,4,490));d.rectangle((x,y,x+len(text)*7,y+13),fill=(25,25,25));d.text((x+2,y),text,fill=col)
 for j in sel:
  if not valid[t,j]:continue
  x,y=uv[t,j];xp,yp=pred[t,j]
  d.line((float(x),float(y),float(xp),float(yp)),fill=(250,80,160),width=1)
  d.ellipse((float(x-2),float(y-2),float(x+2),float(y+2)),fill=(20,255,245))
  d.line((float(xp-2),float(yp),float(xp+2),float(yp)),fill=(255,70,150),width=1)
  d.line((float(xp),float(yp-2),float(xp),float(yp+2)),fill=(255,70,150),width=1)
 d.rectangle((0,0,518,19),fill=(20,20,20));d.text((5,4),'Step20 | cyan: observed tracks | pink: fitted projection',fill='white')
 frames.append(np.asarray(im))
indices=np.linspace(0,len(raw)-1,96).round().astype(int)
with iio.get_writer(p/'source_step20.mp4',fps=24,codec='libx264',macro_block_size=1) as wr,iio.get_writer(p/'source_step20_annotated.mp4',fps=24,codec='libx264',macro_block_size=1) as wa:
 for t in indices:
  r=raw[t];r=(r*255).clip(0,255).astype('uint8') if r.dtype!=np.uint8 else r
  wr.append_data(np.asarray(Image.fromarray(r).resize((720,720))));wa.append_data(np.asarray(Image.fromarray(frames[t]).resize((720,720))))
(p/'annotation_provenance.json').write_text(json.dumps(dict(source_scene=reference(D/'runs/v4_courtyard_r1/step20_visible_soft_containment_free/scene.json'),source_rgb=m['source_rgb'],masks_manifest=s['masks_manifest'],track_source=reference(D/'runs/v4_courtyard_r1/step20_visible_soft_containment_free/landmark_fit.npz'),displayed_point_indices=sel.tolist(),points='cyan observed image tracks; pink fitted projections, not independent 3D ground truth',diffusion_step=20,timeline='all49 source frames resampled to96 frames,4s,24fps; no optical frame interpolation'),indent=2))
print('Source and annotated Step20 videos ready')
