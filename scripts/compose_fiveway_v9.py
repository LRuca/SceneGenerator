from pathlib import Path
import json
import numpy as np
import imageio.v2 as iio
from PIL import Image,ImageDraw,ImageFont
D=Path('/home/tione/notebook/home/keithanwang/vlm-vla/gafu/sceneblocker/SceneGenerator');p=D/'runs/v9_s20_fiveway_r1'
paths=[p/'source_step20.mp4',p/'source_step20_annotated.mp4',D/'runs/v9_s20_original_pipeline_r3/blocking/white_model_reference.mp4',p/'generated_original.mp4',p/'generated_swapped.mp4']
labels=['1  Step20 source RGB','2  Step20 masks / tracks / fit','3  Proxy + lateral camera','4  A: forklift / blue pickup','5  B: coupe / delivery van']
fontpath=next(iter(Path('/usr/share/fonts').rglob('DejaVuSans.ttf')),None);font=ImageFont.truetype(str(fontpath),17) if fontpath else ImageFont.load_default()
clips=[];metas=[]
for f in paths:
 r=iio.get_reader(f);m=r.get_meta_data();frames=[np.asarray(Image.fromarray(x).resize((512,512))) for x in r];r.close();clips.append(frames);metas.append(dict(file=str(f),frames=len(frames),fps=m['fps'],duration=m.get('duration')))
print(metas,flush=True)
contact=Image.new('RGB',(2560,3*552),(24,28,36))
with iio.get_writer(p/'fiveway_step20.mp4',fps=24,codec='libx264',macro_block_size=1,quality=8) as w:
 for k in range(96):
  board=Image.new('RGB',(2560,552),(24,28,36));d=ImageDraw.Draw(board)
  for j,(frames,meta,label) in enumerate(zip(clips,metas,labels)):
   raw_idx=round(k/24*meta['fps']);index=min(raw_idx,len(frames)-1);board.paste(Image.fromarray(frames[index]),(j*512,30));d.text((j*512+8,6),label,fill='white',font=font)
   if raw_idx>=len(frames):d.text((j*512+350,516),'end-frame hold',fill='#FFD580',font=font)
  w.append_data(np.asarray(board))
  if k in [0,48,95]:contact.paste(board,(0,[0,48,95].index(k)*552))
contact.save(p/'fiveway_contact.jpg',quality=93)
(p/'fiveway_metadata.json').write_text(json.dumps(dict(order=labels,clips=metas,comparison_duration_s=4,comparison_fps=24,generated_timeline='preserve24fps; hold final frame if shorter; do not speed-change generated motion',source_timeline='all49 Step20 frames resampled to96/4s',geometry_camera='source panels retain original camera; white model and both Seedance clips use the added lateral camera',same_reference_video_sha256='7616f4b5b386be1a4622d8d283f714dc0fe44d1f74fe6310889695f5810a8234'),indent=2))
