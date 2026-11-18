from pathlib import Path
import imageio.v2 as iio
from PIL import Image,ImageDraw,ImageFont,ImageOps
import numpy as np,json
p=Path('/home/tione/notebook/home/keithanwang/vlm-vla/gafu/sceneblocker/SceneGenerator/runs/v14_rainy_s20_agent_motion_package_r1');out=p/'three_way_sync_r1';out.mkdir(exist_ok=True)
font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',20)
def read(path):
 r=iio.get_reader(path); fs=[x for x in r]; m=r.get_meta_data();r.close();return fs,m
wan,wm=read(p.parent/'v11_rainy_inputs_r1/rgb_step_020.mp4');white,wmeta=read(p/'white_model_reference.mp4');seed,smeta=read(p/'generated.mp4');N=96;fps=24
prompt='On a rainy night, a car emerges from an alley and passes a parked car. The camera slowly tracks sideways.'
def panel(x):return ImageOps.contain(Image.fromarray(x),(600,600))
def put(c,x,slot):
 im=panel(x);c.paste(im,(slot*640+(600-im.width)//2,130+(600-im.height)//2))
# Normalized shot-time mapping: Wan 49 frames span 0..(49-1)/16; white 96 frames span 0..4; both correspond to the same generated stage sequence.
def wan_idx(k):return int(round(k*(len(wan)-1)/(N-1)))
def seed_idx(k):return min(k,len(seed)-1)
with iio.get_writer(out/'wan_white_seedance_synced.mp4',fps=fps,codec='libx264',macro_block_size=1,quality=8) as w:
 for k in range(N):
  wi=wan_idx(k);si=seed_idx(k);c=Image.new('RGB',(1920,780),(20,25,32));put(c,wan[wi],0);put(c,white[k],1);put(c,seed[si],2);d=ImageDraw.Draw(c);t=k/fps
  labels=[f'WAN RGB | t={t:.2f}s | source frame {wi}/48',f'WHITE MODEL | t={t:.2f}s | frame {k}/95',f'SEEDANCE | t={t:.2f}s | frame {si}/{len(seed)-1}']
  for j,l in enumerate(labels):d.text((j*640+10,18),l,font=font,fill='white')
  d.text((10,78),prompt,font=font,fill='#ffd580');w.append_data(np.asarray(c))
c=Image.new('RGB',(1920,780),(20,25,32));k=48;put(c,wan[wan_idx(k)],0);put(c,white[k],1);put(c,seed[k],2);d=ImageDraw.Draw(c)
for j,l in enumerate([f'WAN RGB | t=2.00s | source frame {wan_idx(k)}/48',f'WHITE MODEL | t=2.00s | frame 48/95',f'SEEDANCE | t=2.00s | frame 48/{len(seed)-1}']):d.text((j*640+10,18),l,font=font,fill='white')
d.text((10,78),prompt,font=font,fill='#ffd580');c.save(out/'contact.jpg',quality=93)
(out/'manifest.json').write_text(json.dumps({'mapping':'all outputs use 96 frames at24fps over0..4s; Wan source frame round(k*48/95), white frame k, Seedance frame k','wan_native_fps':wm.get('fps'),'wan_native_frames':len(wan),'white_fps':wmeta.get('fps'),'white_frames':len(white),'seedance_fps':smeta.get('fps'),'seedance_frames':len(seed),'warning':'Wan native duration is 49/16=3.0625s; this comparison uses normalized event-time retiming to the 4-second white-model timeline; it does not claim native wall-clock timing equivalence.'},indent=2))
