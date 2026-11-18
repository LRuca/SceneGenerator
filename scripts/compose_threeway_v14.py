from pathlib import Path
import imageio.v2 as iio
from PIL import Image,ImageDraw,ImageFont,ImageOps
import numpy as np,json
p=Path('/home/tione/notebook/home/keithanwang/vlm-vla/gafu/sceneblocker/SceneGenerator/runs/v14_rainy_s20_agent_motion_package_r1');out=p/'three_way_r4';out.mkdir(exist_ok=True)
font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',22)
def read(path):
 r=iio.get_reader(path); frames=[x for x in r]; meta=r.get_meta_data();r.close();return frames,meta
wan,wm=read(p.parent/'v11_rainy_inputs_r1/rgb_step_020.mp4');white,wmeta=read(p/'white_model_reference.mp4');seed,smeta=read(p/'generated.mp4');N=96
prompt='On a rainy night, a car emerges from an alley and passes a parked car. The camera slowly tracks sideways.'
def panel(x):
 return ImageOps.contain(Image.fromarray(x), (620,620))
def put(canvas,x,slot):
 im=panel(x);canvas.paste(im,(slot*640+(620-im.width)//2,120+(620-im.height)//2))
with iio.get_writer(out/'wan_white_seedance.mp4',fps=24,codec='libx264',macro_block_size=1,quality=8) as w:
 for k in range(N):
  canvas=Image.new('RGB',(1920,760),(20,25,32));put(canvas,wan[min(round(k/(N-1)*(len(wan)-1)),len(wan)-1)],0);put(canvas,white[min(k,len(white)-1)],1);put(canvas,seed[min(k,len(seed)-1)],2);d=ImageDraw.Draw(canvas)
  for i,label in enumerate(['ORIGINAL WAN RGB / Step20 x0','COARSE WHITE MODEL / Agent motion decision','SEEDANCE 2.5 / FINAL OUTPUT']):d.text((i*640+12,18),label,font=font,fill='white')
  d.text((12,70),prompt,font=font,fill='#ffd580');w.append_data(np.asarray(canvas))
contact=Image.new('RGB',(1920,760),(20,25,32));
for i,x in enumerate([wan[min(24,len(wan)-1)],white[min(48,len(white)-1)],seed[min(48,len(seed)-1)]]):put(contact,x,i)
d=ImageDraw.Draw(contact)
for i,label in enumerate(['ORIGINAL WAN RGB / Step20 x0','COARSE WHITE MODEL / Agent motion decision','SEEDANCE 2.5 / FINAL OUTPUT']):d.text((i*640+12,18),label,font=font,fill='white')
d.text((12,70),prompt,font=font,fill='#ffd580');contact.save(out/'contact.jpg',quality=93)
(out/'prompt_original.txt').write_text(prompt);(out/'manifest.json').write_text(json.dumps({'source_wan':str(p.parent/'v11_rainy_inputs_r1/rgb_step_020.mp4'),'source_white':str(p/'white_model_reference.mp4'),'source_seedance':str(p/'generated.mp4'),'output':'three equal 640px columns, 96 frames at24fps; Wan frames uniformly resampled, no crop','original_wan_frames':len(wan),'white_frames':len(white),'seedance_frames':len(seed),'original_wan_meta':wm,'white_meta':wmeta,'seedance_meta':smeta},indent=2,default=str))
