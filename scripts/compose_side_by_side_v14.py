from pathlib import Path
import imageio.v2 as iio
from PIL import Image,ImageDraw,ImageFont
import numpy as np,json
p=Path('/home/tione/notebook/home/keithanwang/vlm-vla/gafu/sceneblocker/SceneGenerator/runs/v14_rainy_s20_agent_motion_package_r1');out=p/'side_by_side_r2';out.mkdir()
font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',22)
white=iio.get_reader(p/'white_model_reference.mp4');seed=iio.get_reader(p/'generated.mp4');wf=[x for x in white];sf=[x for x in seed];white.close();seed.close();n=96
prompt='On a rainy night, a car emerges from an alley and passes a parked car. The camera slowly tracks sideways.'
with iio.get_writer(out/'white_vs_seedance.mp4',fps=24,codec='libx264',macro_block_size=1,quality=8) as w:
 for k in range(n):
  a=Image.fromarray(wf[min(k,len(wf)-1)]).resize((640,640)); b=Image.fromarray(sf[min(k,len(sf)-1)]); b.thumbnail((640,640)); canvas=Image.new('RGB',(1280,760),(20,25,32));canvas.paste(a,(0,120));canvas.paste(b,(640+(640-b.width)//2,120));d=ImageDraw.Draw(canvas);d.text((12,18),'WHITE MODEL / Agent static-motion decision',font=font,fill='white');d.text((652,18),'SEEDANCE 2.5 / final output',font=font,fill='white');d.text((12,70),prompt,font=font,fill='#ffd580');w.append_data(np.asarray(canvas))
contact=Image.new('RGB',(1280,760),(20,25,32));contact.paste(Image.fromarray(wf[48]).resize((640,640)),(0,120));b=Image.fromarray(sf[48]);b.thumbnail((640,640));contact.paste(b,(640+(640-b.width)//2,120));d=ImageDraw.Draw(contact);d.text((12,18),'WHITE MODEL / Agent static-motion decision',font=font,fill='white');d.text((652,18),'SEEDANCE 2.5 / final output',font=font,fill='white');d.text((12,70),prompt,font=font,fill='#ffd580');contact.save(out/'contact.jpg',quality=93)
(out/'prompt_original.txt').write_text(prompt);(out/'prompt_seedance_actual.txt').write_text((p/'prompt.txt').read_text());(out/'manifest.json').write_text(json.dumps({'source_white_model':str(p/'white_model_reference.mp4'),'source_seedance':str(p/'generated.mp4'),'frames':n,'fps':24,'layout':'left white model, right Seedance output, prompt shown in header','original_prompt':prompt},indent=2))
