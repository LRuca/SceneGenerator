from pathlib import Path
import imageio.v2 as iio
from PIL import Image,ImageDraw,ImageOps
p=Path('/home/tione/notebook/home/keithanwang/vlm-vla/gafu/sceneblocker/SceneGenerator/runs/v14_rainy_s20_agent_motion_package_r1');out=p/'foreground_check_r1';out.mkdir(exist_ok=True)
clips=[]
for n in ['source_rgb.mp4','white_model_reference.mp4','generated.mp4']:
 r=iio.get_reader(p/n);fs=[x for x in r];print(n,r.get_meta_data(),len(fs));r.close();clips.append(fs)
s=Image.new('RGB',(7*256,3*280),(25,30,36));d=ImageDraw.Draw(s)
for i,fs in enumerate(clips):
 for j,t in enumerate([0,16,32,48,64,80,95]):
  im=ImageOps.contain(Image.fromarray(fs[min(t,len(fs)-1)]),(256,256));s.paste(im,(j*256,i*280+24));d.text((j*256+4,i*280+4),f'{["source","white","Seedance"][i]} frame{t}',fill='white')
s.save(out/'timeline.jpg')
