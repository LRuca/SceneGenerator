from pathlib import Path
import imageio.v2 as iio
import argparse
from scene_baseline.storage import inside,new_run
ap=argparse.ArgumentParser();ap.add_argument("--generated",required=True);ap.add_argument("--out",required=True);args=ap.parse_args()
from PIL import Image,ImageDraw,ImageOps
p=Path('/home/tione/notebook/home/keithanwang/vlm-vla/gafu/sceneblocker/SceneGenerator/runs/v14_rainy_s20_agent_motion_package_r1');out=new_run(args.out)
clips=[]
for n in ['white_model_reference.mp4','generated.mp4']:
 r=iio.get_reader(inside(args.generated) if n=='generated.mp4' else p/n);fs=[x for x in r];r.close();clips.append(fs)
r=iio.get_reader('/home/tione/notebook/home/keithanwang/vlm-vla/gafu/sceneblocker/SceneGenerator/runs/v11_rainy_inputs_r1/rgb_step_020.mp4');clips.insert(0,[x for x in r]);r.close()
s=Image.new('RGB',(7*256,3*280),(25,30,36));d=ImageDraw.Draw(s)
for i,fs in enumerate(clips):
 for j,t in enumerate([0,16,32,48,64,80,95]):
  idx=round(t*(len(fs)-1)/95);im=ImageOps.contain(Image.fromarray(fs[idx]),(256,256));s.paste(im,(j*256,i*280+24));d.text((j*256+4,i*280+4),f'{["WAN","WHITE","V15"][i]} t{t} src{idx}',fill='white')
s.save(out/'timeline.jpg')
