from pathlib import Path
import json,numpy as np,imageio.v2 as iio
from PIL import Image,ImageDraw,ImageFont,ImageOps
D=Path('/home/tione/notebook/home/keithanwang/vlm-vla/gafu/sceneblocker/SceneGenerator');out=D/'runs/v11_rainy_delivery_r1';out.mkdir(exist_ok=False)
fontpath=next(iter(Path('/usr/share/fonts').rglob('DejaVuSans.ttf')),None);font=ImageFont.truetype(str(fontpath),15) if fontpath else ImageFont.load_default();boards={};metas=[]
def fit_image(x):
 im=ImageOps.contain(Image.fromarray(x),(320,320));canvas=Image.new('RGB',(320,320),(12,16,22));canvas.paste(im,((320-im.width)//2,(320-im.height)//2));return np.asarray(canvas)
totalN=96
for step in [5,10,20,40]:
 rr=iio.get_reader(D/f'runs/v11_rainy_s{step:02d}_seedance_r1/generated.mp4');totalN=max(totalN,int(np.ceil(rr.count_frames()/rr.get_meta_data()['fps']*24)));rr.close()
for step in [5,10,20,40]:
 p=D/f'runs/v11_rainy_s{step:02d}_package_r2';generated=D/f'runs/v11_rainy_s{step:02d}_seedance_r1/generated.mp4';paths=[p/'source_rgb.mp4',p/'depth_reference.mp4',p/'annotated_source.mp4',p/'white_model_reference.mp4',generated];names=['RGB / predicted clean x0','VGGT depth (clip fixed range)','Observed instance masks','Coarse white model','Seedance 2.5'];clips=[];meta=[]
 for path in paths:
  r=iio.get_reader(path);m=r.get_meta_data();frames=[fit_image(x) for x in r];r.close();clips.append(frames);meta.append(dict(file=str(path),fps=m['fps'],frames=len(frames),duration=m.get('duration'),size=m.get('size')))
 contact=Image.new('RGB',(1600,3*355),(24,28,36));bs=[]
 with iio.get_writer(out/f'fiveway_step{step:02d}.mp4',fps=24,codec='libx264',macro_block_size=1,quality=8) as wr:
  for k in range(totalN):
   board=Image.new('RGB',(1600,356),(24,28,36));dd=ImageDraw.Draw(board)
   for j,(clip,m,label) in enumerate(zip(clips,meta,names)):
    idx=round(k/24*m['fps']);board.paste(Image.fromarray(clip[min(idx,len(clip)-1)]),(j*320,30));dd.text((j*320+5,6),f'S{step:02d} {label}',font=font,fill='white')
    if idx>=len(clip):dd.text((j*320+5,332),'end-frame hold',font=font,fill='#FFD580')
   if step==5:dd.text((3*320+4,332),'NO ACTOR EVIDENCE',font=font,fill='#FFBA70');dd.text((4*320+4,332),'SCRIPT COMPLETION',font=font,fill='#FFBA70')
   wr.append_data(np.asarray(board));bs.append(np.asarray(board))
   if k in [0,48,totalN-1]:contact.paste(board,(0,[0,48,totalN-1].index(k)*355))
 boards[step]=bs;contact.save(out/f'fiveway_step{step:02d}_contact.jpg',quality=93);metas.append(dict(step=step,clips=meta))
with iio.get_writer(out/'all_stages_comparison.mp4',fps=24,codec='libx264',macro_block_size=1,quality=8) as wr:
 for k in range(totalN):wr.append_data(np.concatenate([boards[s][k] for s in [5,10,20,40]],axis=0))
Image.fromarray(np.concatenate([boards[s][48] for s in [5,10,20,40]],axis=0)).save(out/'all_stages_contact.jpg',quality=94)
# Large 2x2 original-stage preview.
with iio.get_writer(out/'diffusion_stages_5_10_20_40.mp4',fps=24,codec='libx264',macro_block_size=1,quality=8) as wr:
 for k in range(96):
  canvas=Image.new('RGB',(960,1020),(20,25,32));dd=ImageDraw.Draw(canvas)
  for i,s in enumerate([5,10,20,40]):
   frame=boards[s][k][30:350,:320];canvas.paste(Image.fromarray(frame).resize((480,480)),(i%2*480,i//2*510+30));dd.text((i%2*480+8,i//2*510+8),f'Wan2.1 | Step {s} / 50 | predicted clean x0',font=font,fill='white')
  wr.append_data(np.asarray(canvas))
(out/'comparison_metadata.json').write_text(json.dumps(dict(stages=metas,duration_s=totalN/24,fps=24,source_retiming='49 original frames uniformly resampled to96; full source retained',generated_timeline='native24fps; explicit last-frame hold only if shorter',no_alignment_warp=True,no_cross_stage_geometry=True),indent=2));print(out,flush=True)
