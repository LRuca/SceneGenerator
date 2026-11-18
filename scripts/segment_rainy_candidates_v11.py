"""Per-frame SAM3 candidates; association/selection is a separate agent-reviewed step."""
from pathlib import Path
import json,argparse
import numpy as np
import torch
from PIL import Image,ImageDraw
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from scene_baseline.contracts import load_bundle
from scene_baseline.storage import new_run,reference,write_json
ap=argparse.ArgumentParser();ap.add_argument('--step',type=int,required=True);args=ap.parse_args();step=args.step
R=Path('/home/tione/notebook/home/keithanwang/vlm-vla/gafu');D=R/'sceneblocker/SceneGenerator';gm=D/f'runs/v11_rainy_s{step:02d}_geometry_r1/manifest.json';m,a=load_bundle(gm)
p=new_run(D/f'runs/v11_rainy_s{step:02d}_sam_candidates_r1')
torch.set_num_threads(6);model=build_sam3_image_model(checkpoint_path=str(R/'sceneblocker/baseline/weights/facebook/sam3/sam3.pt'),load_from_HF=False);proc=Sam3Processor(model,confidence_threshold=.25)
rows=[]
for t,rgb in enumerate(a['rgb']):
 im=Image.fromarray((rgb*255).clip(0,255).astype('uint8'));record={}
 with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
  state=proc.set_image(im)
  for eid,prompt in [('cars','car'),('ground','road'),('buildings','building')]:
   proc.reset_all_prompts(state);out=proc.set_text_prompt(prompt=prompt,state=state)
   masks=out['masks'].detach().cpu().numpy().reshape(-1,im.height,im.width)>0
   scores=out['scores'].detach().float().cpu().numpy().reshape(-1)
   boxes=out['boxes'].detach().float().cpu().numpy().reshape(-1,4)
   np.savez_compressed(p/f'{eid}_{t:03d}.npz',masks=masks,scores=scores,boxes=boxes)
   record[eid]=dict(count=len(masks),scores=scores.tolist(),boxes=boxes.tolist())
 rows.append(record);print(step,t,{k:v['count'] for k,v in record.items()},flush=True)
board=Image.new('RGB',(5*320,3*345),(22,26,32));draw=ImageDraw.Draw(board)
for r,eid in enumerate(['cars','ground','buildings']):
 for c,t in enumerate([0,12,24,36,48]):
  rgb=(a['rgb'][t]*255).astype('uint8');im=Image.fromarray(rgb);d=ImageDraw.Draw(im)
  with np.load(p/f'{eid}_{t:03d}.npz') as z:
   for j,(mask,score,box) in enumerate(zip(z['masks'],z['scores'],z['boxes'])):
    col=[(255,190,35),(65,135,255),(210,80,220),(70,210,155)][j%4]
    arr=np.asarray(im).copy();arr[mask]=(arr[mask]*.55+np.array(col)*.45).astype('uint8');im=Image.fromarray(arr);d=ImageDraw.Draw(im);x,y,x2,y2=box;d.rectangle([float(x),float(y),float(x2),float(y2)],outline=col,width=2);d.text((float(x),float(y)),f'{j}:{score:.2f}',fill=col)
  board.paste(im.resize((320,320)),(c*320,r*345+25));draw.text((c*320+5,r*345+5),f'S{step} {eid} / t{t}',fill='white')
board.save(p/'candidate_contact.jpg')
write_json(p/'manifest.json',dict(geometry_manifest=reference(gm),sam_checkpoint=reference(R/'sceneblocker/baseline/weights/facebook/sam3/sam3.pt'),method='SAM3 per-frame text candidates at confidence .25; no temporal identities assumed',frames=rows,status='pending_current_chat_identity_review'))
