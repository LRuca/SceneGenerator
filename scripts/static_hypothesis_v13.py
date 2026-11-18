from pathlib import Path
import copy,numpy as np,imageio.v2 as iio
from PIL import Image,ImageDraw
from scene_baseline.storage import new_run,write_json,read_json,verify
from scene_baseline.scene import load_scene
from scene_baseline.fit_scene import fit
from scene_baseline.geometry import scene_parts,render_parts
from scene_baseline.lighting_v7 import render
D=Path('/home/tione/notebook/home/keithanwang/vlm-vla/gafu/sceneblocker/SceneGenerator');p=new_run(D/'runs/v13_s20_static_hypothesis_r1');initial,m,a=load_scene(D/'runs/v12_rainy_s20_orientation_r1/initial_scene.json');e=next(e for e in initial['entities'] if e['id']=='car_01');poses=np.array(e['entity_to_world']);anchor=poses[0].copy();anchor[:3,3]=np.median(poses[:,:3,3],0);e['entity_to_world']=[anchor.tolist()];e['motion']='static';e['renderable']=[True]*len(a['depth']);write_json(p/'initial_scene.json',initial);fit(p/'initial_scene.json',p/'fit',steps=160,resolution=48,device='cuda',lr=.01,entity_ids=['car_01'])
static,_,_=load_scene(p/'fit/scene.json');old,_,_=load_scene(D/'runs/v12_rainy_s20_orientation_r1/review/scene.json');candidate=copy.deepcopy(old);se=next(e for e in static['entities'] if e['id']=='car_01');se['proxy_encoding']=next(e for e in old['entities'] if e['id']=='car_01')['proxy_encoding'];candidate['entities']=[se if e['id']=='car_01' else e for e in candidate['entities']];write_json(p/'scene.json',candidate)
md=read_json(verify(old['masks_manifest']))
with np.load(verify(md['data'])) as z:mm=z['masks'][z['ids'].tolist().index('car_01')]
N,H,W=a['depth'].shape;w=320;K=a['K'].copy();K[:,0,:]*=w/W;K[:,1,:]*=w/H;colors={b['id']:b['rgb'] for b in old['proxy_encoding']['identity_bindings']};frames=[];stats=[]
for t in range(N):
 ims=[];metrics=[];target=np.array(Image.fromarray(mm[t]).resize((w,w),Image.Resampling.NEAREST));target_depth=np.asarray(Image.fromarray(a['depth'][t]).resize((w,w),Image.Resampling.NEAREST))
 for name,s in [('dynamic',old),('static',candidate)]:
  parts=scene_parts(s,t);ims.append(render(parts,K[t],a['c2w'][t],w,w,s['ground_normal'],colors,a['c2w'][0,:3,0]));d,lab=render_parts(parts,K[t],a['c2w'][t],w,w);pred=np.isin(lab,[i for i,x in enumerate(parts) if x['entity_id']=='car_01']);hit=pred&target;metrics.append(dict(model=name,iou=float((hit).sum()/max((pred|target).sum(),1)),observed_coverage=float(hit.sum()/max(target.sum(),1)),relative_depth_error=float(np.median(np.abs(d[hit]-target_depth[hit])/target_depth[hit])) if hit.any() else None))
 board=Image.new('RGB',(960,350),(25,30,36));dd=ImageDraw.Draw(board)
 for j,(im,label) in enumerate(zip([(a['rgb'][t]*255).astype('uint8'),*ims],['Step20 source','Dynamic / orientation repaired','Static hypothesis / same camera'])):board.paste(Image.fromarray(im).resize((w,w)),(j*w,30));dd.text((j*w+5,8),label,fill='white')
 frames.append(np.array(board));stats.append(dict(frame=t,metrics=metrics))
with iio.get_writer(p/'static_vs_dynamic.mp4',fps=24,codec='libx264',macro_block_size=1) as wr:
 for t in np.linspace(0,N-1,96).round().astype(int):wr.append_data(frames[t])
sheet=Image.new('RGB',(960,350*4))
for j,t in enumerate([0,24,39,48]):sheet.paste(Image.fromarray(frames[t]),(0,350*j))
sheet.save(p/'contact.jpg');summary={}
for name,j in [('dynamic',0),('static',1)]:summary[name]={k:float(np.mean([x['metrics'][j][k] for x in stats if x['metrics'][j][k] is not None])) for k in ['iou','observed_coverage','relative_depth_error']}
summary['static_fit_objective']=read_json(p/'fit/report.json')['best_loss'];summary['dynamic_fit_objective']=read_json(D/'runs/v12_rainy_s20_orientation_r1/fit/report.json')['best_loss'];write_json(p/'comparison.json',dict(summary=summary,per_frame=stats,notes=['same-camera diagnostic, not proof of actual stationarity','dynamic reference includes persistence/support while static fit does not; visible metrics compare actual candidates','objective values compare pre-persistence fit candidates; same loss, evidence, optimization budget, shared initial shape; nonconvex starts differ in pose parameterization','no independent heldout evaluation']));print(summary,flush=True)
