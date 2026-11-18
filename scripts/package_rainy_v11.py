from pathlib import Path
import argparse,json,copy
import numpy as np,cv2,imageio.v2 as imageio
from PIL import Image,ImageDraw
from scipy.ndimage import distance_transform_edt
from scene_baseline.storage import read_json,write_json,new_run,reference,verify
from scene_baseline.scene import load_scene
from scene_baseline.persistence import apply,effective_cameras
from scene_baseline.proxy_encoding_v8 import encode,binding_prompt
from scene_baseline.geometry import scene_parts,render_parts
from scene_baseline.lighting_v7 import render
ap=argparse.ArgumentParser();ap.add_argument('--step',type=int,required=True);ap.add_argument('--width',type=int,default=256);args=ap.parse_args();step=args.step
D=Path('/home/tione/notebook/home/keithanwang/vlm-vla/gafu/sceneblocker/SceneGenerator');b=D/f'runs/v11_rainy_s{step:02d}_reconstruction_r1';p=new_run(D/f'runs/v11_rainy_s{step:02d}_package_r2');(p/'references').mkdir()
policy=read_json(D.parent/'baseline/experiments/persistence_v1/policy.json');policy['smooth_camera']=False
# Preserve observation cameras exactly: do not smooth camera without jointly refitting object poses.
write_json(p/'persistence_policy.json',policy);apply(b/'fit/scene.json',p/'persistence_policy.json',p/'persistent')
s,m,a=load_scene(p/'persistent/scene.json');md=read_json(verify(s['masks_manifest']))
with np.load(verify(md['data'])) as z:masks=z['masks'];ids=z['ids'].tolist()
rgb=(a['rgb']*255).clip(0,255).astype('uint8');exclusive=masks&(masks.sum(0)[None]==1);refs=[];dec=[]
for e in s['entities']:
 eid=e['id'];mm=exclusive[ids.index(eid)];index=len(refs)+1;dst=p/'references'/f'{index:02d}_{eid}.png'
 if eid.startswith('car'):
  scores=[]
  for t in range(len(mm)):
   er=cv2.erode(mm[t].astype('uint8'),np.ones((3,3),np.uint8))>0;lap=cv2.Laplacian(cv2.cvtColor(rgb[t],cv2.COLOR_RGB2GRAY),cv2.CV_32F)
   scores.append(mm[t].sum()*np.sqrt(1+float(lap[er].var()) if er.any() else 1))
  t=int(np.argmax(scores));mask=mm[t];y,x=np.nonzero(mask);box=[int(x.min()),int(y.min()),int(x.max()+1),int(y.max()+1)];arr=rgb[t].copy();arr[~mask]=224;im=Image.fromarray(arr).crop(box);scale=560/max(im.size);im=im.resize(tuple(max(1,round(v*scale)) for v in im.size));canvas=Image.new('RGB',(720,720),(224,224,224));canvas.paste(im,((720-im.width)//2,(720-im.height)//2));canvas.save(dst)
  row=dict(entity=eid,file=reference(dst),role='single same-stage masked appearance; centered and independently normalized',source_frame=t,crop=box,outside_mask_source_pixels=0,geometry_scale_preserved=False)
  dec.append(dict(id=eid,role='key_actor' if 'crossing' in e['label'] else 'key_prop',appearance_image=index,appearance_label=e['label'],prompt_id=eid,evidence='Current chat reviewed same-stage track contact; script fidelity not forced',geometric_duties=['occlusion','contact']))
 else:
  best=None
  for t0 in range(len(mm)):
   dist=distance_transform_edt(np.pad(mm[t0],1))[1:-1,1:-1];y0,x0=np.unravel_index(np.argmax(dist),dist.shape)
   if best is None or dist[y0,x0]>best[0]:best=(float(dist[y0,x0]),t0,int(y0),int(x0))
  distance,t,y,x=best;r=max(1,min(18,int(distance*.5)));box=[x-r,y-r,x+r+1,y+r+1];assert mm[t,y-r:y+r+1,x-r:x+r+1].all();Image.fromarray(rgb[t]).crop(box).resize((720,720)).save(dst)
  row=dict(entity=eid,file=reference(dst),role='interior material swatch' if step>5 else 'uncertain local color/tone swatch only',source_frame=t,crop=box,all_source_pixels_inside_exclusive_mask=True,semantic_region='agent-proposed band, not SAM-measured boundary')
  dec.append(dict(id=eid,role='structural',appearance_image=index,evidence='Agent-proposed coarse structural surface from same-stage depth, not measured complete shape',geometric_duties=['support','occlusion','layout']))
 refs.append(row)
# Generic structural continuity prior; retain fitted planes and all actor shapes/poses.
from itertools import product
floor=next(e for e in s['entities'] if e['id']=='ground');wall=next(e for e in s['entities'] if e['id']=='background')
F=np.array(floor['entity_to_world'][0]);fm=np.array(floor['parts'][0]['canonical_to_entity']);up=np.array(s['ground_normal']);plane=(F@fm)[:3,3];wr=np.array(wall['entity_to_world'][0]);wm=np.array(wall['parts'][0]['canonical_to_entity']);worldM=wr@wm
axis=wr[:3,1];den=axis@up
if abs(den)>.5:
 top=wm[1,3]+wm[1,1]/2;bottom=(plane@up-wr[:3,3]@up-(wr[:3,:3]@np.array([wm[0,3],0,wm[2,3]]))@up)/den
 if bottom<top:
  wm[1,1]=top-bottom;wm[1,3]=(top+bottom)/2;wall['parts'][0]['canonical_to_entity']=wm.tolist()
verts=np.array(list(product([-.5,.5],repeat=3)));world=[]
for e in s['entities']:
 for po in np.array(e['entity_to_world']):
  for part in e['parts']:
   mat=po@np.array(part['canonical_to_entity']);world.append(verts@mat[:3,:3].T+mat[:3,3])
world.append(a['c2w'][:,:3,3]);local=(np.concatenate(world)-F[:3,3])@F[:3,:3];lo=local[:,[0,2]].min(0);hi=local[:,[0,2]].max(0);margin=(hi-lo)*.08
fm[0,0],fm[2,2]=(hi-lo+2*margin);fm[0,3],fm[2,3]=(lo+hi)/2;floor['parts'][0]['canonical_to_entity']=fm.tolist()
write_json(p/'structural_completion.json',dict(kind='support-continuity prior',measured_plane_position_and_normal_unchanged=True,actor_geometry_and_poses_unchanged=True,inferred=['extend ground footprint to all inferred actor support footprints, wall and camera ground projections','extend background lower edge to ground plane'],caution='extrapolated surface extent, not observed geometry'))
s=encode(s,dict(actor='current_chat_codex',entities=dec));write_json(p/'decisions.json',dict(actor='current_chat_codex',entities=dec));write_json(p/'scene.json',s)
base='''On a rainy night, a car emerges from an alley and passes a parked car. The camera slowly tracks sideways.
Render a continuous cinematic live-action night shot with wet asphalt, restrained blue street lighting, realistic reflections, coherent contact shadows and natural motion blur. No cuts, collage, text or logos.
[Video 1] provides the coarse scene geometry, relative scale, object trajectories, framing, timing, occlusion and camera path. Turn the simple boxes into plausible cars while retaining their coarse occupied regions and root paths. Do not copy the proxy colors as car paint. Preserve instance identities, natural rigid car proportions, road contact, and depth ordering. Keep the camera motion present in Video 1; do not add a zoom or a new camera move.
The original script above is the creative intent. The observed proxy contains the source generator's actual traffic, which may differ from the intended two-car story: do not force observed moving cars to become stationary or merge distinct identities.
Each isolated image is appearance only: its orientation, original pose, screen location and independently normalized size must not override Video 1. Material swatches provide texture and color only, no boundaries or layout. No original RGB video, complete scene image or RGB first frame is supplied.
'''
if step==5:base+='''LOW-EVIDENCE CONDITION: Video 1 contains only uncertain coarse background surfaces; no car identity or trajectory could be reliably reconstructed from this early stage. Complete the intended car action from the short script as a new plausible realization. This completion is not evidence of recovered source motion. Use the two images only as uncertain night color/tone swatches.\n'''
else:
 for i,row in enumerate(refs,1):
  if not row['entity'].startswith('car'):base+=f'[Image {i}] supplies only the interior material of {row["entity"]}.\n'
prompt=base+'\n'+binding_prompt(s['proxy_encoding']);(p/'prompt.txt').write_text(prompt)
colors={b['id']:b['rgb'] for b in s['proxy_encoding']['identity_bindings']};cams=effective_cameras(s,a);n,H,W=a['depth'].shape;width=args.width;height=round(H*width/W)//2*2;K=a['K'].copy();K[:,0,:]*=width/W;K[:,1,:]*=height/H
np.savez_compressed(p/'camera_trajectory.npz',source_c2w=a['c2w'],render_c2w=cams,K=a['K'],times_s=a['times_s']);frames={};annotated={};diagnostics=[];sample=np.linspace(0,n-1,96).round().astype(int)
for t in sorted(set(sample)):
 parts=scene_parts(s,int(t));frames[t]=render(parts,K[t],cams[t],height,width,s['ground_normal'],colors,cams[0,:3,0]);depth,labels=render_parts(parts,K[t],cams[t],height,width)
 arr=rgb[t].copy();im=Image.fromarray(arr);draw=ImageDraw.Draw(im)
 for e in s['entities']:
  if not e['id'].startswith('car'):continue
  mask=masks[ids.index(e['id']),t];yy,xx=np.nonzero(mask);col=tuple(colors[e['id']]);arr[mask]=(arr[mask]*.55+np.array(col)*.45).astype('uint8')
 im=Image.fromarray(arr);draw=ImageDraw.Draw(im)
 for e in s['entities']:
  if not e['id'].startswith('car'):continue
  mask=masks[ids.index(e['id']),t];yy,xx=np.nonzero(mask)
  if len(xx):draw.text((int(xx.min()),max(0,int(yy.min())-15)),e['id'],fill=tuple(colors[e['id']]))
 annotated[t]=np.asarray(im);diagnostics.append(dict(frame=int(t),surface_coverage=float(np.isfinite(depth).mean()),visible_pixels={e['id']:int(sum(np.count_nonzero(labels==j) for j,x in enumerate(parts) if x['entity_id']==e['id'])) for e in s['entities']}));print('render',step,t,flush=True)
for name,source in [('white_model_reference',frames),('annotated_source',annotated),('source_rgb',dict(enumerate(rgb)))]:
 with imageio.get_writer(p/(name+'.mp4'),fps=24,codec='libx264',macro_block_size=1) as video:
  for t in sample:video.append_data(np.asarray(Image.fromarray(source[t]).resize((720,720))))
# Fixed-global depth visualization from original geometry bundle.
d=a['depth'];near,far=np.percentile(d,[.5,99.5]);depth_frames={t:cv2.cvtColor(cv2.applyColorMap((np.clip((d[t]-near)/(far-near+1e-8),0,1)*255).astype('uint8'),cv2.COLORMAP_TURBO),cv2.COLOR_BGR2RGB) for t in set(sample)}
with imageio.get_writer(p/'depth_reference.mp4',fps=24,codec='libx264',macro_block_size=1) as v:
 for t in sample:v.append_data(np.asarray(Image.fromarray(depth_frames[t]).resize((720,720))))
board=Image.new('RGB',(960,3*265),(25,28,33));draw=ImageDraw.Draw(board)
for r,t in enumerate([0,n//2,n-1]):
 for c,(im,label) in enumerate([(rgb[t],'RGB'),(annotated[t],'observed masks'),(frames[t],'coarse reconstruction')]):board.paste(Image.fromarray(im).resize((320,240)),(c*320,r*265+25));draw.text((c*320+5,r*265+5),f'step{step} t{t} {label}',fill='white')
board.save(p/'review_contact.jpg');sheet=Image.new('RGB',(320*3,345*int(np.ceil(len(refs)/3))),(30,30,30));dd=ImageDraw.Draw(sheet)
for i,row in enumerate(refs):sheet.paste(Image.open(verify(row['file'])).resize((320,320)),(i%3*320,i//3*345+25));dd.text((i%3*320+5,i//3*345+5),f'Image {i+1}: {row["entity"]}',fill='white')
sheet.save(p/'appearance_sheet.jpg');write_json(p/'validation.json',dict(render=diagnostics,observed_vehicle_count=sum(e['id'].startswith('car') for e in s['entities']),camera_changed=False,sparse_point_refinement=False,fit='same-stage masks + shared shape + visible depth + free space + support; temporal smoothing; no verified body-fixed sparse landmarks',limitations=['VGGT static-world camera estimate may be biased by moving cars','single planar background is incomplete; foreground pole not modeled','hidden shape and unobserved lifespan extrapolated priors','stage5 no recoverable actor; completion is script-conditioned']))
write_json(p/'manifest.json',dict(schema='scene_baseline.blocking_package.v11',scene=reference(p/'scene.json'),source_geometry=reference(verify(s['geometry_manifest'])),references=refs,reference_video=reference(p/'white_model_reference.mp4'),prompt=reference(p/'prompt.txt'),source_diffusion_step=step,no_source_rgb_video=True,no_source_full_frame_image=True,no_first_frame_image=True,duration_s=4,fps=24,seedance_model='seedance-2.5',validation=reference(p/'validation.json')))
print('PACKAGE COMPLETE',step,flush=True)
