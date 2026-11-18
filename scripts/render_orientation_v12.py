from pathlib import Path
import copy,numpy as np,imageio.v2 as iio
from PIL import Image,ImageDraw
from scene_baseline.storage import new_run,write_json,reference,read_json,verify
from scene_baseline.scene import load_scene,validate_scene
from scene_baseline.geometry import scene_parts,render_parts
from scene_baseline.lighting_v7 import render
D=Path('/home/tione/notebook/home/keithanwang/vlm-vla/gafu/sceneblocker/SceneGenerator');base=D/'runs/v12_rainy_s20_orientation_r1';p=new_run(base/'review');old,m,a=load_scene(D/'runs/v11_rainy_s20_package_r2/scene.json');new,_,_=load_scene(base/'persistent/scene.json');result=copy.deepcopy(old);replacement=copy.deepcopy(next(e for e in new['entities'] if e['id']=='car_01'));orig=next(e for e in old['entities'] if e['id']=='car_01');replacement['proxy_encoding']=orig['proxy_encoding'];result['entities']=[replacement if e['id']=='car_01' else e for e in result['entities']];result['orientation_repair_source']=reference(base/'orientation_report.json');validate_scene(result,len(a['depth']));write_json(p/'scene.json',result)
N,H,W=a['depth'].shape;w=320;h=320;K=a['K'].copy();K[:,0,:]*=w/W;K[:,1,:]*=h/H;colors={x['id']:x['rgb'] for x in result['proxy_encoding']['identity_bindings']};md=read_json(verify(old['masks_manifest']))
with np.load(verify(md['data'])) as z:mm=z['masks'][z['ids'].tolist().index('car_01')]
frames={};rows=[]
for t in range(N):
 ims=[];ious=[]
 for scene in [old,result]:
  parts=scene_parts(scene,t);im=render(parts,K[t],a['c2w'][t],h,w,scene['ground_normal'],colors,a['c2w'][0,:3,0]);ims.append(im);_,lab=render_parts(parts,K[t],a['c2w'][t],h,w);pred=np.isin(lab,[i for i,x in enumerate(parts) if x['entity_id']=='car_01']);target=np.asarray(Image.fromarray(mm[t]).resize((w,h),Image.Resampling.NEAREST));ious.append(float((pred&target).sum()/max((pred|target).sum(),1)))
 rgb=(a['rgb'][t]*255).astype('uint8');rgb[mm[t]]=(.5*rgb[mm[t]]+.5*np.array([250,190,35])).astype('uint8');board=Image.new('RGB',(960,350),(25,30,36));d=ImageDraw.Draw(board)
 for j,(im,label) in enumerate(zip([rgb,*ims],['Step20 RGB / yellow mask','Before / erroneous spin','After / conservative axis'])):board.paste(Image.fromarray(im).resize((320,320)),(j*320,30));d.text((j*320+5,8),label,fill='white')
 frames[t]=(np.asarray(board),ims[1]);rows.append(dict(frame=t,old_visible_mask_iou=ious[0],new_visible_mask_iou=ious[1]));print('render',t,flush=True)
sample=np.linspace(0,N-1,96).round().astype(int)
for name,idx in [('before_after',0),('white_model_reference',1)]:
 with iio.get_writer(p/(name+'.mp4'),fps=24,codec='libx264',macro_block_size=1,quality=8) as wr:
  for t in sample:wr.append_data(frames[t][idx])
contact=Image.new('RGB',(960,350*4),(25,30,36))
for j,t in enumerate([0,30,39,48]):contact.paste(Image.fromarray(frames[t][0]),(0,j*350))
contact.save(p/'contact.jpg')
metrics={}
for name,s in [('before',old),('after',result)]:
 e=next(e for e in s['entities'] if e['id']=='car_01');R=np.array(e['entity_to_world'])[:,:3,:3];delta=np.arccos(np.clip((np.trace(R[1:]@R[:-1].transpose(0,2,1),axis1=1,axis2=2)-1)/2,-1,1))*180/np.pi;metrics[name]=dict(max_frame_rotation_deg=float(delta.max()),sum_rotation_deg=float(delta.sum()),mean_visible_mask_iou=float(np.mean([r['old_visible_mask_iou' if name=='before' else 'new_visible_mask_iou'] for r in rows])),box_axis_lengths=np.linalg.norm(np.array(e['parts'][0]['canonical_to_entity'])[:3,:3],axis=0).tolist())
assert metrics['after']['max_frame_rotation_deg']<3
assert all(x==y for x,y in zip([e for e in old['entities'] if e['id']!='car_01'],[e for e in result['entities'] if e['id']!='car_01']))
write_json(p/'metrics.json',dict(metrics=metrics,per_frame=rows,cameras_unchanged=True,other_entities_and_structures_unchanged=True,mask_iou_is_training_evidence_not_heldout=True,limitations=['dominant axial hypothesis excludes unverified turns','body front and hidden full shape unresolved','rotation stability does not prove orientation correctness','same-stage VGGT depth/camera may remain biased']))
print(metrics,flush=True)
