from pathlib import Path
import json,time
import numpy as np
import torch
from PIL import Image
from sam3.model_builder import build_sam3_video_predictor
from scene_baseline.storage import reference,write_json,new_run
from scene_baseline.contracts import load_bundle
R=Path('/home/tione/notebook/home/keithanwang/vlm-vla/gafu');D=R/'sceneblocker/SceneGenerator';gm=D/'runs/v10_courtyard_s10_geometry_r1/manifest.json'
m,a=load_bundle(gm);p=new_run(D/'runs/v10_s10_sam_r1');frames=p/'frames';frames.mkdir()
for t,rgb in enumerate(a['rgb']):Image.fromarray((rgb*255).clip(0,255).astype('uint8')).save(frames/f'{t:05d}.jpg',quality=95)
entities=[]
for eid,label,motion,prompt,kind,role in [('forklift','yellow forklift','rigid','yellow forklift','box','coarse whole entity'),('parked_truck','blue pickup truck','static','blue pickup truck','box','coarse whole entity'),('ground','paved ground','static','paved ground','plane','ground'),('wall','brick wall','static','brick wall','plane','wall'),('gray_obstacle','gray metal kiosk','static','gray metal kiosk','plane','vertical_surface')]:
 entities.append(dict(id=eid,label=label,motion=motion,segmentation_prompt=prompt,confidence=.7,parts=[dict(id=eid+'_proxy',kind=kind,role=role)]))
hyp=dict(schema='scene_baseline.hypothesis.v1',actor='current_chat_codex; semantic vocabulary and coarse representation prior only; no later-stage coordinates/masks',geometry_manifest_sha256=reference(gm)['sha256'],entities=entities,relations=[dict(subject='forklift',relation='supported_by',object='ground',status='semantic prior')],tool_requests=[])
write_json(p/'hypothesis.json',hyp)
torch.set_num_threads(8)
predictor=build_sam3_video_predictor(checkpoint_path=str(R/'sceneblocker/baseline/weights/facebook/sam3/sam3.pt'),gpus_to_use=[0],compile=False)
for ent in entities:
 session=None;instances={};out=p/ent['id'];out.mkdir()
 try:
  session=predictor.handle_request(dict(type='start_session',resource_path=str(frames)))['session_id']
  predictor.handle_request(dict(type='add_prompt',session_id=session,frame_index=0,text=ent['segmentation_prompt']))
  for resp in predictor.handle_stream_request(dict(type='propagate_in_video',session_id=session)):
   t=resp['frame_index'];r=resp['outputs'];ids=r['out_obj_ids'].tolist();masks=r['out_binary_masks']
   if torch.is_tensor(masks):masks=masks.cpu().numpy()
   for j,oid in enumerate(ids):instances.setdefault(int(oid),np.zeros(a['depth'].shape,bool))[t]=np.asarray(masks[j],bool).squeeze()
  keys=sorted(instances);ids=[ent['id'] if len(keys)==1 else ent['id']+':candidate_'+str(k) for k in keys];masks=np.stack([instances[k] for k in keys]) if keys else np.empty((0,*a['depth'].shape),bool)
  np.savez_compressed(out/'masks.npz',masks=masks,ids=np.array(ids,dtype='U80'))
  write_json(out/'manifest.json',dict(geometry_manifest=reference(gm),hypothesis=reference(p/'hypothesis.json'),data=reference(out/'masks.npz'),candidate_ids=ids,positive_pixels=masks.sum((2,3)).tolist(),status='candidates_pending_visual_review'))
  print(ent['id'],'candidates',ids,'visible_frames',masks.any((2,3)).sum(1).tolist(),flush=True)
 finally:
  if session is not None:predictor.handle_request(dict(type='close_session',session_id=session))
if hasattr(predictor,'shutdown'):predictor.shutdown()
# Candidate overlay sheet; no silent merging or selection.
rows=[]
for ent in entities:
 with np.load(p/ent['id']/'masks.npz') as z:
  for i,eid in enumerate(z['ids'].tolist()):rows.append((eid,z['masks'][i]))
from PIL import ImageDraw
board=Image.new('RGB',(5*260,max(1,len(rows))*285),(22,26,32));d=ImageDraw.Draw(board)
for j,(eid,mask) in enumerate(rows):
 for k,t in enumerate([0,12,24,36,48]):
  rgb=(a['rgb'][t]*255).astype('uint8');rgb[mask[t]]=(rgb[mask[t]]*.5+np.array([255,100,50])*.5).astype('uint8');board.paste(Image.fromarray(rgb).resize((260,260)),(k*260,j*285+25));d.text((k*260+3,j*285+5),f'{eid} / t{t}',fill='white')
board.save(p/'candidate_contact.jpg')
