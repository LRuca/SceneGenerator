from pathlib import Path
import copy,numpy as np
from scipy.spatial.transform import Rotation
from scene_baseline.storage import new_run,read_json,write_json,verify,reference
from scene_baseline.scene import load_scene
from scene_baseline.persistence import _frame_basis,apply
from scene_baseline.orientation_v12 import recover
from scene_baseline.fit_scene import fit
D=Path('/home/tione/notebook/home/keithanwang/vlm-vla/gafu/sceneblocker/SceneGenerator');p=new_run(D/'runs/v12_rainy_s20_orientation_r1');old=D/'runs/v11_rainy_s20_reconstruction_r1';s,m,a=load_scene(old/'fit/scene.json');init,_,_=load_scene(old/'init/scene.json');b=_frame_basis(s,a);eid='car_01';e=next(e for e in s['entities'] if e['id']==eid);raw=next(e for e in init['entities'] if e['id']==eid);md=read_json(verify(s['masks_manifest']))
with np.load(verify(md['data'])) as z:mm=z['masks'][z['ids'].tolist().index(eid)]
y=[];weights=[];quality=[]
for t,mask in enumerate(mm):
 v,u=np.nonzero(mask);stride=max(1,len(u)//1200);v,u=v[::stride],u[::stride];ray=np.column_stack([u,v,np.ones(len(u))])@np.linalg.inv(a['K'][t]).T;pt=ray*a['depth'][t,v,u,None];pt=pt@a['c2w'][t,:3,:3].T+a['c2w'][t,:3,3];flat=(pt@b)[:,[0,2]];val,vec=np.linalg.eigh(np.cov(flat.T));gap=(val[1]-val[0])/max(val[1]+val[0],1e-12);boundary=bool(mask[0].any() or mask[-1].any() or mask[:,0].any() or mask[:,-1].any());weight=gap**2*(.2 if boundary else 1.)*np.sqrt(mask.sum())*float(np.median(a['confidence'][t,v,u]));weights.append(weight);r=b.T@np.array(raw['entity_to_world'])[t,:3,:3];y.append(float(np.arctan2(r[0,2],r[0,0])));quality.append(dict(frame=t,axis_anisotropy=float(gap),boundary_clipped=boundary))
yaw,report=recover(a['times_s'],y,weights);pose=np.array(e['entity_to_world']);pose[:,:3,:3]=b@Rotation.from_euler('y',yaw).as_matrix();e['entity_to_world']=pose.tolist();e['fit_constraints']={'lock_yaw':True};e['orientation_hypothesis']='dominant same-stage axial mode; semantic front unresolved; not observed per-frame truth';write_json(p/'orientation_report.json',dict(**report,quality=quality,source_init=reference(old/'init/scene.json'),no_cross_stage_evidence=True));write_json(p/'initial_scene.json',s)
fit(p/'initial_scene.json',p/'fit',steps=160,resolution=48,device='cuda',lr=.01,entity_ids=[eid]);policy=read_json(D/'runs/v11_rainy_s20_package_r2/persistence_policy.json');write_json(p/'policy.json',policy);apply(p/'fit/scene.json',p/'policy.json',p/'persistent');print('REPAIR DONE',flush=True)
