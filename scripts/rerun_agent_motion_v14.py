from pathlib import Path
import json,numpy as np
from scene_baseline.storage import read_json,write_json,new_run,reference,verify
from scene_baseline.scene import load_scene
from scene_baseline.motion_policy_v1 import apply
from scene_baseline.motion_policy_v1 import fit_locked as fit
D=Path('/home/tione/notebook/home/keithanwang/vlm-vla/gafu/sceneblocker/SceneGenerator');p=new_run(D/'runs/v14_rainy_s20_agent_motion_r1');src=D/'runs/v12_rainy_s20_orientation_r1/initial_scene.json';s=read_json(src);ids=[e['id'] for e in s['entities']]
rows=[]
for e in s['entities']:
 if e['id']=='car_01':motion='static';ev='Agent semantic judgment from script/world-state review: right-edge yellow proxy is parked/world-stationary; apparent screen displacement is attributed to camera/parallax unless contradictory stable body correspondences are found.';alt='rigid_dynamic'
 elif e['id'].startswith('car'):motion='rigid_dynamic';ev='Agent semantic judgment: tracked traffic vehicle exhibits instance-local displacement across same-stage frames; not used to define camera.';alt='static'
 else:motion='static';ev='Agent semantic judgment: road/background structural surface is world-stationary.';alt=None
 rows.append(dict(id=e['id'],motion=motion,confidence='agent qualitative; not fitted',evidence=ev,alternative=alt,allowed_dof='shared_pose' if motion=='static' else 'translation_yaw_with_soft_smoothness',source_stage=20))
dec=dict(source_scene=reference(src),schema='scene_baseline.agent_motion_decision.v1',actor='current_chat_codex',entities=rows,decision_basis='script + full same-stage sequence + entity identity review; fitting objective cannot relabel motion')
write_json(p/'agent_motion_decisions.json',dec);base=apply(src,p/'agent_motion_decisions.json',p/'motion_locked');fit(p/'motion_locked/scene.json',p/'fit',steps=160,resolution=48,device='cuda',lr=.01);print('DONE',p,flush=True)
