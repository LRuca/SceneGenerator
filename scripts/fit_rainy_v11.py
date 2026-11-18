from pathlib import Path
import argparse,numpy as np
from scene_baseline.storage import read_json,write_json,new_run,reference
from scene_baseline.masks import import_masks
from scene_baseline.scene import initialize
from scene_baseline.fit_scene import fit
ap=argparse.ArgumentParser();ap.add_argument('--step',type=int,required=True);a=ap.parse_args();step=a.step
D=Path('/home/tione/notebook/home/keithanwang/vlm-vla/gafu/sceneblocker/SceneGenerator');src=D/f'runs/v11_rainy_s{step:02d}_association_r1';p=new_run(D/f'runs/v11_rainy_s{step:02d}_reconstruction_r1');gm=D/f'runs/v11_rainy_s{step:02d}_geometry_r1/manifest.json'
h=read_json(src/'hypothesis.json')
with np.load(src/'masks.npz') as z:ms=z['masks'];ids=z['ids'].tolist()
notes=[]
if step==10:
 ix=ids.index('car_04');ms=np.delete(ms,ix,0);ids.pop(ix);h['entities']=[e for e in h['entities'] if e['id']!='car_04'];notes.append('Reject car_04: foreground mask includes broad road/background and lacks stable vehicle boundary. Foreground vehicle remains unresolved.')
if step in [20,40]:
 ms[ids.index('car_03')]|=ms[ids.index('car_04')];ix=ids.index('car_04');ms=np.delete(ms,ix,0);ids.pop(ix);h['entities']=[e for e in h['entities'] if e['id']!='car_04'];notes.append('Merge car_03/car_04: same silver sedan crossing left-to-right behind opposite-direction white hatchback, continuous same-stage visual identity across occlusion.')
labels={10:{'car_01':'silver car crossing right to left','car_02':'dim silver sedan crossing left to right','car_03':'dark car at right edge; world-stationarity unverified'},20:{'car_01':'dark car at right edge; world-stationarity unverified','car_02':'white compact hatchback crossing right to left','car_03':'silver sedan crossing left to right','car_05':'large dark foreground sedan crossing left to right'},40:{'car_01':'dark car at right edge; world-stationarity unverified','car_02':'white compact hatchback crossing right to left','car_03':'silver sedan crossing left to right','car_05':'large dark foreground sedan crossing left to right'}}
for e in h['entities']:
 if e['id'].startswith('car'):e['label']=labels[step][e['id']]
h['relations']=[r for r in h['relations'] if r['subject'] in ids];write_json(p/'hypothesis.json',h);np.savez_compressed(p/'reviewed_masks.npz',masks=ms,ids=np.array(ids))
write_json(p/'agent_review.json',dict(actor='current_chat_codex',source=reference(src/'track_contact.jpg'),stage=step,notes=notes,decision='accept listed stage-local candidates with explicit geometry uncertainty',no_later_stage_evidence=True))
import_masks(gm,p/'hypothesis.json',p/'reviewed_masks.npz',p/'masks');initialize(gm,p/'hypothesis.json',p/'masks/manifest.json',p/'init');fit(p/'init/scene.json',p/'fit',steps=160,resolution=48,device='cuda',lr=.015)
print('FIT DONE',step,flush=True)
