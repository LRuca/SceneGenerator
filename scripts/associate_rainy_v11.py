"""Stage-local candidate association; mask centroids are association cues, never 3D centers."""
from pathlib import Path
import argparse,json
import numpy as np,cv2
from scipy.optimize import linear_sum_assignment
from PIL import Image,ImageDraw
from scene_baseline.storage import new_run,write_json,reference
from scene_baseline.contracts import load_bundle
ap=argparse.ArgumentParser();ap.add_argument('--step',type=int,required=True);a0=ap.parse_args();step=a0.step
R=Path('/home/tione/notebook/home/keithanwang/vlm-vla/gafu');D=R/'sceneblocker/SceneGenerator';gm=D/f'runs/v11_rainy_s{step:02d}_geometry_r1/manifest.json';m,a=load_bundle(gm)
cand=D/f'runs/v11_rainy_s{step:02d}_sam_candidates_r1';p=new_run(D/f'runs/v11_rainy_s{step:02d}_association_r1')
rgb=(a['rgb']*255).astype('uint8');gray=[cv2.cvtColor(x,cv2.COLOR_RGB2GRAY) for x in rgb];N,H,W=a['depth'].shape;tracks=[]
def centroid(mask):
 y,x=np.nonzero(mask);return np.array([np.mean(x),np.mean(y)])
def hist(t,mask):
 hsv=cv2.cvtColor(rgb[t],cv2.COLOR_RGB2HSV);v=cv2.calcHist([hsv],[0,1],mask.astype('uint8'),[12,8],[0,180,0,256]);return v/(np.linalg.norm(v)+1e-6)
for t in range(N):
 with np.load(cand/f'cars_{t:03d}.npz') as z:
  masks=z['masks'];scores=z['scores']
 selected=[]
 for j in np.argsort(scores)[::-1]:
  mask=masks[j]
  if mask.sum()<120:continue
  if any((mask&x[0]).sum()/max((mask|x[0]).sum(),1)>.50 for x in selected):continue
  selected.append((mask,float(scores[j]),centroid(mask),hist(t,mask)))
 active=[i for i,tr in enumerate(tracks) if t-tr['last']<=5];cost=np.full((len(active),len(selected)),1e4)
 for ii,i in enumerate(active):
  tr=tracks[i];last=tr['last'];pm=tr['masks'][last];prev=centroid(pm);pred=prev.copy();flow=False
  pts=cv2.goodFeaturesToTrack(gray[last],mask=cv2.erode(pm.astype('uint8'),np.ones((3,3),np.uint8)),maxCorners=40,qualityLevel=.01,minDistance=5)
  if pts is not None and len(pts)>=3:
   nex,st,err=cv2.calcOpticalFlowPyrLK(gray[last],gray[t],pts,None,winSize=(31,31),maxLevel=4)
   if nex is not None:
    good=st[:,0].astype(bool)&np.isfinite(nex[:,0]).all(1)
    if good.sum()>=3:pred+=np.median(nex[good,0]-pts[good,0],axis=0);flow=True
  if not flow and len(tr['observations'])>=2:
   aa,bb=tr['observations'][-2:];pred+=(centroid(tr['masks'][bb])-centroid(tr['masks'][aa]))*(t-last)/(bb-aa)
  shift=pred-prev;warped=cv2.warpAffine(pm.astype('uint8'),np.array([[1,0,shift[0]],[0,1,shift[1]]],np.float32),(W,H))>0
  size=max(np.sqrt(pm.sum()),25)
  for j,(mask,score,c,h) in enumerate(selected):
   dist=np.linalg.norm(c-pred)/max(size,np.sqrt(mask.sum()));ratio=max(mask.sum(),pm.sum())/min(mask.sum(),pm.sum());overlap=(mask&warped).sum()/max((mask|warped).sum(),1);color=float(np.sum(tr['hist']*h))
   if dist<1.5 and ratio<4.5 and color>.18:cost[ii,j]=.7*dist+.6*(1-overlap)+.4*(1-color)
 assigned=set()
 if len(active) and len(selected):
  rr,cc=linear_sum_assignment(cost)
  for ii,j in zip(rr,cc):
   if cost[ii,j]>1.5:continue
   tr=tracks[active[ii]];mask,score,c,h=selected[j];tr['masks'][t]=mask;tr['last']=t;tr['observations'].append(t);tr['scores'].append(score);tr['hist']=.7*tr['hist']+.3*h;tr['hist']/=np.linalg.norm(tr['hist'])+1e-9;assigned.add(j)
 for j,(mask,score,c,h) in enumerate(selected):
  if j in assigned:continue
  mm=np.zeros((N,H,W),bool);mm[t]=mask;tracks.append(dict(masks=mm,last=t,observations=[t],scores=[score],hist=h))
tracks=[tr for tr in tracks if len(tr['observations'])>=5]
tracks.sort(key=lambda tr:tr['observations'][0]);ids=[f'car_{i+1:02d}' for i in range(len(tracks))]
# Two coarse structural regions are agent image-space proposals, not SAM measurements.
# They are fixed safe interior bands; cars are explicitly excluded before depth fitting.
Y,X=np.mgrid[:H,:W];car_union=np.any(np.stack([tr['masks'] for tr in tracks]),axis=0) if tracks else np.zeros((N,H,W),bool)
ground=np.broadcast_to((Y>int(H*.57))&(X>int(W*.16)),(N,H,W)).copy()&~car_union
background=np.broadcast_to((Y>int(H*.15))&(Y<int(H*.46))&(X>int(W*.18)),(N,H,W)).copy()&~car_union
# Stage5: no identifiable cars/semantics. Only two explicitly uncertain structural bands.
arrays=[tr['masks'] for tr in tracks]+[ground,background];allids=ids+['ground','background'];np.savez_compressed(p/'masks.npz',masks=np.stack(arrays),ids=np.array(allids))
entities=[]
for eid in ids:entities.append(dict(id=eid,label='car; detailed identity pending visual review',motion='rigid',segmentation_prompt='car',confidence=.55,parts=[dict(id=eid+'_body',kind='box',role='coarse shared car envelope')]))
entities.extend([dict(id='ground',label='wet road' if step>5 else 'unresolved lower surface',motion='static',segmentation_prompt='road',confidence=.3 if step==5 else .6,parts=[dict(id='ground_surface',kind='plane',role='ground')]),dict(id='background',label='background wall' if step>5 else 'unresolved upper surface',motion='static',segmentation_prompt='background wall',confidence=.25 if step==5 else .5,parts=[dict(id='background_surface',kind='plane',role='wall')])])
write_json(p/'hypothesis.json',dict(schema='scene_baseline.hypothesis.v1',actor='current_chat_codex; same-stage contact review; script is underdefined',geometry_manifest_sha256=reference(gm)['sha256'],entities=entities,relations=[dict(subject=eid,relation='supported_by',object='ground',status='agent ground-contact prior, not measured contact') for eid in ids],tool_requests=[]))
write_json(p/'association_report.json',dict(geometry_manifest=reference(gm),source_candidates=reference(cand/'manifest.json'),tracks=[dict(id=eid,observed_frames=tr['observations'],mean_score=float(np.mean(tr['scores'])),max_pixels=int(tr['masks'].sum((1,2)).max())) for eid,tr in zip(ids,tracks)],structural_regions='Agent-proposed image-space interior bands; not SAM observations; depth is measured VGGT prediction. Source is current stage only.',vehicle_center_policy='mask centroid used only for 2D association; geometry centers are fitted shared shapes',script_fidelity='parked identity/alley emergence unverified; do not force two instances',status='pending_current_chat_visual_track_review'))
colors=[(250,190,35),(45,110,245),(235,65,65),(45,185,120),(170,90,235),(245,125,35),(35,185,220),(230,95,165)]
board=Image.new('RGB',(5*260,max(1,len(ids))*285),(22,26,32));draw=ImageDraw.Draw(board)
for i,tr in enumerate(tracks):
 visible=tr['observations'];samples=[visible[j] for j in np.linspace(0,len(visible)-1,5).round().astype(int)]
 for k,t in enumerate(samples):
  arr=rgb[t].copy();mask=tr['masks'][t];arr[mask]=(arr[mask]*.5+np.array(colors[i%len(colors)])*.5).astype('uint8');board.paste(Image.fromarray(arr).resize((260,260)),(k*260,i*285+25));draw.text((k*260+4,i*285+5),f'{ids[i]} / t{t}',fill='white')
if not tracks:draw.text((15,15),'No identifiable car track: no later-stage vehicle masks copied.',fill='white')
board.save(p/'track_contact.jpg');print('STAGE',step,'tracks',[(eid,len(tr['observations'])) for eid,tr in zip(ids,tracks)],flush=True)
