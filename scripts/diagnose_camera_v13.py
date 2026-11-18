from pathlib import Path
import json,numpy as np,cv2
from PIL import Image,ImageDraw
from scene_baseline.scene import load_scene
from scene_baseline.storage import new_run,read_json,verify,write_json,reference
from scene_baseline.contracts import load_bundle
D=Path('/home/tione/notebook/home/keithanwang/vlm-vla/gafu/sceneblocker/SceneGenerator');p=new_run(D/'runs/v13_s20_motion_diagnosis_r2');s,m,a=load_scene(D/'runs/v12_rainy_s20_orientation_r1/review/scene.json');md=read_json(verify(s['masks_manifest']))
with np.load(verify(md['data'])) as z:ms=z['masks'];ids=z['ids'].tolist()
rgb=(a['rgb']*255).astype('uint8');gray=[cv2.cvtColor(x,cv2.COLOR_RGB2GRAY) for x in rgb];N,H,W=a['depth'].shape;car=ms[ids.index('car_01')];union=np.any(ms[[i for i,x in enumerate(ids) if x.startswith('car')]],0);Y,X=np.mgrid[:H,:W];bg=np.broadcast_to((Y>.13*H)&(Y<.46*H)&(X>.18*W),(N,H,W)).copy()&~union
rows=[]
for t in range(N):
 v,u=np.nonzero(car[t]);world=[]
 rays=np.column_stack([u,v,np.ones(len(u))])@np.linalg.inv(a['K'][t]).T;cam=rays*a['depth'][t,v,u,None];world=cam@a['c2w'][t,:3,:3].T+a['c2w'][t,:3,3]
 rows.append(dict(frame=t,bbox=[int(u.min()),int(v.min()),int(u.max()),int(v.max())],area=len(u),depth_quantiles=np.percentile(a['depth'][t,v,u],[10,50,90]).tolist(),visible_world_median=np.median(world,0).tolist(),confidence_median=float(np.median(a['confidence'][t,v,u]))))
results=[];visuals=[];pairs={}
for kind,masks in [('yellow',car),('background',bg)]:
 for t0 in [0,12,24,32]:
  interior=cv2.erode(masks[t0].astype('uint8'),np.ones((7,7),np.uint8));pts=cv2.goodFeaturesToTrack(gray[t0],maxCorners=160,qualityLevel=.008,minDistance=6,mask=interior)
  if pts is None:continue
  pts=pts[:,0];cur=pts.copy();valid=np.ones(len(pts),bool);pix=np.rint(pts).astype(int);rays=np.column_stack([pts,np.ones(len(pts))])@np.linalg.inv(a['K'][t0]).T;z0=a['depth'][t0,pix[:,1],pix[:,0]];world=rays*z0[:,None];world=world@a['c2w'][t0,:3,:3].T+a['c2w'][t0,:3,3];prev=t0
  for t in range(t0+1,min(N,t0+17)):
   nex,st,err=cv2.calcOpticalFlowPyrLK(gray[prev],gray[t],cur[:,None].astype('float32'),None,winSize=(25,25),maxLevel=4);back,bs,_=cv2.calcOpticalFlowPyrLK(gray[t],gray[prev],nex,None,winSize=(25,25),maxLevel=4);nex=nex[:,0];valid&=(st[:,0]>0)&(bs[:,0]>0)&(np.linalg.norm(back[:,0]-cur,axis=1)<1)&(err[:,0]<20)&np.isfinite(nex).all(1);valid&=(nex[:,0]>=2)&(nex[:,0]<W-2)&(nex[:,1]>=2)&(nex[:,1]<H-2);cur=nex;prev=t
   pc=(world-a['c2w'][t,:3,3])@a['c2w'][t,:3,:3];uv=pc@a['K'][t].T;uv=uv[:,:2]/uv[:,2:];ij=np.rint(cur).astype(int);ij[:,0]=ij[:,0].clip(0,W-1);ij[:,1]=ij[:,1].clip(0,H-1);good=valid&masks[t,ij[:,1],ij[:,0]]
   if good.sum()<4:continue
   obsray=np.column_stack([cur,np.ones(len(cur))])@np.linalg.inv(a['K'][t]).T;zt=a['depth'][t,ij[:,1],ij[:,0]];pw=obsray*zt[:,None];pw=pw@a['c2w'][t,:3,:3].T+a['c2w'][t,:3,3];flow=cur-pts;res=cur-uv
   if t in [t0+4,t0+8,t0+16]:
    pairs[(kind,t0,t)]=(world[good].copy(),cur[good].copy())
    results.append(dict(kind=kind,anchor=t0,target=t,points=int(good.sum()),observed_flow_median=np.median(flow[good],0).tolist(),static_reprojection_median_px=float(np.median(np.linalg.norm(res[good],axis=1))),median_depth_ratio=float(np.median(zt[good]/z0[good])),median_3d_correspondence_displacement=float(np.median(np.linalg.norm(pw[good]-world[good],axis=1))),static_projection_flow_median=np.median((uv-pts)[good],0).tolist()))
    if t==t0+16:
     im=Image.fromarray(rgb[t]);dr=ImageDraw.Draw(im)
     for q,obs in zip(uv[good],cur[good]):
      x,y=obs;dr.ellipse((x-2,y-2,x+2,y+2),fill='cyan');dr.line((float(q[0]),float(q[1]),float(x),float(y)),fill='yellow',width=1)
     visuals.append((kind,t0,t,im))
# Background-only camera correction diagnostic, with held-out feature points.
camera_tests=[]
for (kind,t0,t),(world,uv) in pairs.items():
 if kind!='background' or len(world)<15:continue
 test=np.arange(len(world))%3==0;train=~test
 ok,rv,tv,inliers=cv2.solvePnPRansac(world[train].astype('float64'),uv[train].astype('float64'),a['K'][t].astype('float64'),None,iterationsCount=200,reprojectionError=2.,confidence=.999,flags=cv2.SOLVEPNP_EPNP)
 if not ok or inliers is None or len(inliers)<8:continue
 ix=inliers[:,0];rv,tv=cv2.solvePnPRefineLM(world[train][ix].astype('float64'),uv[train][ix].astype('float64'),a['K'][t].astype('float64'),None,rv,tv)
 R=cv2.Rodrigues(rv)[0];c=-R.T@tv[:,0]
 def error(points,target,new):
  pc=points@R.T+tv[:,0] if new else (points-a['c2w'][t,:3,3])@a['c2w'][t,:3,:3]
  q=pc@a['K'][t].T;return float(np.median(np.linalg.norm(q[:,:2]/q[:,2:]-target,axis=1)))
 row=dict(anchor=t0,target=t,background_train_inliers=len(ix),background_heldout_points=int(test.sum()),background_error_before=error(world[test],uv[test],False),background_error_after=error(world[test],uv[test],True),camera_translation_correction=float(np.linalg.norm(c-a['c2w'][t,:3,3])))
 if ('yellow',t0,t) in pairs:
  wy,uy=pairs[('yellow',t0,t)];row.update(yellow_points=len(wy),yellow_static_error_before=error(wy,uy,False),yellow_static_error_after=error(wy,uy,True))
 camera_tests.append(row)
write_json(p/'camera_tests.json',dict(tests=camera_tests,limitation='PnP on predicted anchor depth, fixed K, background-only points with point holdout, no temporal regularity; not calibrated camera truth'))
print('CAMERA TESTS',json.dumps(camera_tests,indent=2))
# Same-stage frame contact: no labels borrowed from a later stage.
board=Image.new('RGB',(4*320,3*350),(25,28,33));dd=ImageDraw.Draw(board)
for k,t in enumerate([0,12,24,28,30,32,34,36,39,42,45,48]):
 arr=rgb[t].copy();arr[car[t]]=(arr[car[t]]*.5+np.array([250,190,35])*.5).astype('uint8');board.paste(Image.fromarray(arr).resize((320,320)),(k%4*320,k//4*350+30));dd.text((k%4*320+5,k//4*350+5),f'frame{t} / yellow observed mask',fill='white')
board.save(p/'yellow_mask_contact.jpg');board=Image.new('RGB',(4*320,max(1,int(np.ceil(len(visuals)/4)))*350),(25,28,33));dd=ImageDraw.Draw(board)
for k,(kind,t0,t,im) in enumerate(visuals):board.paste(im.resize((320,320)),(k%4*320,k//4*350+30));dd.text((k%4*320+5,k//4*350+5),f'{kind} {t0}->{t} cyanLK/yellowresidual',fill='white')
board.save(p/'track_residuals.jpg');write_json(p/'report.json',dict(frame_stats=rows,correspondence_tests=results,source_geometry=s['geometry_manifest'],limitations=['LK consistency is not physical correspondence proof','background is agent ROI; no ground truth','static residual tests joint VGGT depth/camera and correspondences, not camera alone']))
print(json.dumps(results,indent=2));print('frame stats',json.dumps([rows[t] for t in [0,24,32,40,48]],indent=2))
