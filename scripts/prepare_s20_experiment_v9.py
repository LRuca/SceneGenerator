from pathlib import Path
import json,copy
import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt
from scene_baseline.scene import load_scene
from scene_baseline.storage import reference,verify,read_json,write_json
R=Path('/home/tione/notebook/home/keithanwang/vlm-vla/gafu');D=R/'sceneblocker/SceneGenerator'
scene=D/'runs/v4_courtyard_r1/step20_visible_soft_containment_free/scene.json';s,m,a=load_scene(scene)
exp=D/'experiments/appearance_swap_s20_v9';exp.mkdir(exist_ok=False)
orig=D/'runs/v9_s20_original_appearance_r1';orig.mkdir();(orig/'references').mkdir()
md=read_json(verify(s['masks_manifest']))
with np.load(verify(md['data'])) as z:masks=z['masks'];ids=z['ids'].tolist()
exclusive=masks & (masks.sum(0)[None]==1);refs=[]
for idx,eid in enumerate(['forklift','parked_truck','gray_obstacle','wall','ground'],1):
 mm=exclusive[ids.index(eid)];dst=orig/'references'/f'{idx:02d}_{eid}.png'
 if idx<=2:
  t=int(np.argmax(mm.sum((1,2))));mask=mm[t];y,x=np.nonzero(mask);box=[int(x.min()),int(y.min()),int(x.max()+1),int(y.max()+1)]
  rgb=(a['rgb'][t]*255).astype('uint8');rgb[~mask]=224;im=Image.fromarray(rgb).crop(box);scale=560/max(im.size);im=im.resize(tuple(max(1,round(v*scale)) for v in im.size));canvas=Image.new('RGB',(720,720),(224,224,224));canvas.paste(im,((720-im.width)//2,(720-im.height)//2));canvas.save(dst)
  row=dict(entity=eid,file=reference(dst),role='single isolated centered appearance only',source_frame=t,crop=box,outside_mask_source_pixels=0,geometry_scale_preserved=False)
 else:
  best=None
  for t0 in range(len(mm)):
   dist=distance_transform_edt(np.pad(mm[t0],1))[1:-1,1:-1];y0,x0=np.unravel_index(np.argmax(dist),dist.shape)
   if best is None or dist[y0,x0]>best[0]:best=(float(dist[y0,x0]),t0,int(y0),int(x0))
  distance,t,y,x=best;r=max(1,min(20,int(distance*.6)));box=[x-r,y-r,x+r+1,y+r+1]
  assert 0<=x-r and x+r<mm.shape[2] and 0<=y-r and y+r<mm.shape[1] and mm[t,y-r:y+r+1,x-r:x+r+1].all()
  Image.fromarray((a['rgb'][t]*255).astype('uint8')).crop(box).resize((720,720)).save(dst)
  row=dict(entity=eid,file=reference(dst),role='interior material patch only',source_frame=t,crop=box,all_source_pixels_inside_exclusive_mask=True)
 refs.append(row)
common='''Create one continuous four-second video of a miniature practical toy set, square 1:1, 720p, without audio.
[Video 1] is the sole source of camera trajectory, framing, layout, chronological coarse object poses, relative scale, ground contact, depth order, occlusion and phase timing. The saturated YELLOW proxy is moving instance I1; the saturated BLUE proxy is stationary-in-world instance I2. These are identity colors, not required final body colors. Gray geometry is structural context, including the separate kiosk, ground and rear wall. Preserve each identity through all frames and partial occlusion. The boxes are coarse occupancy guides, not literal block-shaped vehicles; translate them into the specified vehicle types while keeping their roots, path, coarse occupied region and timing. Keep exactly two vehicles and one kiosk. Do not invent extra turns, motion phases or departures. The world-stationary vehicle changes screen position only because the camera moves. Follow the lateral camera translation and resulting parallax; do not stabilize it away, replace it with a zoom or add a cut.
[Image 1] and [Image 2] define the two vehicles' appearance only; their pose, facing direction, image position and scale must not override Video 1. [Image 3], [Image 4] and [Image 5] are isolated interior material swatches; they contain no surface dimensions or layout. Infer surface extent, support and occlusion solely from Video 1. Natural daylight, coherent cast/contact shadows, consistent miniature scale and focus. No people, extra vehicles, identity swaps, teleportation, stretching, floating wheels, flicker, text, panels, logos or watermark. Output one continuous shot, not a collage.
'''
orig_description='''APPEARANCE BINDINGS: I1 / [Image 1] is one yellow toy forklift with black wheels, operator compartment and upright lifting structure. I2 / [Image 2] is one blue toy pickup truck with black wheels. [Image 3] supplies gray ribbed metal for the kiosk; [Image 4] supplies warm brown brick for the rear wall; [Image 5] supplies pale gray paved ground. Preserve this palette and these two vehicle types while following the proxy geometry and camera.\n'''
new_description='''APPEARANCE BINDINGS: I1 / [Image 1] is one glossy forest-green vintage two-door toy coupe with round headlights, chrome trim and bumpers, small black tires and silver hubs. It is a passenger car, with no forklift mast, forks or cargo attachment. I2 / [Image 2] is one white box-bodied toy delivery van, with a short cab, dark glazing, black tires and an enclosed rectangular cargo body; it is not an open-bed pickup. [Image 3] supplies deep teal painted metal for the kiosk; [Image 4] supplies warm ivory lime plaster for the rear wall, replacing visible brickwork; [Image 5] supplies warm reddish sandstone for the ground, replacing the pale gray paving. The green coupe is always the yellow proxy identity; the white van is always the blue proxy identity. Do not copy the source cutouts' orientation or their backgrounds.\n'''
(orig/'prompt.txt').write_text(common+orig_description)
write_json(orig/'manifest.json',dict(source_scene=reference(scene),source_geometry=reference(verify(s['geometry_manifest'])),references=refs,no_source_rgb_video=True,no_source_full_frame_image=True,no_first_frame_image=True,source_diffusion_step=20))
new=D/'runs/v9_s20_swapped_appearance_r1';new.mkdir();(new/'references').mkdir()
am=read_json(D/'runs/v9_appearance_assets_r1/assets_manifest.json');newrefs=[]
import shutil
for row in am['assets']:
 src=D/'runs/v9_appearance_assets_r1/references'/row['file'];dst=new/'references'/src.name;shutil.copyfile(src,dst);newrefs.append({**row,'file':reference(dst)})
(new/'prompt.txt').write_text(common+new_description)
write_json(new/'manifest.json',dict(source_scene=reference(scene),references=newrefs,no_source_rgb_video=True,no_source_full_frame_image=True,no_first_frame_image=True,source_diffusion_step=20,asset_sources=reference(D/'runs/v9_appearance_assets_r1/assets_manifest.json')))
base=read_json(D/'experiments/proxy_encoding_v8/decisions.json')
for row in base['entities']:
 row['evidence']=row['evidence'].replace('Same-stage existing sequence','Step20 existing sequence')
 row['prompt_id']={'forklift':'I1','parked_truck':'I2'}.get(row['id'],row['id'])
newdec=copy.deepcopy(base)
for row in newdec['entities']:
 row['appearance_label']={'forklift':'green vintage toy coupe','parked_truck':'white box-bodied toy delivery van'}.get(row['id'],row['id'])
write_json(exp/'original_decisions.json',base);write_json(exp/'swapped_decisions.json',newdec)
length=next(e for e in s['entities'] if e['id']=='forklift')['parts'][0]['canonical_to_entity'][0][0];shift=-.6*length
for name,dec,src in [('original','original_decisions.json',orig),('swapped','swapped_decisions.json',new)]:
 config=dict(schema='scene_baseline.pipeline.v1',tasks=[dict(id='identity',kind='proxy.encode',arguments=dict(scene=str(scene),decisions=str(exp/dec))),dict(id='blocking',kind='blocking.package',arguments=dict(scene={'ref':'identity.scene'},source_package=str(src),displacement=shift,width=320))])
 write_json(exp/(name+'_pipeline.json'),config)
write_json(exp/'comparison_design.json',dict(source_diffusion_step=20,source_scene=reference(scene),same_geometry=True,same_camera_translation=shift,same_seed=40124,appearance_only_no_full_scene_inputs=True,variables=['vehicle types and appearance','kiosk/wall/ground material patches','matching appearance/type prompt clauses']))
print('Prepared Step20 paired experiment; camera shift',shift)
