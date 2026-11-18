"""Prepare explicit Seedance 2.5 multi-reference request; no hidden media uploads."""
import argparse,base64,json
import numpy as np
from PIL import Image
from scene_baseline.storage import BASE,new_run,reference,write_json,verify
from scene_baseline.scene import load_scene
from scene_baseline.geometry import mesh_preview,scene_parts
from scene_baseline.envelope_v6 import projected_mask
def run(scene_path,out):
 s,m,a=load_scene(scene_path);p=new_run(out);refs=p/"references";refs.mkdir()
 n,H,W=a["depth"].shape;idx=np.linspace(0,n-1,5).round().astype(int);items=[]
 for j,t in enumerate(idx):
  frame=mesh_preview(scene_parts(s,t),a["K"][t],a["c2w"][t],h=H,w=W)
  dst=refs/f"{j+1:02d}_blocking_t{t:02d}.png";Image.fromarray(frame).resize((720,720)).save(dst)
  items.append({"file":reference(dst),"role":"geometry_and_timing","frame":int(t),"output_time_s":float(j)})
 md=__import__("scene_baseline.storage",fromlist=["read_json"]).read_json(verify(s["masks_manifest"]))
 with np.load(verify(md["data"])) as z:masks=z["masks"];ids=z["ids"].tolist()
 # Last upright appearance only; crop is not a motion-reference video.
 t=n-1;rgb=(a["rgb"][t]*255).astype("uint8");pm=masks[ids.index("person"),t];v,u=np.nonzero(pm)
 box=(max(int(u.min())-8,0),max(int(v.min())-8,0),min(int(u.max())+9,W),min(int(v.max())+9,H))
 actor=Image.fromarray(rgb).crop(box);canvas=Image.new("RGB",(720,720),(236,231,220));actor.thumbnail((600,680));canvas.paste(actor,((720-actor.width)//2,(720-actor.height)//2));canvas.save(refs/"06_actor_appearance.png")
 items.append({"file":reference(refs/"06_actor_appearance.png"),"role":"person_identity_clothing_only","frame":t,"crop_box":list(box)})
 for num,eid in [(7,"wall"),(8,"ground")]:
  mm=masks[ids.index(eid),t];v,u=np.nonzero(mm)
  # A source-pixel patch chosen inside the reviewed component; no fabricated background.
  good=np.argwhere(mm)
  if len(good)==0:raise ValueError("Missing component image")
  from scipy.ndimage import distance_transform_edt
  distance=distance_transform_edt(mm);y,x=np.unravel_index(np.argmax(distance),mm.shape)
  r=max(2,min(25,int(distance[y,x]*.65)));x=int(np.clip(x,r,W-r));y=int(np.clip(y,r,H-r))
  im=Image.fromarray(rgb).crop((x-r,y-r,x+r,y+r)).resize((720,720))
  im.save(refs/f"{num:02d}_{eid}_material.png");items.append({"file":reference(refs/f"{num:02d}_{eid}_material.png"),"role":eid+"_material","crop_box":[x-r,y-r,x+r,y+r],"frame":t})
 Image.fromarray((a["rgb"][0]*255).astype("uint8")).resize((720,720)).save(refs/"09_scene_appearance.png")
 items.append({"file":reference(refs/"09_scene_appearance.png"),"role":"appearance_and_lighting_only","frame":0})
 prompt="""Generate one continuous four-second photorealistic full-body shot of exactly one adult in a simple empty room. Output square 1:1, 720p, no audio.
REFERENCE ROLES: [Image 1] through [Image 5] are the same reconstructed white blocking scene at chronological output times 0, 1, 2, 3, and 4 seconds. Use them only for camera framing, floor contact, coarse body position, knee/waist bending and timing. They are sequential poses of ONE person, not five people. [Image 6] shows the person's appearance and clothing. [Image 7] is the rear-wall material patch. [Image 8] is the floor material patch. [Image 9] shows the original overall room appearance and soft lighting; use its materials and composition, not as an extra shot or a collage.
SCENE AND APPEARANCE: one short-haired adult wearing a bright royal-blue long-sleeved crew-neck top and plain dark navy trousers, barefoot. Matte warm off-white rear wall, narrow pale skirting board, flat light beige/tan floor. Broad diffuse indoor daylight, gentle shadow below the feet, natural skin and cloth. No furniture or props.
CAMERA: stationary tripod at approximately waist/chest height, level optical axis, normal perspective. Full body and feet stay visible with the subject near the center. Preserve the framing and floor/wall boundary of the blocking references; do not orbit, pan, zoom or cut.
ACTION: begin slightly bent forward, head following the upper torso. During 0–1 second bend farther at the waist into the deepest forward bend seen in Image 2. Keep both feet planted at the same floor position. Knees flex only gently as the reference indicates. During 1–2 seconds smoothly begin to rise; during 2–3 seconds continue to straighten. By 3–4 seconds stand nearly upright and settle with small natural movement. Keep hands near the front of the thighs/knees while bent, then loosely together in front when upright. Maintain plausible balance, stable body proportions, continuous motion and natural smooth knee/waist articulation.
TRANSLATE THE PROXY: replace the sphere with a natural attached head; replace the smooth coarse body envelope with a normally proportioned clothed person and two natural legs/arms. The envelope is a coarse occupied-space guide, not a literal tube or an inflated body. Keep its root position and large-scale action. Do not reproduce white plastic, faceted surfaces, disconnected pieces or extra joints.
CONSISTENCY: same person and clothing throughout, steady background, stable feet, no sliding, no teleportation, no flickering geometry, no additional people, no camera shake, no text, no labels, no reference-image panels or montage in the output."""
 (p/"prompt.txt").write_text(prompt)
 params=dict(prompt=prompt,resolution="720p",ratio="1:1",duration=4,seed=40123,generate_audio=False,watermark=False,
  reference_images=[{"url":"data:image/png;base64,"+base64.b64encode(verify(x["file"]).read_bytes()).decode()} for x in items])
 write_json(p/"request.json",{"model":"seedance-2.5","generation_type":"image2video","params":params})
 write_json(p/"manifest.json",dict(source_scene=reference(scene_path),model="seedance-2.5",reference_images=items,prompt=reference(p/"prompt.txt"),request=reference(p/"request.json"),reference_video_submitted=False,
  scope="Multi-image re-generation from blocking key poses + component appearance; no full source RGB video. Cannot establish video-control fidelity from one sample.",
  duration_policy="source motion rescaled from ~3s to API minimum4s"))
 tiles=[]
 for x in items:
  im=Image.open(verify(x["file"])).resize((240,240));tiles.append(im)
 board=Image.new("RGB",(720,720))
 for j,im in enumerate(tiles):board.paste(im,((j%3)*240,(j//3)*240))
 board.save(p/"reference_board.jpg")
 (p/"index.html").write_text('<!doctype html><meta charset="utf-8"><title>Seedance 2.5 输入包</title><style>body{font:16px system-ui;max-width:1100px;margin:30px auto}img{max-width:100%}pre{white-space:pre-wrap}</style><h1>Seedance 2.5：白模与组件参考图</h1><p>1–5：白模关键帧；6：人物外观；7：墙材质；8：地面材质；9：场景光照与外观。尚无视频参考输入，动作由关键帧和文字约束。</p><img src="reference_board.jpg"><h2>详细 prompt</h2><pre>'+__import__("html").escape(prompt)+'</pre><p><a href="manifest.json">输入来源与角色</a></p>')
 return p
if __name__=="__main__":
 ap=argparse.ArgumentParser();ap.add_argument("--scene",required=True);ap.add_argument("--out",required=True);a=ap.parse_args();run(a.scene,a.out)
