from pathlib import Path
import json,time,gc
import numpy as np
import imageio.v2 as iio
from PIL import Image,ImageDraw
from matplotlib import colormaps
from scene_baseline.vggt import export
from scene_baseline.contracts import load_bundle
R=Path('/home/tione/notebook/home/keithanwang/vlm-vla/gafu');D=R/'sceneblocker/SceneGenerator';source=D/'runs/v10_courtyard_early_inputs_r1'
for step in [5,10]:
 dest=D/f'runs/v10_courtyard_s{step:02d}_geometry_r1'
 export(source/f'rgb_step_{step:03d}.npy',source/'run.json',dest)
 import torch
 gc.collect();torch.cuda.empty_cache()
 m,a=load_bundle(dest/'manifest.json');d=a['depth'];rgb=(a['rgb']*255).clip(0,255).astype('uint8')
 # Each stage has arbitrary VGGT scale; use one whole-clip scale, never frame-wise.
 scale=float(np.median(d[0]));norm=d/scale;lo,hi=np.quantile(norm,[.005,.995]);hi=max(float(hi),float(lo)+1e-6)
 color=(colormaps['magma'](((norm-lo)/(hi-lo)).clip(0,1))[...,:3]*255).astype('uint8')
 for kind in ['depth','rgb_depth']:
  width=518 if kind=='depth' else 1036
  with iio.get_writer(dest/f'{kind}_step_{step:03d}.mp4',fps=16,codec='libx264',macro_block_size=1,quality=8) as writer:
   for t in range(len(d)):
    im=Image.new('RGB',(width,560),(20,24,32));draw=ImageDraw.Draw(im)
    draw.text((8,7),f'Courtyard / Step{step} of50 / frame{t} | dark near, bright far',fill='white')
    draw.text((8,23),'Whole-clip fixed color scale; no far geometry discarded',fill='white')
    if kind=='rgb_depth':im.paste(Image.fromarray(rgb[t]),(0,42))
    im.paste(Image.fromarray(color[t]),(0 if kind=='depth' else 518,42));writer.append_data(np.array(im))
 (dest/'depth_display.json').write_text(json.dumps(dict(raw_depth_saved=True,normalization='first-frame median once per clip; color range .5-99.5% across full clip, clamp colors only, never remove depth pixels',scale=scale,color_limits=[float(lo),hi],relative_arbitrary_scale=True),indent=2))
 print('DEPTH READY',step,flush=True)
