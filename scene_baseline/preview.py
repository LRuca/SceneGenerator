"""Depth previews use a fixed full-clip range and retain raw data."""
import numpy as np,imageio.v2 as imageio
from PIL import Image,ImageDraw
from matplotlib import colormaps
from .contracts import load_bundle
from .storage import inside

def colors(d,lo,hi):
    return (colormaps["magma"](np.clip((np.log(np.maximum(d,1e-6))-lo)/max(hi-lo,1e-6),0,1))[...,:3]*255).astype("uint8")

def bundle_preview(manifest):
    m,a=load_bundle(manifest);p=inside(m["geometry"]["path"]).parent
    path=p/"rgb_depth.mp4"
    if path.exists():raise FileExistsError(path)
    log=np.log(a["depth"]);lo=float(log.min());hi=float(log.max())
    d=colors(a["depth"],lo,hi);n,h,w=a["depth"].shape
    frames=[]
    for t in range(n):
        canvas=Image.new("RGB",(640,350),"#141820")
        canvas.paste(Image.fromarray((a["rgb"][t]*255).astype("uint8")).resize((320,320)),(0,30))
        canvas.paste(Image.fromarray(d[t]).resize((320,320)),(320,30))
        ImageDraw.Draw(canvas).text((10,8),f'RGB / DEPTH | frame {a["frame_indices"][t]} | full-clip log range',fill="white")
        frames.append(np.asarray(canvas))
    if len(a["times_s"])>1:fps=1/np.median(np.diff(a["times_s"]))
    else:fps=m["fps"]
    imageio.mimwrite(path,frames,fps=float(fps),codec="libx264",quality=8,macro_block_size=1)
    return str(path)
