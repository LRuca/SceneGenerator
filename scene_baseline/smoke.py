"""Known synthetic geometry: test primitive fitting, coverage and reference white renderer."""
import numpy as np,imageio.v2 as imageio
from PIL import Image,ImageDraw
from .geometry import primitive_depth,box_matrix,mesh_preview,agreement
from .fit import fit_rigid_box
from .storage import new_run,write_json

def synthetic(out):
    p=new_run(out);n=3;h=w=96
    K=np.tile(np.array([[100.,0,48],[0,100.,48],[0,0,1]]),(n,1,1))
    c2w=np.tile(np.eye(4),(n,1,1));c2w[:,0,3]=[-.15,0,.15]
    centers=np.array([[-.18,0,3.],[0,0,3.05],[.18,0,3.1]])
    size=np.array([1.2,.7,.8])
    depth=np.stack([primitive_depth("box",box_matrix(c,size),k,camera,h,w) for c,k,camera in zip(centers,K,c2w)])
    mask=np.isfinite(depth);target=np.where(mask,depth,6.)
    fit=fit_rigid_box(target,mask,K,c2w,centers+[.08,.06,.15],size,max_nfev=60)
    if fit["final_mse"]>=fit["initial_mse"]*.15:raise AssertionError("Synthetic fit did not reduce error enough")
    if min(x["coverage"] for x in fit["per_frame"])<.9:raise AssertionError("Fit missed target")
    frames=[]
    for t in range(n):
        part={"kind":"box","canonical_to_world":fit["matrices"][t]}
        image=mesh_preview([part],K[t],c2w[t],w,h)
        canvas=Image.new("RGB",(384,120),"#141820")
        canvas.paste(Image.fromarray(image).resize((192,96)),(0,24))
        m=(mask[t]*255).astype("uint8");canvas.paste(Image.fromarray(m).convert("RGB").resize((192,96)),(192,24))
        ImageDraw.Draw(canvas).text((4,6),f'Fitted white model / known mask | t{t}',fill="white")
        frames.append(np.asarray(canvas))
    imageio.mimwrite(p/"synthetic_white.mp4",frames,fps=2,codec="libx264",macro_block_size=1)
    Image.fromarray(frames[1]).save(p/"synthetic_white.jpg")
    missing=agreement(np.full((2,2),np.inf),np.ones((2,2)))
    if missing["capped_relative_error_with_misses"]!=1:raise AssertionError("Missing depth not penalized")
    write_json(p/"fit.json",fit);write_json(p/"test_scene.json",{"truth_centers":centers.tolist(),"size":size.tolist(),"c2w":c2w.tolist(),"K":K.tolist()})
    return {"fit":fit,"render":"SceneBlocker reference CPU rasterize","status":"synthetic_component_pass_not_real_baseline"}
