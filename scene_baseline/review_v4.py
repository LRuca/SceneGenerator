"""Full-frame source/global render comparisons, trajectories and geometry diagnostics."""
import numpy as np
from PIL import Image,ImageDraw
import imageio.v2 as imageio
from .storage import new_run,write_json,reference,verify,read_json
from .scene import load_scene
from .geometry import scene_parts,part_mesh,mesh_preview,render_parts
from .global_views import look_at
from .physical_audit import audit

def title(a,label):
    im=Image.fromarray(a);d=ImageDraw.Draw(im);d.rectangle((0,0,im.width,20),fill="#152030");d.text((4,4),label,fill="white");return np.asarray(im)

def export(old_path,new_path,out,width=224):
    old,m,a=load_scene(old_path);new,_,_=load_scene(new_path)
    if old["geometry_manifest"]!=new["geometry_manifest"]:raise ValueError("Cross-stage comparison")
    p=new_run(out);n,H,W=a["depth"].shape;h=round(width*H/W);sx=width/W;sy=h/H
    A=np.array([[sx,0,(sx-1)/2],[0,sy,(sy-1)/2],[0,0,1.]])
    with np.load(verify(read_json(verify(new["masks_manifest"]))["data"])) as z:masks=z["masks"];ids=z["ids"].tolist()
    clouds=[]
    for s in [old,new]:
        for t in [0,n//2,n-1]:
            for part in scene_parts(s,t):clouds.append(part_mesh(part)[0])
    xyz=np.concatenate(clouds);lo,hi=np.quantile(xyz,[.005,.995],axis=0);ctr=(lo+hi)/2;rad=max(float(np.linalg.norm(hi-lo)/2),.1)
    up=np.asarray(new["ground_normal"],dtype=float);up/=np.linalg.norm(up)
    right=a["c2w"][0,:3,0].copy();right-=up*(right@up);right/=np.linalg.norm(right)
    front=np.cross(right,up);gc=look_at(ctr+rad*(front*2.2+right*.8+up*.9),ctr,up)
    gK=np.array([[width*.95,0,(width-1)/2],[0,width*.95,(h-1)/2],[0,0,1.]])
    def camera_overlay(frame,t):
        im=Image.fromarray(frame);draw=ImageDraw.Draw(im)
        camxyz=a["c2w"][:,:3,3];v=(camxyz-gc[:3,3])@gc[:3,:3];proj=v@gK.T
        xy=proj[:,:2]/proj[:,2:].clip(1e-6)
        for j in range(1,n):
            if min(v[j-1,2],v[j,2])>0:draw.line([tuple(xy[j-1]),tuple(xy[j])],fill="#4ddfb0",width=2)
        if v[t,2]>0:
            x,y=xy[t];draw.ellipse((x-3,y-3,x+3,y+3),fill="#ffee60")
        return np.asarray(im)
    source=[];globalv=[];white=[];metrics={};depth_errors=[]
    for t in range(n):
        K=A@a["K"][t];cam=a["c2w"][t]
        rgb=np.array(Image.fromarray((a["rgb"][t]*255).astype("uint8")).resize((width,h)))
        op=scene_parts(old,t);np_=scene_parts(new,t)
        ow=mesh_preview(op,K,cam,h=h,w=width);nw=mesh_preview(np_,K,cam,h=h,w=width)
        rd,lab=render_parts(np_,K,cam,h,width)
        td=np.array(Image.fromarray(a["depth"][t]).resize((width,h),Image.Resampling.NEAREST))
        err=np.minimum(np.abs(rd-td)/td,.5);err[~np.isfinite(err)]=.5
        heat=np.stack([err*510,80*(1-err*2),255*(1-err*2)],-1).clip(0,255).astype("uint8")
        source.append(np.concatenate([title(rgb,f"RGB f{t}"),title(ow,"comparison baseline"),title(nw,"candidate coarse geometry"),title(heat,"relative depth error 0..50%")],1));white.append(nw)
        globalv.append(np.concatenate([title(camera_overlay(mesh_preview(op,gK,gc,h=h,w=width),t),"baseline global; green=camera"),
            title(camera_overlay(mesh_preview(np_,gK,gc,h=h,w=width),t),"candidate same global view")],1))
        for eid in ids:
            target=np.array(Image.fromarray(masks[ids.index(eid),t]).resize((width,h),Image.Resampling.NEAREST),bool)
            if target.sum()<12:continue
            pred=np.zeros_like(target)
            for j,part in enumerate(np_):
                if part["entity_id"]==eid:pred|=lab==j
            hit=target&np.isfinite(rd)
            metrics.setdefault(eid,[]).append({"frame":t,"iou":float((pred&target).sum()/max((pred|target).sum(),1)),
               "depth_relative_mae":float(np.mean(np.abs(rd[hit]-td[hit])/td[hit])) if hit.any() else None})
    fps=float(1/np.median(np.diff(a["times_s"]))) if n>1 else 8.
    for name,frames in [("source_comparison",source),("global_comparison",globalv),("white_model",white)]:
        imageio.mimwrite(p/(name+".mp4"),frames,fps=fps,codec="libx264",macro_block_size=1,quality=7)
    picks=np.unique(np.linspace(0,n-1,min(n,5)).round().astype(int))
    Image.fromarray(np.concatenate([source[t] for t in picks],0)).save(p/"contact.jpg")
    Image.fromarray(globalv[n//2]).save(p/"global.jpg")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(12,3.6))
    for s,style,name in [(old,"--","baseline"),(new,"-","candidate")]:
        for e in s["entities"]:
            if e["motion"]=="static":continue
            pos=np.asarray(e["entity_to_world"])[:,:3,3]
            for ax,k in zip(axes,range(3)):ax.plot(a["times_s"],pos[:,k],style,label=e["id"]+" "+name)
    for ax,k in zip(axes,range(3)):ax.plot(a["times_s"],a["c2w"][:,k,3],":",color="#15946f",label="VGGT camera (fixed)")
    write_json(p/"camera_trajectory.json",{"geometry_manifest":new["geometry_manifest"],"times_s":a["times_s"].tolist(),"c2w":a["c2w"].tolist(),"K":a["K"].tolist(),"optimized":False})
    for ax,k in zip(axes,["world X","world Y","world Z"]):ax.set(xlabel="time (s)",ylabel=k);ax.grid(alpha=.3)
    axes[0].legend(fontsize=7);fig.tight_layout();fig.savefig(p/"trajectories.png",dpi=140);plt.close(fig)
    report={"source_scene":reference(old_path),"candidate_scene":reference(new_path),"camera_policy":"same fixed original VGGT camera",
      "camera_trajectory":reference(p/"camera_trajectory.json"),"global_view":"same full geometry view; no depth clipping; camera path green","per_entity":metrics,
      "mean_mask_iou":{k:float(np.mean([r["iou"] for r in rows])) for k,rows in metrics.items()},
      "mean_depth_relative_mae":{k:(float(np.mean([r["depth_relative_mae"] for r in rows if r["depth_relative_mae"] is not None])) if any(r["depth_relative_mae"] is not None for r in rows) else None) for k,rows in metrics.items()},
      "collision_before":audit(old),"collision_after":audit(new),"automatic_acceptance":False}
    write_json(p/"report.json",report)
    (p/"index.html").write_text("""<!doctype html><meta charset="utf-8"><title>Coarse reconstruction V4</title>
<style>body{background:#121923;color:#e7edf5;font:16px system-ui;max-width:1200px;margin:30px auto}video,img{max-width:100%}a{color:#8bc9ff}</style>
<h1>Coarse reconstruction V4 — candidate review</h1><p>Same input stage and cameras. Geometry is coarse and uncertain; no true-geometry claim from image residuals. Global geometry is not truncated. Orange/red depth residual is diagnostic, including imperfect masks and predicted depth.</p>
<h2>RGB / baseline / candidate / depth residual</h2><video src="source_comparison.mp4" controls loop></video>
<h2>Common global view</h2><video src="global_comparison.mp4" controls loop></video>
<h2>White geometry</h2><video src="white_model.mp4" controls loop></video>
<h2>Root trajectories</h2><img src="trajectories.png"><p><a href="report.json">Metrics and unresolved bound overlaps</a></p><img src="contact.jpg"><img src="global.jpg">""")
    return report
