"""Render white-model/depth/masks/residuals, report uncovered input and export trajectories."""
import numpy as np,cv2,imageio.v2 as imageio
from PIL import Image,ImageDraw
from matplotlib import colormaps
from .scene import load_scene
from .geometry import part_mesh,scene_parts,render_parts,mesh_preview,agreement,unit_mesh,transform_mesh
from .storage import new_run,write_json,reference,verify
from .preview import colors

def render(scene_path,out,width=192,evaluation_masks_manifest=None):
    scene,m,a=load_scene(scene_path);n,H,W=a["depth"].shape
    from .persistence import effective_cameras
    camera_poses=effective_cameras(scene,a)
    if width<32 or width>1024:raise ValueError("Render width outside supported range")
    height=round(H/W*width);sx,sy=width/W,height/H
    A=np.array([[sx,0,(sx-1)/2],[0,sy,(sy-1)/2],[0,0,1.]])
    K=A@a["K"];p=new_run(out)
    from .storage import read_json
    mask_ref=scene["masks_manifest"] if evaluation_masks_manifest is None else reference(evaluation_masks_manifest)
    maskdoc=read_json(verify(mask_ref))
    if maskdoc["geometry_manifest"]!=scene["geometry_manifest"]:raise ValueError("Evaluation masks geometry mismatch")
    with np.load(verify(maskdoc["data"]),allow_pickle=False) as z:masks=z["masks"];ids=z["ids"].tolist()
    labels=[];depths=[];white=[];target=[];rgb=[];metrics=[]
    sceneids=[e["id"] for e in scene["entities"]]
    for t in range(n):
        parts=scene_parts(scene,t)
        d,l=render_parts(parts,K[t],camera_poses[t],height,width)
        el=np.full_like(l,-1)
        for k,part in enumerate(parts):el[l==k]=sceneids.index(part["entity_id"])
        z=cv2.resize(a["depth"][t],(width,height),interpolation=cv2.INTER_NEAREST_EXACT)
        target.append(z);depths.append(d);labels.append(el)
        frame=mesh_preview(parts,K[t],camera_poses[t],width,height)
        boundary=((el!=np.roll(el,1,axis=0))|(el!=np.roll(el,1,axis=1)))&(el>=0)
        frame[boundary]=[18,18,18]
        white.append(frame)
        rgb.append(cv2.resize((a["rgb"][t]*255).astype("uint8"),(width,height)))
        row={"frame_index":int(a["frame_indices"][t]),"time_s":float(a["times_s"][t]),"depth":agreement(d,z),"entities":{}}
        for e in scene["entities"]:
            truth=cv2.resize(masks[ids.index(e["id"]),t].astype("uint8"),(width,height),interpolation=cv2.INTER_NEAREST_EXACT)>0
            pred=el==sceneids.index(e["id"]);union=np.logical_or(truth,pred).sum()
            row["entities"][e["id"]]={"visible_mask_iou":float(np.logical_and(truth,pred).sum()/union) if union else None,
                "target_pixels":int(truth.sum()),"predicted_pixels":int(pred.sum()),"depth_on_target":agreement(d,z,truth)}
        metrics.append(row)
    depth=np.stack(depths);target=np.stack(target);labels=np.stack(labels)
    np.savez_compressed(p/"rendered_geometry.npz",depth=depth,instance_index=labels,target_depth=target,frame_indices=a["frame_indices"])
    finite=depth[np.isfinite(depth)&(depth>0)]
    log=np.log(np.r_[target.ravel(),finite]);lo=float(log.min());hi=float(log.max())
    td=colors(target,lo,hi);rd=colors(np.where(np.isfinite(depth),depth,np.exp(lo)),lo,hi)
    rd[~np.isfinite(depth)]=[45,45,45]
    err=np.where(np.isfinite(depth),np.abs(depth-target)/target,1.)
    ec=(colormaps["inferno"](np.clip(err/.5,0,1))[...,:3]*255).astype("uint8")
    ec[~np.isfinite(depth)]=[255,0,255]
    videos=[];errors=[]
    for t in range(n):
        canvas=Image.new("RGB",(width*4,height+34),"#141820");draw=ImageDraw.Draw(canvas)
        for j,(image,label) in enumerate(zip([rgb[t],td[t],white[t],rd[t]],["RGB","Input depth","White model","Rendered depth"])):
            canvas.paste(Image.fromarray(image),(j*width,34));draw.text((j*width+5,8),label,fill="white")
        videos.append(np.asarray(canvas))
        residual=Image.new("RGB",(width*2,height+34),"#141820");dr=ImageDraw.Draw(residual)
        residual.paste(Image.fromarray(white[t]),(0,34));residual.paste(Image.fromarray(ec[t]),(width,34))
        dr.text((5,8),"White / relative error: 0-.5, magenta=missing",fill="white")
        errors.append(np.asarray(residual))
    fps=float(1/np.median(np.diff(a["times_s"]))) if n>1 else m["fps"]
    imageio.mimwrite(p/"comparison.mp4",videos,fps=fps,codec="libx264",macro_block_size=1,quality=8)
    imageio.mimwrite(p/"white.mp4",white,fps=fps,codec="libx264",macro_block_size=1,quality=8)
    imageio.mimwrite(p/"residual.mp4",errors,fps=fps,codec="libx264",macro_block_size=1,quality=8)
    Image.fromarray(videos[n//2]).save(p/"comparison.jpg")
    # One fixed overview camera for the whole scene.
    from .geometry import REF
    import sys
    sys.path.insert(0,str(REF/"scripts"))
    from render_cage_white import overview_pose
    vertices=[]
    for t in sorted({0,n//2,n-1}):
        for part in scene_parts(scene,t):
            vertices.append(part_mesh(part)[0])
    overview=[]
    if vertices:
        v=np.concatenate(vertices);camera=overview_pose((v.min(0),v.max(0)))
        Ko=np.array([[width,0,width/2],[0,width,height/2],[0,0,1.]])
        for t in range(n):
            parts=scene_parts(scene,t);frame=mesh_preview(parts,Ko,camera,width,height)
            _,label=render_parts(parts,Ko,camera,height,width)
            boundary=((label!=np.roll(label,1,axis=0))|(label!=np.roll(label,1,axis=1)))&(label>=0)
            frame[boundary]=[18,18,18];overview.append(frame)
        imageio.mimwrite(p/"overview.mp4",overview,fps=fps,codec="libx264",macro_block_size=1,quality=8)
    import matplotlib;matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig=plt.figure(figsize=(7,6));ax=fig.add_subplot(111,projection="3d")
    c=camera_poses[:,:3,3];ax.plot(c[:,0],c[:,1],c[:,2],label="predicted camera")
    for e in scene["entities"]:
        if e["motion"] in ("rigid","deformable"):
            c=np.asarray(e["entity_to_world"])[:,:3,3];ax.plot(c[:,0],c[:,1],c[:,2],label=e["id"])
    ax.set(xlabel="World X",ylabel="World Y",zlabel="World Z",title="Camera and object paths | arbitrary scale")
    ax.legend();fig.tight_layout();fig.savefig(p/"trajectories.jpg",dpi=130);plt.close(fig)
    write_json(p/"trajectories.json",{"times_s":a["times_s"].tolist(),"camera_to_world":camera_poses.tolist(),"observation_camera_to_world":a["c2w"].tolist(),"K":a["K"].tolist(),
        "entities":{e["id"]:e["entity_to_world"] for e in scene["entities"]},"camera_status":"continuous_estimate_separate_from_original_observations" if "continuous_camera" in scene else m.get("camera_policy","fixed_input_initialization")})
    summary={"mean_coverage":float(np.mean([r["depth"]["coverage"] for r in metrics])),
        "mean_capped_relative_error_with_misses":float(np.mean([r["depth"]["capped_relative_error_with_misses"] for r in metrics]))}
    write_json(p/"metrics.json",{"summary":summary,"frames":metrics,"scene":reference(scene_path),
        "camera_estimate":"continuous_output" if "continuous_camera" in scene else "original_VGGT","evaluation_masks_manifest":mask_ref,"interpretation":"Agreement with predicted input depth/masks. Not independent camera or geometry accuracy; all times fitted."})
    (p/"index.html").write_text('<meta charset="utf-8"><style>body{font:18px system-ui;background:#131925;color:white;max-width:1200px;margin:30px auto}video,img{max-width:100%}a{color:#8cf}</style><h1>粗白模候选与误差</h1><p>相机采用场景记录的估计（原始或连续拟合）；深度和轮廓误差衡量对输入预测的解释程度，不是物理真值精度。深度使用整段统一完整对数范围，灰色是未渲染区域；残差视频品红表示缺失预测。</p><video controls loop muted src="comparison.mp4"></video><p><a href="white.mp4">纯白模</a> · <a href="metrics.json">数值诊断</a> · <a href="trajectories.json">轨迹</a></p><h2>残差</h2><video controls loop muted src="residual.mp4"></video><h2>固定总览</h2><video controls loop muted src="overview.mp4"></video><img src="trajectories.jpg">')
    return summary
