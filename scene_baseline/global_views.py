"""Global inspection of actual scene geometry, with predicted camera frusta."""
import argparse,json
import numpy as np
from PIL import Image,ImageDraw
import imageio.v2 as imageio
from .scene import load_scene
from .geometry import scene_parts,unit_mesh,transform_mesh,mesh_preview,render_parts
from .storage import new_run,write_json,reference

def look_at(eye,center,up):
    f=center-eye;f/=np.linalg.norm(f)
    r=np.cross(f,up);r/=np.linalg.norm(r)
    d=np.cross(f,r)
    c=np.eye(4);c[:3,:3]=np.stack([r,d,f],axis=1);c[:3,3]=eye
    return c

def export(scene_path,out,width=448):
    s,m,a=load_scene(scene_path);p=new_run(out);n=len(a["depth"])
    from .persistence import effective_cameras
    camera_poses=effective_cameras(s,a)
    up=np.asarray(s["ground_normal"]);up/=np.linalg.norm(up)
    right=a["c2w"][0,:3,0].copy();right-=up*(right@up);right/=np.linalg.norm(right)
    front=np.cross(right,up)
    rigid_ids={e["id"] for e in s["entities"] if e["motion"]=="rigid"}
    verts=[];focus_vertices=[]
    for t in range(n):
        for part in scene_parts(s,t):
            v=transform_mesh(unit_mesh("box" if part["kind"]=="plane" else part["kind"]),part["canonical_to_world"])[0]
            verts.append(v)
            if part["entity_id"] in rigid_ids:focus_vertices.append(v)
    points=np.concatenate(verts);center=(points.min(0)+points.max(0))/2
    radius=np.linalg.norm(points-center,axis=1).max()
    K=np.array([[width*.85,0,(width-1)/2],[0,width*.85,(width-1)/2],[0,0,1.]])
    def view(t,angle,focus=None):
        ctr,rad=(center,radius) if focus is None else focus
        eye=ctr+rad*2.15*(np.cos(angle)*right+np.sin(angle)*front+up*.5)
        c=look_at(eye,ctr,up);parts=scene_parts(s,t)
        im=mesh_preview(parts,K,c,width,width)
        _,label=render_parts(parts,K,c,width,width)
        boundary=((label!=np.roll(label,1,0))|(label!=np.roll(label,1,1)))&(label>=0)
        im[boundary]=[24,24,24]
        return im
    angle=np.pi/2+.55
    frames=[view(t,angle) for t in range(n)]
    imageio.mimwrite(p/"global_fixed.mp4",frames,fps=16,codec="libx264",macro_block_size=1,quality=8)
    orbit=[view(n//2,v) for v in np.linspace(np.pi/2,np.pi/2+2*np.pi,96,endpoint=False)]
    imageio.mimwrite(p/"orbit_midtime.mp4",orbit,fps=24,codec="libx264",macro_block_size=1,quality=8)
    sheet=Image.new("RGB",(width*3,width+34),"#151a23");draw=ImageDraw.Draw(sheet)
    for k,v in enumerate([np.pi/2+.55,-np.pi/2+.55,np.pi+.55]):
        sheet.paste(Image.fromarray(view(n//2,v)),(k*width,34));draw.text((k*width+12,10),f"Global view {k+1} | t={a['times_s'][n//2]:.2f}s",fill="white")
    sheet.save(p/"global_views.jpg")
    if focus_vertices:
        fp=np.concatenate(focus_vertices);fc=(fp.min(0)+fp.max(0))/2
        fr=np.linalg.norm(fp-fc,axis=1).max()
        focus=(fc,max(fr,1e-4))
        focused=[view(t,angle,focus) for t in range(n)]
        imageio.mimwrite(p/"global_focus.mp4",focused,fps=16,codec="libx264",macro_block_size=1,quality=8)
        contact=Image.new("RGB",(width*4,width+32),"#151a23");dr=ImageDraw.Draw(contact)
        for j,t in enumerate([0,n//4,2*n//3,n-1]):
            contact.paste(Image.fromarray(focused[t]),(j*width,32));dr.text((j*width+8,8),f"Focused global | frame {t}",fill="white")
        contact.save(p/"focus_views.jpg")
    # Equal world-unit scale; display coordinate system has Z aligned with fitted ground-up.
    import matplotlib;matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    basis=np.stack([right,front,up],axis=1)
    def P(x):return np.asarray(x)@basis
    edges=[(i,j) for i in range(8) for j in range(i) if bin(i^j).count("1")==1]
    camera=camera_poses[:,:3,3]
    dynamic=[e for e in s["entities"] if e["motion"]=="rigid"]
    dynamic_ids=[e["id"] for e in dynamic]
    palette=["#d642a1","#00a5b5","#dc7a00","#70a52a"]
    trajectories=[]
    for i,e in enumerate(dynamic):
        path=np.asarray(e["entity_to_world"])[:,:3,3].copy()
        path[~np.asarray(e["renderable"])]=np.nan
        trajectories.append((path,palette[i%len(palette)],e["id"]))
    whole=P(np.concatenate([points,camera]));lo,hi=whole.min(0),whole.max(0);mid=(lo+hi)/2;extent=(hi-lo).max()*.56
    def draw_world(ax,t,azimuth):
        for part in scene_parts(s,t):
            v,f=transform_mesh(unit_mesh("box" if part["kind"]=="plane" else part["kind"]),part["canonical_to_world"])[:2]
            v=P(v);col=palette[dynamic_ids.index(part["entity_id"])%len(palette)] if part["entity_id"] in dynamic_ids else "#e3e7ec"
            ax.add_collection3d(Poly3DCollection(v[f],facecolor=col,edgecolor="#777777",linewidth=.6,alpha=.28))
        for arr,col,label in [(camera,"#008ecf","VGGT camera")]+trajectories:
            q=P(arr);ax.plot(*q.T,color=col,label=label,lw=2)
            if np.isfinite(q[min(t,len(q)-1)]).all():ax.scatter(*q[min(t,len(q)-1)],color=col,s=18)
        # Sparse predicted camera frusta; no hidden geometry added.
        for i in [0,n//2,n-1]:
            pose=camera_poses[i];h,w=a["depth"].shape[1:];uv=np.array([[0,0,1],[w-1,0,1],[w-1,h-1,1],[0,h-1,1]])
            local=uv@np.linalg.inv(a["K"][i]).T*radius*.1
            corners=local@pose[:3,:3].T+pose[:3,3]
            origin=P(pose[:3,3]);cc=P(corners)
            for end in cc:ax.plot(*np.stack([origin,end]).T,color="#008ecf",lw=.7)
            ax.plot(*np.concatenate([cc,cc[:1]]).T,color="#008ecf",lw=.7)
        ax.set(xlim=(mid[0]-extent,mid[0]+extent),ylim=(mid[1]-extent,mid[1]+extent),zlim=(mid[2]-extent,mid[2]+extent),xlabel="Right",ylabel="Forward",zlabel="Ground up")
        ax.set_box_aspect((1,1,1));ax.view_init(elev=25,azim=azimuth);ax.legend(loc="upper left",fontsize=8)
    fig=plt.figure(figsize=(14,6))
    for k,az in enumerate([-65,115]):
        ax=fig.add_subplot(1,2,k+1,projection="3d");draw_world(ax,n//2,az)
    fig.suptitle("Actual fitted scene | transparent geometry + camera frusta | arbitrary scale")
    fig.tight_layout();fig.savefig(p/"global_wireframe.jpg",dpi=125);plt.close(fig)
    fig=plt.figure(figsize=(8,7));ax=fig.add_subplot(111,projection="3d")
    with imageio.get_writer(p/"global_wireframe.mp4",fps=16,codec="libx264",macro_block_size=1,quality=7) as wr:
        for t in range(n):
            ax.clear();draw_world(ax,t,-65);fig.suptitle(f"Global scene | frame {t} | camera blue / object paths colored")
            fig.canvas.draw();wr.append_data(np.asarray(fig.canvas.buffer_rgba())[...,:3])
    plt.close(fig)
    # OBJ at the middle instant, no guessed/unseen detailed surfaces.
    lines=["# Coarse fitted geometry; arbitrary VGGT scale; frozen middle frame"];offset=1
    for part in scene_parts(s,n//2):
        v,f=transform_mesh(unit_mesh("box" if part["kind"]=="plane" else part["kind"]),part["canonical_to_world"])[:2]
        lines.append("o "+part["entity_id"])
        lines.extend("v %.8f %.8f %.8f"%tuple(q) for q in v)
        lines.extend("f "+" ".join(str(int(i)+offset) for i in face) for face in f);offset+=len(v)
    (p/"scene_midtime.obj").write_text("\n".join(lines)+"\n")
    write_json(p/"manifest.json",{"scene":reference(scene_path),"ground_aligned_visualization":True,"solid_views":"same unmodified scene geometry; opaque occlusion retained","wireframe":"transparency for diagnosis; blue predicted cameras, separate colors per dynamic entity; paths only in renderable intervals","orbit":"frozen middle time; only visualization viewpoint changes","units":"arbitrary common VGGT scale","camera_optimized":False,"camera_estimate":"continuous" if "continuous_camera" in s else "raw"})
    return str(p)
if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--scene",required=True);ap.add_argument("--out",required=True)
    args=ap.parse_args();print(export(args.scene,args.out))
