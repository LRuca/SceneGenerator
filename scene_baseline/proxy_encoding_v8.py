"""Identity/role encoding and camera intervention packages; no geometry refitting.

Explicit agent decisions are required. Encoding never guesses importance from motion,
never replaces geometry, and never modifies observation camera arrays.
"""
import copy
import shutil
from pathlib import Path
import numpy as np
from .storage import read_json, write_json, new_run, reference, verify
from .scene import load_scene

PALETTE = [('yellow',[250,190,35]),('blue',[45,110,245]),
           ('red',[235,65,65]),('green',[45,185,120]),('purple',[170,90,235]),
           ('orange',[245,125,35]),('cyan',[35,185,220]),('pink',[230,95,165])]
ROLES={'key_actor','key_prop','supporting','incidental','structural'}

def encode(scene, decisions):
    result=copy.deepcopy(scene)
    rows=decisions['entities']; ids=[e['id'] for e in scene['entities']]
    if len(rows)!=len(ids) or {r['id'] for r in rows}!=set(ids):
        raise ValueError('Encoding decisions must cover every entity once')
    if not decisions.get('actor'): raise ValueError('Explicit decision actor required')
    bindings=[]; palette_index=0; used=set()
    for entity in result['entities']:
        row=next(r for r in rows if r['id']==entity['id'])
        role=row['role']
        if role not in ROLES or not row.get('evidence'):raise ValueError('Role/evidence required')
        duties=row.get('geometric_duties',[])
        if not set(duties)<={'support','occlusion','contact','collision_boundary','layout'}:
            raise ValueError('Unknown geometric duty')
        prominent=role in {'key_actor','key_prop','supporting'}
        if prominent:
            if palette_index>=len(PALETTE):raise ValueError('Too many identities for distinct palette; supply another encoding policy')
            color_name,color=PALETTE[palette_index];palette_index+=1
            if tuple(color) in used:raise ValueError('Duplicate actor color')
            used.add(tuple(color))
        else: color_name,color='neutral gray',[185,185,185]
        image=row.get('appearance_image')
        if prominent and (type(image) is not int or image<1):
            raise ValueError('Highlighted identity needs an appearance image binding')
        entity['proxy_encoding']={'role':role,'color_name':color_name,'rgb':color,
          'geometric_duties':duties,'appearance_image':image,'evidence':row['evidence']}
        bindings.append({'id':entity['id'],'prompt_id':row.get('prompt_id',entity['id']),'label':row.get('appearance_label',entity['label']),'motion':entity['motion'],**entity['proxy_encoding']})
    result['proxy_encoding']={'schema':'scene_baseline.proxy_encoding.v8',
      'actor':decisions['actor'],'identity_bindings':bindings,
      'policy':'key identities saturated and distinct; structural background neutral; preserve all fitted shapes and physical duties'}
    return result

def prepare(scene_path,decisions_path,out):
    scene,_,_=load_scene(scene_path);decisions=read_json(decisions_path)
    encoded=encode(scene,decisions);p=new_run(out)
    encoded['proxy_encoding']['decisions']=reference(decisions_path)
    write_json(p/'scene.json',encoded)
    write_json(p/'report.json',{'source_scene':reference(scene_path),'decisions':reference(decisions_path),
      'geometry_unchanged':True,'motion_unchanged':True,'role_is_agent_decision_not_motion_heuristic':True})
    return p

def camera_path(c2w,up,displacement):
    """Smooth added truck movement along first-camera ground-plane right; no pan/zoom."""
    base=np.asarray(c2w,float);up=np.asarray(up,float);up=up/np.linalg.norm(up)
    right=base[0,:3,0]-up*(base[0,:3,0]@up);right/=np.linalg.norm(right)
    u=np.linspace(0,1,len(base));ease=u*u*(3-2*u)
    out=base.copy();out[:,:3,3]+=ease[:,None]*float(displacement)*right
    return out,right

def binding_prompt(encoding):
    lines=['PROXY IDENTITY CODE: saturated colors mark narratively important identities, not final materials. Neutral gray marks structural context; preserve its occlusion, contact and support.']
    for b in encoding['identity_bindings']:
        if b['color_name']!='neutral gray':
            lines.append(f"The {b['color_name'].upper()} proxy is exactly {b['label']} (instance {b.get('prompt_id',b['id'])}); bind it ONLY to [Image {b['appearance_image']}]. Keep this identity through occlusion and camera motion. Its final material comes from that appearance image, not from the proxy color.")
    for b in encoding['identity_bindings']:
        if b.get('motion')=='static' and b['color_name']!='neutral gray':
            lines.append(f"MOTION STATE: {b['color_name'].upper()} / {b.get('prompt_id',b['id'])} has one shared world pose for the whole shot. Screen movement is camera/parallax only; do not make it drive or turn. This explicit state supersedes any uncertain historical appearance label.")
    return '\n'.join(lines)

def package(scene_path,source_package,out,displacement=0.0,width=320):
    from PIL import Image,ImageDraw
    import imageio.v2 as imageio
    from .geometry import scene_parts,render_parts
    from .lighting_v7 import render
    scene,_,a=load_scene(scene_path);encoding=scene.get('proxy_encoding')
    if not encoding:raise ValueError('Run proxy.encode first')
    source=Path(source_package);old=read_json(source/'manifest.json')
    if not all(old.get(k) for k in ['no_source_rgb_video','no_source_full_frame_image','no_first_frame_image']):
        raise ValueError('Appearance source does not certify isolation')
    p=new_run(out);(p/'references').mkdir()
    refs=[]
    for row in old['references']:
        src=verify(row['file']);dst=p/'references'/src.name;shutil.copyfile(src,dst)
        refs.append({**row,'file':reference(dst)})
    for b in encoding['identity_bindings']:
        i=b['appearance_image']
        if i is not None and (i>len(refs) or refs[i-1]['entity']!=b['id']):
            raise ValueError('Appearance image index/identity mismatch')
    base_prompt=(source/'prompt.txt').read_text()
    prompt=base_prompt+'\n'+binding_prompt(encoding)+'''\nCAMERA INTERVENTION: follow the physical lateral camera translation visible in Video 1 exactly, including the resulting parallax and reveal/occlusion. This is camera translation, not vehicle movement, a zoom or an invented vehicle departure. Each world-stationary instance remains stationary in world space although its screen position can change. Do not stabilize away the camera move or add a cut.\n'''
    (p/'prompt.txt').write_text(prompt)
    base=np.asarray(a['c2w']);cams,right=camera_path(base,scene['ground_normal'],displacement)
    np.savez_compressed(p/'camera_trajectory.npz',source_c2w=base,render_c2w=cams,K=a['K'],source_times_s=a['times_s'],output_times_s=np.linspace(0,4,len(base)))
    write_json(p/'scene.json',scene)
    colors={b['id']:b['rgb'] for b in encoding['identity_bindings']}
    n,H,W=a['depth'].shape;h=max(2,round(H*width/W)//2*2);K=a['K'].copy();K[:,0,:]*=width/W;K[:,1,:]*=h/H
    sample_ids=np.linspace(0,n-1,96).round().astype(int)
    # Use the same frames/timing in all variants: only palette/camera differ.
    unique={};diagnostics=[];contact=[]
    oldcolors={'forklift':[250,190,35]}
    for t in sorted(set(sample_ids)):
        parts=scene_parts(scene,int(t))
        frames=[render(parts,K[t],base[t],h,width,scene['ground_normal'],oldcolors,base[0,:3,0]),
                render(parts,K[t],base[t],h,width,scene['ground_normal'],colors,base[0,:3,0]),
                render(parts,K[t],cams[t],h,width,scene['ground_normal'],colors,base[0,:3,0])]
        unique[t]=frames
        d,labels=render_parts(parts,K[t],cams[t],h,width)
        diagnostics.append({'frame':int(t),'surface_coverage':float(np.isfinite(d).mean()),
          'visible_pixels':{e['id']:int(sum(np.count_nonzero(labels==j) for j,x in enumerate(parts) if x['entity_id']==e['id'])) for e in scene['entities']}})
        print('render',t,'/',n,flush=True)
    with imageio.get_writer(p/'white_model_reference.mp4',fps=24,codec='libx264',macro_block_size=1) as video, imageio.get_writer(p/'render_comparison.mp4',fps=24,codec='libx264',macro_block_size=1) as comparison:
        for t in sample_ids:
            frames=unique[t];video.append_data(np.asarray(Image.fromarray(frames[2]).resize((720,720))))
            board=Image.new('RGB',(960,350),(30,30,30));draw=ImageDraw.Draw(board)
            for j,(im,label) in enumerate(zip(frames,['V7 / original camera','Identity colors / original camera','Identity colors / lateral camera'])):
                board.paste(Image.fromarray(im).resize((320,320)),(j*320,30));draw.text((j*320+6,8),label,fill='white')
            comparison.append_data(np.asarray(board))
    board=Image.new('RGB',(960,320*3+30),(30,30,30));draw=ImageDraw.Draw(board)
    for j,label in enumerate(['V7 / original camera','Identity colors / original camera','Identity colors / lateral camera']):draw.text((j*320+6,8),label,fill='white')
    for i,t in enumerate([sample_ids[0],sample_ids[len(sample_ids)//2],sample_ids[-1]]):
        for j,im in enumerate(unique[t]):board.paste(Image.fromarray(im).resize((320,320)),(j*320,30+i*320))
    board.save(p/'render_contact.jpg')
    validation={'geometry_and_entity_poses_unchanged':True,'camera_rotations_unchanged':bool(np.array_equal(base[:,:3,:3],cams[:,:3,:3])),
      'added_translation_ground_parallel':bool(abs(right@np.asarray(scene['ground_normal']))<1e-8),
      'added_translation_world_units':float(displacement),'world_scale':'VGGT arbitrary shared scale',
      'render_pixels':diagnostics,'source_rgb_reprojection_not_applicable_to_modified_camera':True,
      'limitation':'Novel camera may expose unobserved surfaces. Coverage is not evidence of complete geometry or correct hidden shape.'}
    write_json(p/'validation.json',validation)
    write_json(p/'manifest.json',{'schema':'scene_baseline.blocking_package.v8','scene':reference(p/'scene.json'),
      'source_scene':reference(scene_path),'source_appearance_package':reference(source/'manifest.json'),
      'references':refs,'proxy_encoding':encoding,'camera_trajectory':reference(p/'camera_trajectory.npz'),
      'reference_video':reference(p/'white_model_reference.mp4'),'prompt':reference(p/'prompt.txt'),
      'no_source_rgb_video':True,'no_source_full_frame_image':True,'no_first_frame_image':True,
      'duration_s':4,'fps':24,'width':720,'height':720,'geometry_refit':False,
      'upstream_reused':'same-stage VGGT, SAM and fitted shared geometry; intervention only downstream camera/palette/prompt',
      'validation':reference(p/'validation.json')})
    return p
