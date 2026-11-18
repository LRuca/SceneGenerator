"""Sequential allowlisted stage dispatcher with immutable outputs and explicit references."""
from .storage import read_json,write_json,new_run,reference,utc,inside
from .contracts import load_bundle

def _resolve(value,outputs):
    if isinstance(value,dict):
        if set(value)=={"ref"}:
            task,key=value["ref"].split(".",1)
            if task not in outputs or key not in outputs[task]:raise ValueError("Unresolved stage output")
            return outputs[task][key]
        return {k:_resolve(v,outputs) for k,v in value.items()}
    if isinstance(value,list):return [_resolve(v,outputs) for v in value]
    return value

def _stage(kind,args,out):
    if kind=="geometry.existing":
        m,a=load_bundle(args["manifest"])
        return {"manifest":str(inside(args["manifest"]))}
    if kind=="geometry.preview":
        from .preview import bundle_preview
        m,a=load_bundle(args["manifest"])
        path=inside(m["geometry"]["path"]).parent/"rgb_depth.mp4"
        if not path.exists():bundle_preview(args["manifest"])
        return {"preview":str(path)}
    if kind=="agent.prepare":
        from .agent import prepare
        prepare(args["manifest"],out)
        return {"request":str(out/"request.json")}
    if kind=="tool.run":
        from .tools import invoke
        invoke(args["request"],out)
        return {"result":str(out/"result.json")}
    if kind=="masks.import":
        from .masks import import_masks
        import_masks(args["manifest"],args["hypothesis"],args["masks"],out)
        return {"manifest":str(out/"manifest.json")}
    if kind=="geometry.vggt":
        from .vggt import export
        export(args["rgb"],args["script"],out,args.get("fps",16),args.get("frames",49))
        return {"manifest":str(out/"manifest.json")}
    if kind=="synthetic.smoke":
        from .smoke import synthetic
        synthetic(out);return {"fit":str(out/"fit.json")}
    if kind=="synthetic.fixture":
        from .fixtures import create
        create(out,args.get("frames",4),args.get("size",48))
        return {k:str(out/v) for k,v in {"geometry_manifest":"geometry_manifest.json","hypothesis":"hypothesis.json","masks_manifest":"masks_manifest.json","initial_scene":"initial_scene.json"}.items()}
    if kind=="depth.derive":
        from .observations import derive_depth
        derive_depth(args["manifest"],args["tool_result"],out)
        return {"manifest":str(out/"manifest.json")}
    if kind=="scene.initialize":
        from .scene import initialize
        initialize(args["manifest"],args["hypothesis"],args["masks_manifest"],out)
        return {"scene":str(out/"scene.json")}
    if kind=="agent.motion_decision":
        from .motion_policy_v1 import apply
        apply(args["scene"],args["decisions"],out)
        return {"scene":str(out/"scene.json"),"report":str(out/"report.json")}
    if kind=="scene.fit_agent":
        from .motion_policy_v1 import fit_locked
        fit_locked(args["scene"],out,steps=args.get("steps",160),resolution=args.get("resolution",48),device=args.get("device","cpu"),lr=args.get("lr",.015),entity_ids=args.get("entities"))
        return {"scene":str(out/"scene.json"),"report":str(out/"report.json")}
    if kind=="scene.fit":
        from .fit_scene import fit
        fit(args["scene"],out,args.get("steps",160),args.get("resolution",48),args.get("device","cpu"),args.get("lr",.015),entity_ids=args.get("entities"))
        return {"scene":str(out/"scene.json"),"report":str(out/"report.json")}
    if kind=="scene.persist":
        from .persistence import apply
        apply(args["scene"],args["policy"],out)
        return {"scene":str(out/"scene.json"),"report":str(out/"report.json")}
    if kind=="proxy.encode":
        from .proxy_encoding_v8 import prepare
        prepare(args["scene"],args["decisions"],out)
        return {"scene":str(out/"scene.json"),"report":str(out/"report.json")}
    if kind=="blocking.package":
        from .proxy_encoding_v8 import package
        package(args["scene"],args["source_package"],out,args.get("displacement",0.),args.get("width",320))
        return {"manifest":str(out/"manifest.json"),"video":str(out/"white_model_reference.mp4"),"comparison":str(out/"render_comparison.mp4"),"prompt":str(out/"prompt.txt"),"validation":str(out/"validation.json")}
    if kind=="scene.render":
        from .delivery import render
        render(args["scene"],out,args.get("width",192))
        return {"metrics":str(out/"metrics.json"),"video":str(out/"comparison.mp4")}
    if kind=="scene.revise":
        from .revision import apply
        apply(args["scene"],args["request"],out)
        return {"scene":str(out/"scene.json")}
    if kind=="agent.accept":
        from .agent import accept
        accept(args["request"],args["response"],out)
        return {"hypothesis":str(out/"hypothesis.json")}
    if kind=="sam3.track":
        from .sam3_worker import track
        track(args["manifest"],args["hypothesis"],args["entity"],args["checkpoint"],out,args.get("max_frames"))
        return {"manifest":str(out/"manifest.json"),"masks":str(out/"masks.npz")}
    raise ValueError("Unknown or not yet implemented stage: "+kind)

def run(config,out):
    doc=read_json(config)
    if doc.get("schema")!="scene_baseline.pipeline.v1":raise ValueError("Pipeline schema")
    p=new_run(out);outputs={};seen=set()
    write_json(p/"configuration.json",doc)
    for task in doc["tasks"]:
        name=task["id"]
        if name in seen or not name or any(x not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for x in name):raise ValueError("Invalid stage id")
        seen.add(name)
        try:
            args=_resolve(task.get("arguments",{}),outputs)
            output=_stage(task["kind"],args,p/name);outputs[name]=output
            write_json(p/(name+".status.json"),{"kind":task["kind"],"status":"completed","outputs":{k:reference(v) for k,v in output.items()},"time":utc()})
        except Exception as error:
            write_json(p/(name+".status.json"),{"kind":task["kind"],"status":"failed","error_type":type(error).__name__,"message":str(error),"time":utc()})
            raise
    write_json(p/"status.json",{"status":"configured_stages_completed","outputs":outputs,"config":reference(config),
        "note":"Completion covers only the configured implemented stages, not full real-scene baseline."})
    return outputs
