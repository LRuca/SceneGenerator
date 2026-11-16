"""Command-line entry points; all heavyweight backends are imported on demand."""
import argparse,json

def parser():
    root=argparse.ArgumentParser(description="Early RGB/depth -> coarse scene baseline")
    sub=root.add_subparsers(dest="command",required=True)
    def add(name,*required):
        p=sub.add_parser(name)
        for field in required:p.add_argument("--"+field.replace("_","-"),required=True)
        return p
    p=add("vggt-export","rgb","script","out");p.add_argument("--fps",type=float,default=16);p.add_argument("--frames",type=int,default=49)
    add("persist","scene","policy","out");add("agent-prepare","manifest","out");add("agent-accept","request","response","out")
    p=add("sam3-track","manifest","hypothesis","entity","checkpoint","out");p.add_argument("--max-frames",type=int)
    add("pipeline-run","config","out")
    add("tool-list");add("tool-run","request","out");add("preview","manifest")
    add("import-masks","manifest","hypothesis","masks","out")
    add("depth-derive","manifest","tool_result","out")
    add("synthetic-smoke","out")
    p=add("fixture","out");p.add_argument("--frames",type=int,default=4);p.add_argument("--size",type=int,default=48)
    add("scene-init","manifest","hypothesis","masks_manifest","out")
    p=add("scene-fit","scene","out");p.add_argument("--steps",type=int,default=160);p.add_argument("--resolution",type=int,default=48);p.add_argument("--device",default="cpu");p.add_argument("--lr",type=float,default=.015);p.add_argument("--entities",nargs="+",default=None)
    p=add("scene-render","scene","out");p.add_argument("--width",type=int,default=192);p.add_argument("--evaluation-masks-manifest",default=None)
    add("scene-revise","scene","request","out")
    return root

def main():
    a=parser().parse_args();command=a.command
    if command=="persist":
        from .persistence import apply
        apply(a.scene,a.policy,a.out);result={"scene":a.out+"/scene.json"}
    elif command=="vggt-export":
        from .vggt import export
        result=export(a.rgb,a.script,a.out,a.fps,a.frames)
    elif command=="agent-prepare":
        from .agent import prepare
        result=prepare(a.manifest,a.out)
    elif command=="agent-accept":
        from .agent import accept
        result=accept(a.request,a.response,a.out)
    elif command=="sam3-track":
        from .sam3_worker import track
        result=track(a.manifest,a.hypothesis,a.entity,a.checkpoint,a.out,a.max_frames)
    elif command=="pipeline-run":
        from .pipeline import run
        result=run(a.config,a.out)
    elif command=="tool-list":
        from .tools import catalog
        result=catalog()
    elif command=="tool-run":
        from .tools import invoke
        result=invoke(a.request,a.out)
    elif command=="preview":
        from .preview import bundle_preview
        result=bundle_preview(a.manifest)
    elif command=="import-masks":
        from .masks import import_masks
        result=import_masks(a.manifest,a.hypothesis,a.masks,a.out)
    elif command=="depth-derive":
        from .observations import derive_depth
        result=derive_depth(a.manifest,a.tool_result,a.out)
    elif command=="synthetic-smoke":
        from .smoke import synthetic
        result=synthetic(a.out)
    elif command=="fixture":
        from .fixtures import create
        result=str(create(a.out,a.frames,a.size))
    elif command=="scene-init":
        from .scene import initialize
        result=initialize(a.manifest,a.hypothesis,a.masks_manifest,a.out)
    elif command=="scene-fit":
        from .fit_scene import fit
        result=fit(a.scene,a.out,a.steps,a.resolution,a.device,a.lr,entity_ids=a.entities)
    elif command=="scene-render":
        from .delivery import render
        result=render(a.scene,a.out,a.width,evaluation_masks_manifest=a.evaluation_masks_manifest)
    else:
        from .revision import apply
        result=apply(a.scene,a.request,a.out)
    print(json.dumps(result,indent=2,ensure_ascii=False))
if __name__=="__main__":main()
