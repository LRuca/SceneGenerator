"""Provider-independent file agent interface. No credentials or network on import."""
from .contracts import load_bundle,validate_hypothesis
from .storage import new_run,reference,write_json,read_json,verify
from .tools import catalog

def prepare(manifest,out):
    m,a=load_bundle(manifest);p=new_run(out)
    import numpy as np
    from PIL import Image,ImageDraw
    from .preview import colors
    count=min(6,len(a["depth"]));indices=np.linspace(0,len(a["depth"])-1,count).round().astype(int)
    sheet=Image.new("RGB",(count*180,400),"white");draw=ImageDraw.Draw(sheet)
    ld=np.log(a["depth"]);colored=colors(a["depth"],float(ld.min()),float(ld.max()))
    for j,t in enumerate(indices):
        sheet.paste(Image.fromarray((a["rgb"][t]*255).astype("uint8")).resize((180,180)),(j*180,25))
        sheet.paste(Image.fromarray(colored[t]).resize((180,180)),(j*180,220))
        draw.text((j*180+3,5),"RGB frame "+str(a["frame_indices"][t]),fill="black")
        draw.text((j*180+3,205),"DEPTH full-clip log",fill="black")
    sheet.save(p/"contact_sheet.jpg")
    doc={"schema":"scene_baseline.agent_request.v1","geometry_manifest":reference(manifest),"contact_sheet":reference(p/"contact_sheet.jpg"),
        "inputs":{"rgb_source":m["source_rgb"],"script":m["script"],"geometry":m["geometry"],
            "rgb_depth_preview":reference(verify(m["geometry"]).parent/"rgb_depth.mp4")},
        "instructions":["Observe RGB, script and depth together. Script is a hypothesis; resolve conflicts against visible evidence.",
            "Return instances, static/rigid/deformable/unknown motion, primitive composition, support and occlusion.",
            "Mark uncertainty, visible time ranges and segmentation prompts/normalized boxes. Do not guess metric coordinates.",
            "Request tools using the allowlisted catalog; no shell requests. Never use later diffusion stages.",
            "Depth is predicted camera-Z in arbitrary units. Distinguish geometry-supported from semantic priors."],
        "tools":catalog(),"expected_response_schema":"scene_baseline.hypothesis.v1",
        "response_example":{"schema":"scene_baseline.hypothesis.v1","geometry_manifest_sha256":reference(manifest)["sha256"],
            "actor":"name/model-and-version","entities":[{"id":"vehicle_0","label":"red pickup truck","motion":"rigid",
                "segmentation_prompt":"red pickup truck","confidence":0.8,"parts":[{"id":"body","kind":"box","role":"body"}],
                "evidence":"visible RGB and depth region","uncertainties":["hidden thickness"]}],
            "relations":[],"tool_requests":[]},
        "provider_status":"file_exchange_ready; remote_vlm_not_configured"}
    write_json(p/"request.json",doc);return doc

def accept(request,response,out):
    request_doc=read_json(request);verify(request_doc["geometry_manifest"])
    doc=validate_hypothesis(read_json(response))
    if doc["geometry_manifest_sha256"]!=request_doc["geometry_manifest"]["sha256"]:raise ValueError("Wrong geometry in agent response")
    p=new_run(out);write_json(p/"hypothesis.json",doc)
    write_json(p/"provenance.json",{"request":reference(request),"response":reference(response),"status":"schema_validated_not_semantically_verified"})
    return doc
