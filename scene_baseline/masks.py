"""Segmentation backend adapters; imported masks remain bound to geometry and hypotheses."""
import numpy as np
from .contracts import load_bundle,validate_masks,validate_hypothesis
from .storage import new_run,write_json,reference,read_json,inside

def import_masks(manifest,hypothesis,masks_path,out):
    m,a=load_bundle(manifest);hyp=validate_hypothesis(read_json(hypothesis))
    if hyp["geometry_manifest_sha256"]!=reference(manifest)["sha256"]:raise ValueError("Hypothesis geometry mismatch")
    with np.load(inside(masks_path),allow_pickle=False) as z:
        masks=z["masks"];ids=z["ids"].tolist()
    validate_masks(masks,ids,a)
    if not set(ids).issubset({e["id"] for e in hyp["entities"]}):raise ValueError("Mask identities absent from hypothesis")
    p=new_run(out);np.savez_compressed(p/"masks.npz",masks=masks,ids=np.asarray(ids))
    doc={"schema":"scene_baseline.masks.v1","geometry_manifest":reference(manifest),"hypothesis":reference(hypothesis),
        "source":reference(masks_path),"data":reference(p/"masks.npz"),"ids":ids,
        "visibility":masks.any(axis=(2,3)).tolist(),"status":"imported_not_independently_verified"}
    write_json(p/"manifest.json",doc);return doc
