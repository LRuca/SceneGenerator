"""Independent narrative priority and geometric necessity; no VLM network implicit calls."""
from .storage import read_json,write_json,new_run,reference,verify
NARRATIVE={"key_actor","key_prop","supporting","incidental","unknown"}
DUTIES={"support","occlusion","contact","collision_boundary","layout"}
REPRESENTATIONS={"primitive_assembly","thin_surface","deformable_cage","description_only","unresolved"}

def validate(policy,scene):
    if policy.get("schema")!="scene_baseline.modeling_policy.v4":raise ValueError("Policy schema")
    if not policy.get("actor"):raise ValueError("Missing decision actor")
    expected={e["id"]:e for e in scene["entities"]}
    rows=policy.get("entities",[])
    if len(rows)!=len(expected) or {e["id"] for e in rows}!=set(expected):raise ValueError("Policy must cover each current entity exactly once")
    for e in rows:
        if e.get("narrative_role") not in NARRATIVE:raise ValueError("Narrative role")
        if not set(e.get("geometric_duties",[]))<=DUTIES:raise ValueError("Geometric duty")
        if e.get("representation") not in REPRESENTATIONS:raise ValueError("Representation")
        if not 0<=e.get("confidence",-1)<=1:raise ValueError("Confidence")
        if not e.get("evidence"):raise ValueError("Evidence required")
        if e["representation"]=="description_only" and (e["geometric_duties"] or e["narrative_role"] in {"key_actor","key_prop"}):
            raise ValueError("Key actors/props and geometric duties require geometry")
        if not 0<=e.get("part_budget",-1)<=24:raise ValueError("Part budget")
        if e["representation"] not in {"description_only","unresolved"} and e["part_budget"]<1:raise ValueError("Geometry needs positive budget")
        if e.get("motion") not in {"static","rigid","deformable","unknown"}:raise ValueError("Motion")
        if e["representation"]=="deformable_cage" and e["motion"]!="deformable":raise ValueError("Cage motion mismatch")
        if not e.get("identity_status") or not e.get("boundary_status"):raise ValueError("Identity/boundary decisions required")
    for rel in policy.get("relations",[]):
        if rel.get("subject") not in expected or rel.get("object") not in expected:raise ValueError("Unknown relation endpoint")
        if rel.get("relation") not in {"supported_by","occludes","contacts","separate_from","part_of"}:raise ValueError("Relation type")
        if not rel.get("evidence"):raise ValueError("Relation evidence")
    return policy

def request(scene_path,out):
    scene=read_json(scene_path);p=new_run(out)
    doc={"schema":"scene_baseline.modeling_request.v4","actor":"current_chat_codex",
      "source_scene":reference(scene_path),"geometry_manifest":scene["geometry_manifest"],
      "script":read_json(verify(scene["geometry_manifest"]))["script"],
      "inventory":[{"id":e["id"],"label":e["label"],"motion_candidate":e["motion"],"existing_parts":e["parts"]} for e in scene["entities"]],
      "instructions":["Inspect the same-stage RGB sequence, script and depth; script is underdefined and not ground truth.",
       "For each identity separately decide narrative role, geometric duties, motion and representation budget.",
       "Background walls must remain geometric if needed for occlusion, support, contact or free-space boundaries.",
       "SAM is auxiliary; propose missing entities or split/merge revisions even when masks are empty or merged.",
       "Do not assign all depth inside a prompt box to an object. Do not emit guessed metric coordinates.",
       "Report identity/boundary uncertainty and provide indexed evidence; no external VLM API is implied.",
       "Revision proposals must be explicit; application creates a new scene and requires new geometry evidence/fit."],
      "response_schema":"scene_baseline.modeling_policy.v4","automatic_acceptance":False}
    write_json(p/"request.json",doc);return doc

def apply(scene_path,policy_path,out,total_steps=1200):
    scene=read_json(scene_path);policy=read_json(policy_path)
    if policy.get("source_scene")!=reference(scene_path):raise ValueError("Policy bound to different scene")
    validate(policy,scene);p=new_run(out)
    for e,decision in [(e,next(x for x in policy["entities"] if x["id"]==e["id"])) for e in scene["entities"]]:
        e["modeling_policy"]=decision
        # Representation changes need separate fitted geometry; never silently fabricate it.
        e["modeling_priority"]="high" if decision["narrative_role"] in {"key_actor","key_prop"} else "structural" if decision["geometric_duties"] else "low"
    scene["modeling_policy_source"]=reference(policy_path)
    plan=execution_plan(scene,total_steps)
    scene["modeling_execution_plan"]=plan
    write_json(p/"execution_plan.json",plan)
    write_json(p/"scene.json",scene)
    write_json(p/"report.json",{"source":reference(scene_path),"policy":reference(policy_path),
      "applied":"validated decisions and executable per-entity budgets; representation revisions are gated",
      "revision_queue":policy.get("revision_proposals",[]),"external_vlm_called":False})
    return scene

def execution_plan(scene,total_steps=1200):
    """Allocate a fixed compute budget without dropping structural background geometry."""
    if type(total_steps) is not int or total_steps<1:raise ValueError("Positive total_steps required")
    jobs=[]
    for e in scene["entities"]:
        p=e.get("modeling_policy")
        if p is None:raise ValueError("Unreviewed entity: "+e["id"])
        rep=p["representation"]
        requires=(rep=="deformable_cage")!=(e["motion"]=="deformable")
        requires |= p["motion"] not in {e["motion"],"unknown"}
        requires |= len(e["parts"])>p["part_budget"] and rep!="description_only"
        weight=4 if p["narrative_role"] in {"key_actor","key_prop"} else 2 if p["geometric_duties"] else 1
        jobs.append({"entity_id":e["id"],"weight":weight,"steps":0,"part_budget":p["part_budget"],
          "representation":rep,"geometric_duties":p["geometric_duties"],
          "status":"revision_required" if requires or rep=="unresolved" else "description_only" if rep=="description_only" else "ready",
          "retain_geometry":rep!="description_only","motion":e["motion"]})
    ready=[x for x in jobs if x["status"]=="ready"]
    if ready:
        denom=sum(x["weight"] for x in ready);remaining=total_steps
        for x in ready:x["steps"]=total_steps*x["weight"]//denom;remaining-=x["steps"]
        for x in sorted(ready,key=lambda x:-x["weight"])[:remaining]:x["steps"]+=1
    return {"schema":"scene_baseline.execution_plan.v4","total_steps":total_steps,"jobs":jobs,
      "requires_revision":[x["entity_id"] for x in jobs if x["status"]=="revision_required"],
      "policy":"fixed total budget; narrative priority and structural duties independent; no silent shape replacement"}

def entity_budget(scene,entity_id,requested_steps):
    plan=scene.get("modeling_execution_plan")
    if plan is None:return requested_steps
    job=next(x for x in plan["jobs"] if x["entity_id"]==entity_id)
    if job["status"]!="ready":raise ValueError("Agent geometry revision required or description-only entity: "+entity_id)
    return min(requested_steps,job["steps"])

def diagnose(scene_path,review_report_path,out):
    """Generate evidence-bound issues for the current-chat agent, never guess replacement coordinates."""
    scene=read_json(scene_path);report=read_json(review_report_path)
    if report["candidate_scene"]!=reference(scene_path):raise ValueError("Diagnostic scene mismatch")
    issues=[]
    for e in scene["entities"]:
        eid=e["id"];iou=report.get("mean_mask_iou",{}).get(eid)
        depth=report.get("mean_depth_relative_mae",{}).get(eid)
        sparse=e.get("sparse_evidence",{})
        missing=[i for i,v in enumerate(e.get("observed",[])) if not v]
        signals=[]
        if iou is not None and iou<.7:signals.append("low_visible_silhouette_agreement")
        if depth is not None and depth>.04:signals.append("depth_or_occlusion_mismatch")
        if missing:signals.append("missing_masks_do_not_remove_identity")
        if sparse and not all(sparse["support"]):signals.append("underconstrained_motion")
        if signals:issues.append({"entity_id":eid,"signals":signals,"missing_mask_frames":missing,
          "agent_actions":["review identity and boundary on indexed RGB","check amodal shape and occluders","revise representation only with evidence; rerun fit"],
          "automatic_coordinate_edit":False})
    p=new_run(out);doc={"schema":"scene_baseline.agent_diagnosis.v4","scene":reference(scene_path),
      "visual_review":reference(review_report_path),"issues":issues,"collision_audit":report.get("collision_after"),
      "external_vlm_called":False,"automatic_acceptance":False}
    write_json(p/"request.json",doc);return doc
