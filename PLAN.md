# Early RGB/depth to coarse white-model baseline
Date: 2026-11-16. Reference: LRuca/SceneBlocker 4a0fa059e24f43a2e0988daed9605b91214be3c7.
Root: /home/tione/notebook/home/keithanwang/vlm-vla/gafu/sceneblocker/baseline

## Objective and flow
One selected diffusion stage RGB -> ONE joint VGGT pass -> raw depth/confidence/K/c2w + depth preview.
Agent sees RGB + script + that depth preview -> instance hypotheses/static-dynamic labels/primitive composition/support/occlusion.
Segmentation/tracking -> masks/identities/visibility.
Fix initial VGGT cameras -> fit shared static geometry, rigid shape, and per-frame object pose.
Render white-model source view and overview, camera/object paths, raw-vs-rendered depth residual and silhouette coverage.
Agent may propose structured revisions or processing tools based on diagnostics. Preserve every branch.

## Scope and evaluation
Start wan_joint_single step10; apply same procedure to step5.
No later-stage RGB in agent parsing, masks or fitting. Later stages are comparison only.
Camera is predicted initialization; fitting geometry does not validate camera accuracy.
Depth residual measures agreement with VGGT, not physical ground-truth accuracy.
Keep one arbitrary global scale; no human170cm or real-car dimensions for a toy car.
Fit and held-out time splits need separate reporting; upstream VGGT still sees full clip.
Do not fit colorized depth MP4; use original float32 depth.
Do not combine existing 49-frame depth with independently inferred 8-frame cameras.

## Reuse
primitive_composition.unit_mesh/fit_part: primitive mesh definitions and point-derived initialization.
proxy_observations.ray_interval: analytic intersections -> full nearest-surface renderer and metric diagnostics.
solver.core/terms: robust energy design, freezing, floor/support constraints.
render_cage_white.rasterize: CPU neutral mesh preview; use explicit camera-Z convention.
shared_scene_state: versioning/provenance philosophy; original validator requires >=2 actual cameras and is not reused unchanged.

## New or adapted
Single-camera clip/geometry/scene contracts, original_to_model transforms, artifact hashes and timestamps.
Agent request/response contract independent of model provider; file exchange initially, configurable remote provider later.
Tool registry with explicit modalities, schemas, risk/geometry effects, immutable outputs, requested observation roles.
Masks adapter for imported/SAM3 outputs; heuristic masks strictly labeled diagnostic.
Direct depth/silhouette rigid fitting; primitive topology is agent-controlled and bounded.
Full-range visualizations; per-stage fixed log range, no per-frame display rescaling.
No hardcoded original /workspace paths.

## Extension contracts
Tools declare: id/version/input modalities/parameter schema/output modality/coordinate effect/geometry effect.
A request records input artifact hash, parameters, actor and reason. Output records parent, software and timestamps.
RGB enhancement is a new observation branch; geometry remains bound to the RGB used by VGGT.
If new RGB should drive geometry, rerun VGGT as a new bundle. Depth filtering is explicit, retaining raw depth and cameras.
Unknown/unavailable tools fail explicitly. No arbitrary shell execution through agent requests.
Temporal filtering without motion compensation can ghost; do not enable by default.

## Delivery
RGB/target-depth/white-model/rendered-depth video; instance masks; residual/coverage; static overview with paths.
Scene JSON, camera trajectory, tool records, fit settings/metrics, limitations and per-stage status.

## Agent-mimic recovery v2
When automatic segmentation/tracking is missing or unreliable, the current-chat agent reviews and supplements evidence, requests fitting/rendering, and iterates. See experiments/agent_mimic_v2/AGENT_LOOP.md. This is a development protocol; it does not modify frozen persistence_v1 or claim automatic integration.
