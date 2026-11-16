# Contracts and extension points

## GeometryBundle v1
manifest.json binds immutable source RGB/script and same-pass geometry.npz by SHA256.
Arrays: rgb[T,H,W,3] [0,1], depth/confidence[T,H,W], K[T,3,3], c2w[T,4,4], times_s[T], original_to_model[T,3,3], frame_indices[T].
OpenCV camera +Z forward/+Y down. Depth is camera-Z, not Euclidean ray length. Arbitrary shared scale.
Original-to-model pixel transform records align_corners=False resize center offset. Mask arrays use model pixels.
External JSON modifications do not alter raw arrays. Bundles are new directories; mismatched or changed source files are rejected.

## AgentHypothesis v1
geometry_manifest_sha256, actor, entities[], relations[], tool_requests[].
Each entity: id,label,motion,segmentation_prompt,confidence,parts[{id,kind,role}], uncertainties/evidence.
Confidence is an agent estimate, not calibrated accuracy.
File agent: agent-prepare -> human/agent fills response -> agent-accept validates.
Remote agent provider is not configured or tested. API credentials are separate from GitHub/HF tokens.

## MaskSet v1
masks[E,T,H,W] bool, ids[E] strings; visibility is observed foreground support, not physical existence.
Import binds to hypothesis and geometry hash. Source-specific SAM candidate IDs need agent identity review.
Do not force multiple detected entities into one identity. Empty/occluded frames are retained explicitly.

## ToolCall v1
tool, modality, input:{path,sha256}, parameters, actor, reason, usage, optional geometry_manifest:{path,sha256}.
usage: agent_preview | geometry_candidate. Existing tools preserve pixels and time.
Invoke allowlisted tool by ID only; no arbitrary imports, URLs or shell commands from agent.
A tool creates a new directory containing result.json and array.npy/report.json.
Records: parent/request/output hashes, tool version, parameters, software versions, runtime, geometry effect.
RGB tools require a NEW VGGT bundle if used as geometry source. Output cannot silently replace raw observations.
Depth filtering retains raw camera bundle and is explicitly a derived-geometry candidate.
Temporal median is experimental/uncompensated and never enabled by default.
Future restoration tool: register schema + implementation + geometry effect; neural weights isolated by backend.
Future crop/resize/stabilization must add an explicit frame/pixel remapping contract before admission; current registry is shape-preserving.

## Geometry/fit prototype
geometry.py reuses pinned reference ray_interval and primitive meshes.
fit_rigid_box is an isolated component fitter (shared size, per-frame center, optional rotation), not the planned full-scene optimizer.
Current silhouette term is projected bbox. Full pixel silhouette, occlusion, floor constraints and real scene fitting remain pending.
Missing depth coverage is penalized in metrics. Numerical depth consistency does not certify cameras.

## Workers
VGGT: existing conda/envs/geometry-forcing, one H20.
SAM3: new conda/envs/sceneblocker-sam3, local checkpoint, no global HF login; source pinned eef6839b.
On-demand worker processes; exchange files rather than importing incompatible model environments into one process.
CPU renderer uses existing numerical environment. Blender optional, not installed for this milestone.

## Scene v1 and implemented joint fitter
Scene: geometry_manifest/hypothesis/masks_manifest artifact references; entities with one shared parts list, one static pose or T rigid poses, observed/renderable flags, ground-normal candidate and relations.
Canonical primitives carry dimensions in canonical_to_entity. entity_to_world is rigid SO3+translation.
Initializer: masked-depth quantiles, floor-normal candidate, local horizontal PCA, bounded missing-time interpolation.
Fitter: fixed input cameras; shared per-entity XYZ shape scales; static center/yaw or per-time rigid center/yaw.
Losses: robust camera-Z depth, physical-scale box/plane surface distance, visibility-gated projected surface-sample mask distance, observed free space, shape regularization, motion and support.
Visibility depends on predicted depth; confidence is not calibrated. Unknown/overlapping mask pixels are excluded from fitting and uncovered output remains reported.
Silhouette is a surface-sample distance approximation, not an exact differentiable renderer. Sphere/cylinder/cone surface-distance terms use an anisotropic approximation.
All current selected times are fitted. Independent held-out evaluation and real-data quality certification are pending.
Renderer reuses reference CPU rasterizer, adjusts its half-pixel convention, and outlines instance boundaries for readability.
Full output: comparison/white/residual/overview videos, numerical rendered depths, per-entity IoU, all-image coverage, trajectories JSON/JPG.

## Agent revision and processed-depth branch
scene-revise accepts only replace_parts and set_renderable, binds parent/request hashes, and produces a new candidate requiring refit.
depth-derive requires geometry_candidate usage and exact parent bundle binding. Preserves raw evidence via parent reference; keeps cameras/time/K identical.
For derived depth, original confidence is retained but fitting confidence is uniform because raw confidence is not calibrated for processed observations.
A processed RGB can feed a NEW vggt-export bundle. It cannot replace an old bundle's raw RGB without hash verification failing.
