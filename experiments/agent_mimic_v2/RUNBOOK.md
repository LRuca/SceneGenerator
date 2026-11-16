# Recovery V2 execution and limits

All commands execute under gafu/sceneblocker/baseline after sourcing env.sh. No new environment,
weights or external VLM service is required. OpenCV LK, existing VGGT data and existing SAM masks
are reused. The current-chat Codex performs the semantic reviews.

## Stages
python -m scene_baseline.recovery_pipeline_v2 prepare --config CONFIG.json --out NEW_PREPARED
This runs tracking (or reuses hash-bound tracks), produces candidate overlays and a current-chat
agent handoff. It does not ask the human user to annotate frames.

The agent records accept_candidate_evidence, or supplement_observations with sparse visual points
and explicit rejected tracks. agent_recovery_v2 performs local LK propagation if requested.
Inferred pixel/pose hypotheses are not inserted as measured observations. Each review produces
a new artifact branch with original mask/depth/camera files unchanged.

python -m scene_baseline.recovery_pipeline_v2 resume --prepared NEW_PREPARED/prepared.json --review REVIEW.json --out NEW_RESULT
This fits optional reviewed static surfaces, shared body landmarks and rigid poses, then available
mask/depth evidence, and renders source/global comparison and a validation_queue.json.
The current-chat agent reviews the render and writes a new agent_validation.json receipt.
Automatic acceptance is disabled because point-fit and physical correctness differ.

## Source map
tracking_v2.py: anchor selection, bidirectional LK, per-pair checks, failed-interval review request.
agent_recovery_v2.py: import visual annotations/rejections, propagate points, preserve source labels.
sparse_fit_v2.py: fixed-camera joint shared-body point/pose fitting; missing-mask frames can use points.
surfaces_v2.py: robust visible vertical-plane fit with explicit thickness prior.
fit_scene_v2.py: available-mask fitting plus sparse landmarks; empty mask does not penalize silhouette.
recovery_pipeline_v2.py: prepare/resume orchestration and artifact binding.
recovery_view_v2.py: actual final-pose reprojection, source/global videos and collision report.
scripts/test_recovery_v2.py: known-motion, zero-evidence, provenance, binding, numerical stability tests.

## Important limits
Vehicle mesh remains a shared coarse proxy inherited from the existing baseline. It is not a
learned amodal reconstruction. Body points refine visible geometry; low point reprojection error
does not validate the hidden vehicle volume, VGGT cameras/depth, or entire trajectory.
Fixed robust motion regularization still exists, although it has linear tails; a rich switching
motion model or probabilistic multi-hypothesis hidden-trajectory posterior is not implemented.
Unsupported times remain tentative joint-fit hypotheses and are labeled POSE UNCERTAIN.
Collision checking uses conservative primitive bounds and does not correct collisions.
Thin-wall backside thickness is a declared prior, not recovered physical thickness.
The current agent manually launches prepare/resume tool calls; there is no unattended VLM loop.

## Evaluation
8 component behavioral checks pass in pv2_recovery_component_tests_r4.
Warehouse required 6 manual visual points on only 2 frames, no numerical 3D pose annotation.
Some unsupported occlusion/out-of-view intervals remain. Current real cases are development and
regression, not new blind generalization evidence. All frozen persistence_v1 source hashes remain intact.
