# Agent-mimic recovery loop v2

Status: development protocol, not an implemented autonomous VLM backend or validated new fitter.
Actor: the Codex assistant in the current chat. No external model/API call is implied.
Frozen persistence_v1 sources, policies and results remain unchanged.

## Trigger
Escalate to the agent when an expected entity lacks a SAM mask, sparse tracking fails or drifts,
observations disagree across tools, reprojection is visibly wrong, or inferred motion occupies
unsupported space. SAM/flow output is auxiliary evidence, never an existence gate.
The agent can initiate review without waiting for every automatic method to fail.

## Concrete loop
1. Review original RGB at the SAME diffusion stage: whole-clip context, neighboring frames,
   and larger crops of the failed interval. Review original depth/confidence and camera estimates.
2. Decide whether this is visible-but-undetected, partial occlusion, complete occlusion,
   out-of-view, identity ambiguity, or unresolved. Record evidence and alternative explanations.
   Empty masks alone cannot establish occlusion. Script intent is a hypothesis, not visual truth.
3. Add the smallest supported observations: an identity-confirmed region or box; body-fixed
   landmark pixels with visibility; reliably foreground depth patches; support/occlusion relations;
   or an explicit unknown state. Boxes are localization only, not object centers or foreground masks.
   Rotating wheel texture and deforming parts are not rigid-body landmarks.
4. Request a bounded tool operation if useful: prompted segmentation, local point tracking,
   depth inspection or a new scene-fit branch. A tool failure returns to this review loop.
   No particular tracker is claimed to have run until its actual output is saved.
5. Fit complete shared geometry to the accepted evidence. Manually supplied 2D points constrain
   pose only after their 3D correspondence/hypothesis is recorded. Do not silently backproject all
   pixels in a localization box, fabricate a mask, or convert a guessed center into an observation.
   If the current fitter cannot consume an evidence type, mark it unsupported and extend the adapter;
   never claim it affected the fit. Do not reuse the frozen mask-only smoother as if it used new points.
6. Render source-view overlays and an independent global view. Inspect alignment at observed
   locations, visibility/occlusion, shared-shape consistency, motion changes and interpenetration.
   Revise the diagnosed cause, then repeat with immutable child artifacts.
7. Accept only with visual checks and evidence-fit diagnostics. Otherwise retain unresolved
   intervals/multiple hypotheses and state which evidence or component is missing.

## Evidence and provenance
Each iteration records actor, parent scene and geometry hashes, entity and frame interval,
trigger and tool outputs, observed facts, inferred hypotheses, added/rejected observations,
requested changes, rendered diagnostics and accept/revise/unresolved decision.
Sources remain distinct: automatic observation, agent visual annotation, inferred geometry/motion.
Agent-provided numerical pose is a hypothesis even if it looks plausible in a rendered view.
Existence persistence never implies a uniquely known hidden trajectory.
Manual annotations and inferred states must be visually distinguishable in the delivery.
No later diffusion stage is used to repair an earlier stage.

## Generality and budget
Use the same escalation rules and evidence schemas for all cases; no hardcoded vehicle events,
frame boundaries or instance-specific motion corrections in reusable code.
Several agent iterations are allowed. Stop an iteration with an explicit unresolved result when
it adds no new evidence, or different plausible states remain observationally indistinguishable.
Agent-mimic-assisted results are reported separately from tool-only baselines.
After feedback-driven development, freeze the procedure and evaluate on new cases; current
repairs do not establish generalization.

## Confirmed starting diagnosis (not a trajectory annotation)
courtyard_occlusion Step20: SAM returned empty masks for frames0-16 despite a visible yellow
vehicle in inspected RGB; first nonempty mask is frame17. The previous persistence module used
constant-velocity backward extrapolation for those frames. No point-tracking experiment has yet
been run for this recovery branch. Exact early 3D trajectory and occlusion transitions remain
to be established from same-stage evidence, not assumed from the script or another stage.
