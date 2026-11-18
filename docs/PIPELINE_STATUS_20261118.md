# Pipeline audit — 2026-11-18

This is a research prototype with tool-assisted, current-chat Codex reviews. It is not a fully autonomous agent or a trained text-to-4D system. Historical run artifacts are immutable and are not retroactively corrected by source changes.

## Actual execution path

Text → original Wan2.1-T2V-1.3B → selected predicted-clean RGB estimates → joint VGGT depth/confidence/K/c2w → agent entity/motion/representation decisions + SAM3 candidate masks + association → shared coarse geometry/pose fitting → explicit persistence and visibility inference → role/identity colors + shadows → isolated appearance cutouts/interior material swatches → Seedance2.5 → visual comparison/audit.

Wan rain experiment: 50-step UniPC, shift5, CFG5, seed829, 49frames/480square. At iterations5/10/20/40 the preview is VAE(sample − sigma × guided model output), not direct VAE decoding of noisy z_t. This measures the entire RGB/depth/agent reconstruction pipeline, not the information upper bound of internal diffusion features. It does not yet establish a compute advantage once all downstream models and rendering are counted.

## Maturity and gaps

| Component | Implemented / exercised | Material limitation |
|---|---|---|
| Diffusion stage export | Single sampling trajectory, stage-local predicted-clean RGB and latent artifacts | Case scripts, original model; no trained feature readout |
| VGGT | Joint multi-frame depth/confidence/intrinsics/camera export | Predicted geometry, arbitrary scale; cameras can be biased in dynamic/generated scenes |
| Agent | Current chat decides identities, geometry, motion, support, priorities; explicit JSON review | No autonomous video-VLM loop; case decision files contain manual judgments and are not general classifiers |
| SAM/association/LK | Segmentation auxiliary, Hungarian/flow association, candidate sparse tracks | ID splits, partial masks, foreground contamination; LK consistency does not prove body-fixed correspondence |
| Motion semantics | `agent.motion_decision` binds decisions to exact source-scene hash; `scene.fit_agent` rejects unreviewed/unknown or inconsistent labels | Legacy `scene.fit` deliberately remains for historical controls; callers must use guarded route. Unknown alternatives are not automatically branched/scored |
| Coarse fitting | Shared primitive shape; static shared pose or rigid pose sequence; visible-depth/free-space/support losses | Fixed input camera during object fitting; no joint camera-object solver. Box center/hidden extent remain weakly constrained. Velocity/acceleration are soft regularizers, not hard bounds |
| Human proxy | Sphere head, coarse waist-hinge/tube representation and constrained deformation helpers | Dedicated experimental fitter, not a general articulated-body recovery system |
| Persistence | Shared static state, smooth dynamic trajectories and explicit inferred hidden intervals | Smoothing does not validate motion; wrong trajectories may become smoothly wrong |
| Orientation repair | Dominant axial hypothesis, rejected swaps, guarded branch handling | Conservative fallback can suppress genuine turns; no semantic-front guarantee |
| Rendering | Instance importance colors, neutral structural context, world lighting/shadows; fit/render share scene state | Coarse geometry, support surfaces extended by priors, incomplete unseen regions. Source camera path is not necessarily the intended camera move |
| Appearance isolation | Same-stage isolated centered subjects, internally cropped material patches | Single subject cutout still carries pose/view; partial cutouts have incomplete appearance |
| Seedance | Authorized actual API runs from local Mac using uploaded complete white-model videos | Soft reference conditioning; timing/identity/layout may change; adaptive output changes duration/aspect. No verified precise control |
| Reproducibility | Immutable run directories, file hashes, manifests, provenance; unit/synthetic checks | Several scripts are original-server-specific; private reference geometry helpers and model environments required |

## Findings from rainy Step20

- Original Wan clip itself contains additional traffic rather than a faithful single-moving/single-parked story. We should not force the observed RGB into the script.
- Step5 cannot reliably resolve actors. Its Seedance output is script completion, not evidence of successful actor reconstruction. Step10 remains unreliable.
- Yellow instance was originally given a dynamic pose sequence without a motion-model decision. Fitted center advanced by about0.79 world units despite mask median depth staying around1.15–1.18.
- PCA initialization used visible depth extent as a proxy for full-object center. Partially clipped/occluded observations and weak shape/pose binding allowed biased centers, elongated geometry and spurious motion.
- Rotation repair reduced instability but did not fix translation. Static-vs-dynamic diagnostic with original cameras retained similar silhouette IoU (about0.54 vs0.53), but static rendered-depth overlap error was larger (about14% vs7%). Actual outputs differ in persistence treatment; this is diagnosis, not controlled heldout model selection.
- Background-only PnP often improved heldout background projections but did not consistently improve yellow correspondence errors. No evidence yet justifies blaming all apparent object motion on camera underestimation. Near-planar background makes camera translation weakly observable.
- V14 uses a human/current-chat reviewed static decision for yellow and dynamic decisions for the other cars. It verifies a shared static state can be maintained. It does not validate a general automatic motion classifier.

## Audit corrections

1. The previous V15 contact script read V14 `generated.mp4` and labeled it V15. Therefore the old contact and the conclusion based on it were invalid. `check_v15.py` now requires an explicit `--generated` path and a new immutable `--out` directory.
2. Actual V15 output task8167869 differs in SHA256 from V14 task8165834. Reviewing the actual V15 video shows the large foreground passage occurs later than V14, closer to the proxy's later entry. Timing/layout still differ and an extra early nearer vehicle appears. Prompt hardening had some visible effect; it was incorrect to claim it had none from the mislabeled contact. No precise quantitative compliance or causal generalization is established by one sample.
3. Earlier comparison videos called "synced" use the same normalized source-frame map as prior comparisons; adding time labels did not fix Seedance event timing. Equal frame rates/normalized shot positions are not semantic action alignment. Preserve full native outputs and explicitly label any retiming/hold.
4. V14's prompt carried the stale appearance label "world-stationarity unverified" despite a static scene state. New identity bindings append the explicit motion state from the scene. Historical request bodies remain unchanged.
5. Earlier function-style motion tests were not collected by `unittest discover`. Motion and orientation tests are now unittest classes. Release verification runs all47tests, including guarded pipeline application/fitting.

## Next experiments, in order

1. Move agent decisions before geometry initialization for new runs; reject missing motion semantics in the main recipe. Test explicit unknown branches rather than treating uncertainty as unrestricted motion. Current integration starts from an existing scene and can inherit its initialization bias.
2. Bind shared coarse shapes to visible body parts, depth and silhouettes across time. Separate partial visible surfaces from hidden full-shape priors. Evaluate on heldout correspondences/frames and coarse geometry plausibility, not fit loss alone.
3. Camera correction: static background tracks with depth/K priors; fix first camera and shared scale; background holdout gating; then bounded alternating camera/object updates. Prevent dynamic traffic from dominating camera updates. No current joint implementation is claimed.
4. Propagate authoritative agent motion/identity/visibility state into rendering and prompt templates, and audit the exact submitted artifact hashes.
5. Evaluate Seedance separately from reconstruction: same proxy/assets/settings, true output provenance, full duration/aspect retained, record entry/exit/occlusion deviations. Do not hide deviations by post-hoc event warping.

## Guarded pipeline usage

Create an agent JSON containing nonempty `actor`, `source_scene` equal to the exact scene reference `{path, sha256}`, and one `entities` row per instance with `id`, `motion` (`static`, `rigid_dynamic`, `deformable`, `unknown`), and `evidence`. Unknown is a valid review result but cannot enter fitting. Confidence is agent-provided metadata, not a calibrated model probability.

Run `agent.motion_decision` with `scene` and `decisions`; feed its `scene` output to `scene.fit_agent`. The latter uses the primitive fitter and explicitly rejects deformables; use their dedicated fitter after topology review. Legacy fitting and low-level functions remain available for controlled ablations and are not an enforcement boundary against arbitrary code.

Source and tests are published; weights, original RGB, fitted artifacts, video outputs, credentials, installed environments and private reference source are excluded from Git. See README for environment/dependency setup.

## Release verification

2026-11-18: `python -m unittest discover -s tests -v` — 47 tests passed. `python -m compileall -q scene_baseline scripts` passed. `git diff --cached --check` passed. Source credential-pattern scan found no credential literals; weights, outputs, caches and credentials remain excluded. No fresh end-to-end GPU generation was run during the release audit.
