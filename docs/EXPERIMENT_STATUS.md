# Recorded experiment status

These are development/regression cases selected with feedback, not a blind generalization evaluation. Depth values have arbitrary common scale and are not meters.

## V3 front/back ambiguity audit

The comparison optimizes only the parked blue vehicle translation. Camera, shape, orientation, other entities and the yellow vehicle trajectory stay fixed.

| Case | Initial visible-entry depth error | After visible-face correction | Overlap frames, before → after |
| --- | ---: | ---: | --- |
| Step 20 | 0.06210 | 0.00365 | 21 → 11 |
| Step 50 | 0.00340 | 0.00465 | 0 → 0 |

For Step 20, the fraction whose exit surface was closer than the entry surface fell from 97.68% to 0.76%. The older unsigned surface loss could fit the back face. Loss normalization also differed between whole-image depth and per-instance surface terms. This evidence explains an important source of stage sensitivity; it does not prove all remaining differences or overlaps share one cause.

The controlled known-geometry test uses identical initialization, camera, shape and 300 optimization steps, with other objectives disabled:

| Surface objective | Mean position error |
| --- | ---: |
| Unsigned SDF | 0.59186 |
| Hard first-hit | 0.51640 |
| Visible bounded faces | 0.02171 |

The visible-face threshold is 0.08. Hard first-hit failed recovery and is retained as a negative result, not the recommended default. V3 supports selected box/slab proxies and raises on unsupported selected primitives; compound self-occlusion is unresolved.

## Earlier recovery work

The v2 synthetic suite has eight behavioral checks. LK recovered 17 early SAM-empty frames in the courtyard development case without manual points. A warehouse regression still needed six manual visual points across frames 24 and 36. These are measured image annotations, not supplied 3D trajectories.

Entity persistence and smoothing reduce disappearance and jitter but do not establish physically correct motion. Collision auditing is available; full physical collision correction is not implemented. Low point reprojection error is not a volume-quality guarantee; see FEATURE_PROXY_ALIGNMENT.md.

Original artifacts remain on the server and are excluded from Git. Historical FREEZE manifests are provenance references; packaged source path adaptations mean they should not be treated as checksums for this checkout.

## Source snapshot validation

Before the initial push, the packaged checkout passed 23 unittest cases, all eight v2 recovery behavioral checks, the v3 visible-face recovery threshold, and the synthetic initialize/fit/render pipeline. The existing server environment and authorized private SceneBlocker reference were used. This does not validate clean-environment installation or rerun the GPU model backends. Generated test artifacts remain excluded from Git.
