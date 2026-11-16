# Iterative coarse-scene reconstruction V4

Implement the five accepted directions on SceneGenerator, preserving V1-V3 and their artifacts.

1. Fitter: controlled ablations of soft point containment, a fixed local reference convention, staged/alternating optimization, camera-visible depth, and observed free-space constraints. Do not prescribe a pose-graph replacement. Keep negative results and compare combinations at equal input and step budgets.
2. Representation: shared rigid primitive assemblies plus low-dimensional bending/radial cage deformation for people; optimization and rendering consume the same state.
3. Modeling priority: narrative importance and geometric necessity are separate. Key actors/props get action-relevant coarse geometry; support/occlusion/collision boundaries remain geometric even for background entities; unrelated details can be simplified.
4. Agent: current-chat Codex reviews identity, boundaries, motion type, relationships and representation. SAM masks are auxiliary observations, missing masks are not disappearance. Explicit immutable, input-bound review and revision records.
5. Validation: controlled known geometry (including partial visibility, noisy depth, bad tracks and proxy mismatch), then original independent Step20/50 courtyard cases, then deformable-person case. Deliver source/global videos, trajectories, depth/reprojection/occupancy metrics and review queue.

No later diffusion stage may supervise earlier-stage reconstruction. All paths, caches and artifacts stay under gafu. Original cameras remain fixed during this first experiment family. Geometry/visibility errors, tracking residuals and conservative collision bounds must be reported separately.

Execution:
A. Add opt-in V4 fitter terms and immutable ablation runner; verify baseline equivalence before comparing new terms.
B. Run controlled clean/noisy/partial/proxy-mismatch checks with deterministic observations and equal budgets.
C. Inspect evidence, tune only on controlled development cases, freeze a candidate configuration before real-case comparison.
D. Add common policy/review contracts and constrained cage representation in the existing scene pipeline.
E. Run Step20 and Step50 independently; visually review and keep failures. Run a controlled deformable person and an available real/generated human clip if assets permit.
F. Run existing regressions and publish a review page linked to exact source/config hashes. No unconditional quality claim from low point error or lower acceleration alone.
