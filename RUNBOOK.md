# Commands and functional checkpoints

Always first:
    cd /home/tione/notebook/home/keithanwang/vlm-vla/gafu
    source sceneblocker/baseline/env.sh

Inspect CLI and processing tool schemas:
    python -m scene_baseline --help
    python -m scene_baseline tool-list

Tests (no external service):
    python -m unittest discover -s tests -v
    python -m scene_baseline synthetic-smoke --out runs/a_new_synthetic_test
Output directories are immutable; use a new name for reruns.

Existing completed same-pass Stage10:
    runs/stage10_geometry/manifest.json
    runs/stage10_geometry/rgb_depth.mp4
Use configs/prepare_existing_stage10.json to prepare a fresh file-agent request:
    python -m scene_baseline pipeline-run --config configs/prepare_existing_stage10.json --out runs/a_new_prepare_run
This only prepares inputs; it does not invoke a remote VLM or fit a real scene.

Next new readout, Stage5:
    CUDA_VISIBLE_DEVICES=1 python -m scene_baseline vggt-export --rgb ../pre-analysis/outputs/wan_joint_single/rgb_step_005.npy --script ../pre-analysis/outputs/wan_joint_single/run.json --out runs/stage5_geometry
Do not mix separate model passes. Readout output includes raw npz and standalone RGB/depth arrays for tools.

Tool examples:
    runs/tool_requests/*.json
    python -m scene_baseline tool-run --request <request.json> --out <new_directory>
Current tools are shape/time preserving. Future motion compensation/cropping requires pixel/time transform support.

SAM3 independent environment:
    source scripts/sam3_env.sh
    CUDA_VISIBLE_DEVICES=2 python -m scene_baseline sam3-track --manifest runs/stage10_geometry/manifest.json --hypothesis runs/stage10_agent_example/hypothesis.json --entity vehicle_0 --checkpoint weights/facebook/sam3/sam3.pt --out runs/stage10_sam3_vehicle
The example hypothesis is initial Codex visual inspection, not a tested remote-VLM result.
Multiple detections remain candidates; no forced union. Review identities before import.
An inference success only tests the segmentation component, not joint geometry accuracy.

Download/install logs:
    logs/download-sam3-modelscope.log
    logs/install-sam3.log
Current source and environment details:
    reports/environment.json
    reports/sam3-readiness.json
    reports/sam3-weight.json (created only after successful SHA256 verification)
    reports/status.json

## End-to-end code path (no downloads)
    python -m scene_baseline pipeline-run --config configs/synthetic_end_to_end.json --out runs/a_new_complete_synthetic_run

Completed example:
    runs/synthetic_end_to_end/
    runs/synthetic_end_to_end_delivery_v2/index.html

Real imported masks path:
    python -m scene_baseline scene-init --manifest <geometry manifest> --hypothesis <accepted hypothesis> --masks-manifest <imported masks manifest> --out <new init dir>
    python -m scene_baseline scene-fit --scene <init/scene.json> --out <new fit dir> --steps 160 --resolution 48 --device cpu
    python -m scene_baseline scene-render --scene <fit/scene.json> --out <new delivery dir> --width 192
    python -m scene_baseline scene-revise --scene <fit/scene.json> --request <revision.json> --out <new revised dir>

Depth processing adoption:
    python -m scene_baseline depth-derive --manifest <raw geometry manifest> --tool-result <geometry_candidate tool result> --out <new derived bundle>
Preview-only results are rejected. Cameras/K/time are inherited; confidence is set to uniform fit weights, with raw_confidence retained.

Independent backend downloads/installations do not block any code command above.

## Real Step20/50 reconstruction and global inspection
Outputs: runs/stage20_50_comparison/index.html.
Real independent per-stage VGGT geometry -> SAM3 tracking -> reviewed stationary scenery grouping
-> scene-init -> 200-step64x64 fixed-camera scene-fit -> scene-render and global_views.
Scene files runs/stage{20,50}_fit/scene.json; delivery_v2 contains source-view/error videos;
global_v2 contains fixed world view, middle-frame orbit, transparent camera-frustum animation, OBJ.
The pipeline is completed but reconstruction quality is NOT accepted: the pillar is depth-elongated,
the car is one box, and single-view hidden extent/camera accuracy remain unvalidated.
All frames fitted; source-depth/mask agreement is not independent 3D accuracy.
Step20 shelves missing masks frames0–9; no cross-stage masks or camera transfer.
Neutral lighting was adjusted for readable white geometry; this changes no depth/shape/metrics.

## Two-car Codex-agent experiment
runs/two_cars_comparison/index.html and report.json document independent Step20/50 processing
of the prior wan_joint clip. SAM3 candidate0=near truck frames0-15; candidate1=far truck
frames10-48 at Step20,11-48 at Step50, verified visually rather than assumed by scale.
R0 two boxes -> R1 two multipart pickups -> reject near multipart regression ->
R2 restore/freeze R0 near and background; optimize only far multipart.
scene-fit --entities far_truck restricts optimization; frozen entities are preserved exactly.
Selected files: runs/two_cars_s{20,50}_fit_r2/scene.json.
global_final has full/vehicle-focused fixed views and transparent paths; overlap has
a frame12 two-car orbit and OBJ (the temporal middle instant has only one observed car).
No extrapolated hidden trajectories. Unmodeled ceiling/windows/table remain missing.
Agent is current chat Codex, not an external API; all decisions and assumptions recorded.

## User-reviewed semantic correction
runs/two_cars_semantic_comparison/index.html compares old/new heading and bookcase topology.
SAM wheel4=rear same-side,5=front same-side,6=front opposite were visually reviewed.
Ray/ground intersections initialize yaw; --entities far_truck shelf_left shelf_right,
per-entity fit_constraints.lock_yaw and 2D contact landmark term preserve semantic front.
Landmarks are serialized in fitted entity scale. Delivery supports
--evaluation-masks-manifest to compare against the same ORIGINAL masks after fit-mask filtering.
Bookcases use two uprights+six boards, thin bounded depth; contents unmodeled.
Final geometry runs/two_cars_s{20,50}_semantic_final/scene.json extends only the ground
patch XZ footprint beneath geometry; same normal/height, unobserved extension is a prior.
Full coverage worsens because books/occluders remain missing; vehicle silhouette improves.
Table SAM query failed/fragmented and was not silently replaced.
