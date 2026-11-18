# Role-aware proxy video and identity bindings

`proxy.encode` requires explicit agent decisions for every entity. Narrative importance is independent of motion: a parked pickup can be a key prop. Important actors/props and supporting identities receive stable distinct saturated colors. Structural entities retain neutral gray and their fitted occlusion/support geometry. This stage annotates the scene and never changes shapes, poses, fitting budgets or existence flags. Existing modeling_policy_v4 continues to own fitting/representation budgets.

`blocking.package` verifies and reuses the isolated appearance references, then writes full-video conditions and prompt identity bindings from the same encoding table. It rejects missing appearance bindings or unsafe reference packages. No source RGB frame/video is submitted. Reference index/identity consistency is checked.

The camera intervention is stored separately from observations: smoothstep translation along first-camera right projected to the ground plane, keeping source rotations and K. Negative displacement trucks left; it does not pan or zoom. Original VGGT cameras remain immutable. The package camera_trajectory.npz is authoritative for the modified render; source scene.json still identifies observation geometry.

Example: courtyard_pipeline.json; SceneGenerator env plus SCENEGEN_REFERENCE must be configured as for the existing renderer. Run `python -m scene_baseline pipeline-run --config experiments/proxy_encoding_v8/courtyard_pipeline.json --out runs/<new-name>`.

Latest run: runs/v8_courtyard_identity_truckleft_r2. r1 preserved an environment-path failure before render, corrected by explicitly setting SCENEGEN_REFERENCE, with no fallback renderer. Original same-stage VGGT, SAM masks and fit are reused; no upstream model inference or refit is claimed. Colors: moving forklift yellow/Image1; parked pickup blue/Image2; ground/wall/kiosk neutral gray. Translation: 0.6 fitted forklift length toward camera-left (arbitrary VGGT scale).

Outputs: render_comparison.mp4 (previous palette/original camera, encoded/original camera, encoded/translated camera), white_model_reference.mp4 (full 4s,720 square,24fps), camera_trajectory.npz, role/identity table, prompt, manifest and visible-pixel diagnostics. Modified-camera RGB reprojection cannot be compared to original view as ground truth. Newly exposed hidden surfaces remain coarse unobserved priors. Geometry quality and vehicle orientation are unchanged by this experiment.

Local submission uses the existing working Venus COS upload+Seedance helper, seed40124, adaptive/-1 provider settings; returned job and actual uploaded request must be retained beside the local package. This is an explicit remote service call, not an automatic pipeline fallback. The generation changes both palette encoding and camera; the three-way render comparison isolates them at the condition stage, not at the generated-video stage.

Validation: 35 unittest checks passed (including identity/appearance consistency and ground-parallel camera intervention).
