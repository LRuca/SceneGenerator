# Code map
scene_baseline/storage.py — boundary enforcement, hashes, immutable output directories.
contracts.py — geometry, agent hypothesis and masks contracts.
vggt.py — same-token camera/depth export, no point head.
agent.py — RGB/depth contact sheets and file-based agent exchange.
tools.py — allowlisted processing/diagnostic calls with role and provenance.
observations.py — explicit derived-depth bundle adoption.
masks.py / sam3_worker.py — imported masks and isolated optional SAM3 worker.
scene.py — scene validation and depth/mask initialization.
fit_scene.py — shared static/rigid scene fitting under fixed cameras.
geometry.py — pinned reference ray intersections/meshes and camera-Z renderer.
delivery.py — white-model/depth/residual/overview videos, per-entity metrics and trajectories.
revision.py — parent-bound agent revisions requiring refit.
pipeline.py — allowlisted staged dispatcher with per-stage status/artifacts.
fixtures.py / tests/ — known-answer synthetic integration and contract tests.
fit.py / smoke.py — earlier isolated box component tests.
__main__.py — CLI commands.

PLAN.md / EXECUTION.md / RUNBOOK.md / docs/INTERFACES.md document scope and execution.
No code command requires waiting for SAM3 downloads to test the core geometry pipeline.
