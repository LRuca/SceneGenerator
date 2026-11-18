# SceneGenerator

Research baseline: **RGB at a selected diffusion stage + script + joint VGGT depth/cameras → coarse white geometry + object trajectories + camera visualization**.

The current route reconstructs a generated RGB estimate. It does not directly decode diffusion features or implement a trained text-to-4D model.

## Latest audit

See [2026-11-18 pipeline status and audit corrections](docs/PIPELINE_STATUS_20261118.md). The recommended motion-reviewed route is `agent.motion_decision` followed by `scene.fit_agent`; legacy `scene.fit` remains an unguarded experimental control. Motion decisions are supplied by the current-chat agent, not inferred automatically. Joint camera/object optimization and hard dynamic speed bounds are not implemented.

## Current components

- Immutable observation manifests, hashes, tool registry, and explicit agent request/review interfaces.
- Joint VGGT depth/confidence/intrinsics/camera export; SAM 3 segmentation as auxiliary evidence.
- Primitive scene initialization, fixed-camera fitting, rendering, error and overlap diagnostics.
- Persistent entities and motion constraints; bidirectional LK tracking and sparse recovery when masks fail.
- Experimental v3 visible-face fitting for front/back depth ambiguity, retaining the old unsigned SDF as a control.

The agent in the recorded experiments was Codex in the chat, with explicit review files. No external VLM API or fully autonomous agent loop was validated. See [geometry limitations](docs/FEATURE_PROXY_ALIGNMENT.md) and [audit results](docs/EXPERIMENT_STATUS.md).

## Setup

Python 3.10+ and PyTorch for optimization are required. Full geometry execution also requires access to the **private** [SceneBlocker](https://github.com/LRuca/SceneBlocker) reference at commit 4a0fa059e24f43a2e0988daed9605b91214be3c7. Its source is intentionally not included. Place an authorized checkout under dependencies/SceneBlocker, or set SCENEGEN_REFERENCE to a checkout beneath your authorized root. Imported helpers: proxy_observations.ray_interval, primitive_composition.unit_mesh, and primitive_composition.transform_mesh.

Use an environment beneath the authorized workspace (on the original server, only gafu/conda). Source env.sh after selecting that environment, then install the package if needed:

    source ./env.sh
    python -m pip install -e '.[fit]'
    python -m scene_baseline --help
    python -m unittest discover -s tests -v
    python scripts/test_recovery_v2.py "$SCENEGEN_BASE/runs/recovery_check"
    python scripts/test_visible_surface_v3.py "$SCENEGEN_BASE/runs/visible_face_check"
    python -m scene_baseline pipeline-run --config configs/synthetic_end_to_end.json --out "$SCENEGEN_BASE/runs/synthetic_pipeline"

Outputs are immutable: choose a new output directory for every run. The synthetic path needs the reference geometry helpers and PyTorch, but not model checkpoints or a GPU.

For VGGT export, use Geometry Forcing's external.vggt implementation (including shortcut_forward) and compatible VGGT-1B weights. Configure SCENEGEN_GF_SOURCE and SCENEGEN_VGGT_WEIGHTS; defaults are dependencies/GeometryForcing and weights/VGGT-1B. This adapter is not a claim of compatibility with every upstream VGGT release. SAM 3 needs its own compatible environment and checkpoint; original server installation/download recipes are retained in scripts/.

## Snapshot scope

This is a source snapshot of the running server prototype, with minimal path configuration changes. Weights, videos, observations, fitted scenes, environments, credentials and third-party source are excluded. It is not a standalone end-to-end model release.

SCENEGEN_BASE defaults to this checkout and SCENEGEN_ROOT to the same directory. Set the root to a wider authorized workspace **before** sourcing env.sh if inputs or reference dependencies are siblings. Storage rejects paths outside that root, including symlink escapes.

Historical plans, configurations, FREEZE*.json and case/report scripts retain original server paths for provenance. Those case scripts are server-specific, may write to the original baseline, and should not be run blindly on a new installation. Historical hashes bind the original code/artifacts, not this adapted checkout; see [original source manifest](docs/ORIGINAL_SOURCE_MANIFEST.json). configs/synthetic_end_to_end.json is the portable smoke configuration. The generic CLI remains scene-baseline / python -m scene_baseline, with scene-generator as an alias.

See [PLAN.md](PLAN.md), [EXECUTION.md](EXECUTION.md), [CODE_MAP.md](CODE_MAP.md), [RUNBOOK.md](RUNBOOK.md) and [interfaces](docs/INTERFACES.md) for development history. Those documents describe plans as well as implemented components; the status and limitations above take precedence over historical ambitions.
