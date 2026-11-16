# Objective and acceptance
Implement persistent entity existence, continuous rigid motion, and explicit separation of observation-supported and inferred states.
All execution, artifacts and caches stay within the authorized GAFU root.

1. A versioned scene state separates lifespan, observation_available, observation_used, visibility explanation, and state source.
2. Existence persists within a continuous shot by default; mask absence never deletes an entity. Explicit lifespan events override that prior.
3. A robust natural cubic trajectory fits shared rigid states in seconds; hidden internal intervals are interpolated and boundary intervals extrapolated with continuous position/velocity/acceleration. Ground contact is an explicit optional prior.
4. Finite-confidence extrapolation is marked inferred, with growing uncertainty proxies. Dense-time sampling checks continuity and ground contact. Unsupported interactions/collisions are diagnosed, not called physically correct.
5. Original RGB/depth/camera observations are preserved. Camera smoothing, if exported, is a separately labeled estimate; calibration must not be silently changed.
6. Tests cover missing interior and boundary observations, disappearance invariants, explicit lifetime events, irregular times, angle wrap, heldout contiguous intervals, rigid transforms and uncertainty growth.
7. Freeze implementation, configuration, agent input policy and metrics before inspecting new generated-case inputs. The two inputs below are preregistered now; no seed selection or regeneration based on results.
8. New cases use the same pipeline/settings. Agent may inspect RGB/script/depth to choose entity identities, motion classes and explicit support priors BEFORE reconstruction; it may not adjust code/templates/thresholds after seeing reconstruction errors.
9. Report all preregistered cases, including generation/detection/reconstruction failures, raw and continuous trajectories, observation/inference videos, global views, heldout tests and observational consistency. New generated videos provide no independent physical 3D truth.

Development cases: existing one/two-car artifacts and synthetic geometric fixtures.
Validation cases: warehouse crossing vehicles and courtyard occlusion, fresh prompts and seeds.
Not claimed: new network training, complete physical simulation, calibrated probabilistic uncertainty, automatic VLM semantic planning.
