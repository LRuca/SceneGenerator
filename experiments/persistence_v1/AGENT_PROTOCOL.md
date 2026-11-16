# Frozen semantic planning protocol
The current chat Codex acts as the agent. No external VLM API is claimed.
Before viewing fitted geometry, inspect each stage's RGB/depth contact sheet and SAM outputs.
Specify actual visible entities, static/rigid status, stable SAM identity bindings and support relation.
The script is a hypothesis; visual evidence takes precedence. No cross-stage masks/geometry transfer.
All entities use a one-box proxy except the ground plane; no sample-specific topology, wheel prompt,
manual numerical pose annotation or orientation constraints on the heldout cases.
Separate SAM instances stay separate unless a recorded identity review demonstrates duplication.
If an intended entity is not generated or detected, record failure; do not synthesize its track.
Lifespans default to the full shot. Only explicit visual/script evidence of creation/destruction
may restrict them, never simple occlusion or leaving the source field of view.
One initial semantic plan per stage; no further plan revisions or numerical changes after fitted
geometry/metrics are inspected. Failures are published with the same frozen procedure.
