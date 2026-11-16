# Execution plan and gates

| Stage | Action | Dependencies/downloads | Functional check |
|---|---|---|---|
| E0 | Inventory, persist plan, boundaries | existing geometry-forcing conda | import, GPU, path checks |
| E1 | Artifact/clip/tool/agent contracts | stdlib, numpy, OpenCV/SciPy | reject outside writes, hash drift, invalid shape; tool provenance |
| E2 | Unified VGGT export | existing VGGT weights and code | real stage10 SAME tokens depth+camera; finite SO3/K/transforms |
| E3 | Tool dispatch and preview | existing packages | real RGB denoise/sharpen/depth filter, immutable parent and geometry binding |
| E4 | Agent request + masks adapter | file agent interface; SAM3 repo+authorized weights+isolated env | response schema; masks timeline/IDs; SAM3 real inference when ready |
| E5 | Primitive renderer + fit skeleton | pinned SceneBlocker source; existing SciPy | known camera box depth; fitting known synthetic perturbation; misses penalized |
| E6 | White-model visualization skeleton | CPU reference mesh renderer | known scene source render + depth/residual/camera path |
| E7 | Real clip baseline | validated E2/E4/E5 | stage10 masks and fit, held-out reporting; then stage5 |
| E8 | Agent/tool correction loop | existing contracts | reproducible branch revision and before/after metrics |

## Download/start order
1. Reuse local VGGT, official source snapshot, CUDA runtime; no download/reinstall.
2. Fetch official SAM3 source pinned by commit; inspect current actual README, not stale dependency claims.
3. Download facebook/sam3 from ModelScope (user requested), pinned revision f8ee5d16, verify repository SHA256. HF connectivity failed; no global login change.
4. After requirements and ModelScope file access are known, create conda/envs/sceneblocker-sam3; do not alter geometry-forcing.
5. Start SAM3 as per-job worker, no always-on GPU server required.
6. Start CPU renderer and tool functions on demand. Current local HTTP viewer can serve previews via existing outputs tree.
7. Blender, MASt3R/HSfM, new enhancement neural weights and always-on VLM service are deferred until needed and tested.

## Current status
Implementation and environment checks are recorded in reports/status.json and reports/environment.json.
A passing synthetic fit is a component test; it is not a completed real-world baseline.

## Code milestone completed after download-independent implementation
Implemented CLI/pipeline stages: scene.initialize, scene.fit, scene.render, scene.revise, depth.derive, plus agent/masks/tools/VGGT stages.
Synthetic end-to-end config exercises masked-depth initialization -> shared scene fitting -> delivery.
Component tests cover reference-vs-torch intersections for five primitives, joint scene fit, revision parent binding, geometry mismatch, and explicit processed-depth adoption.
Real Step10 unified geometry/tools are tested. Real SAM3 tracking and remote VLM connection remain backend integration tasks and are not declared passed.
Downloads and environment jobs run independently. No completion gate waits for them when testing core code.

## Agent-mimic recovery v2
When automatic segmentation/tracking is missing or unreliable, the current-chat agent reviews and supplements evidence, requests fitting/rendering, and iterates. See experiments/agent_mimic_v2/AGENT_LOOP.md. This is a development protocol; it does not modify frozen persistence_v1 or claim automatic integration.
