# Baseline development boundary
All writes, installs, caches, logs and artifacts must stay under /home/tione/notebook/home/keithanwang/vlm-vla/gafu. Follow its AGENTS.md.
New environments belong under gafu/conda. Never source the reference repository's env.sh.
Preserve raw observations. Tools produce new immutable artifact directories and provenance. Do not silently overwrite geometry or recalibrate scales.
Never use later diffusion stages to improve an early-stage fit. Keep external agent requests explicit and exclude credentials from artifacts.
