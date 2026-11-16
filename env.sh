#!/usr/bin/env bash
# Source after selecting your environment; no environment activation or installation.
export SCENEGEN_BASE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export SCENEGEN_ROOT="${SCENEGEN_ROOT:-$SCENEGEN_BASE}"
export SB_BASELINE_ROOT="$SCENEGEN_BASE"
export PYTHONPATH="$SCENEGEN_BASE${PYTHONPATH:+:$PYTHONPATH}"
case "$SCENEGEN_BASE/" in "$SCENEGEN_ROOT/"*) ;; *) echo "Project is outside SCENEGEN_ROOT" >&2; return 1 ;; esac
export TMPDIR="$SCENEGEN_BASE/.runtime/tmp"
export XDG_CACHE_HOME="$SCENEGEN_BASE/.runtime/cache"
export HF_HOME="$SCENEGEN_BASE/.runtime/huggingface"
export PIP_CACHE_DIR="$SCENEGEN_BASE/.runtime/pip"
export CONDA_PKGS_DIRS="$SCENEGEN_BASE/.runtime/conda-pkgs"
export TORCH_HOME="$SCENEGEN_BASE/.runtime/torch"
export TORCH_EXTENSIONS_DIR="$SCENEGEN_BASE/.runtime/torch-extensions"
export TRITON_CACHE_DIR="$SCENEGEN_BASE/.runtime/triton"
export CUDA_CACHE_PATH="$SCENEGEN_BASE/.runtime/cuda"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$HF_HOME" "$PIP_CACHE_DIR" "$CONDA_PKGS_DIRS" "$TORCH_HOME" "$TORCH_EXTENSIONS_DIR" "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH" "$SCENEGEN_BASE/runs/tests"
cd "$SCENEGEN_BASE"
