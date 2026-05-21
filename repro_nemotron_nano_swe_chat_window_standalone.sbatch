#!/bin/bash

#SBATCH --job-name=vllm-nano-swe-repro
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --partition=cluster
#SBATCH --exclusive
#SBATCH --output=/beegfs/outputs/vllm-nemotron-nano-swe-chat-window-%j.log
#SBATCH --error=/beegfs/outputs/vllm-nemotron-nano-swe-chat-window-%j.log

set -euo pipefail

cd /home/daniel/git/vllm-nemotron-vllm-repro

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export PATH="/home/daniel/git/vllm/.venv/bin:$PATH"
export HF_HOME=/beegfs/huggingface
export HF_HUB_CACHE=/beegfs/huggingface/hub
export VLLM_USE_DEEP_GEMM=0
export TRITON_CACHE_DIR=/tmp/triton-cache-$USER-$SLURM_JOB_ID
export VLLM_CACHE_ROOT=/tmp/.vllm_cache_$USER
export FLASHINFER_WORKSPACE_BASE=/tmp/flashinfer-workspace-$USER-$SLURM_JOB_ID

EXTENSION_SOURCE="/home/daniel/git/vllm/vllm"
for rel in \
    "_C.abi3.so" \
    "_C_stable_libtorch.abi3.so" \
    "_flashmla_C.abi3.so" \
    "_flashmla_extension_C.abi3.so" \
    "_moe_C.abi3.so" \
    "cumem_allocator.abi3.so" \
    "spinloop.abi3.so" \
    "third_party/deep_gemm/_C.cpython-312-x86_64-linux-gnu.so" \
    "vllm_flash_attn/_vllm_fa2_C.abi3.so" \
    "vllm_flash_attn/_vllm_fa3_C.abi3.so"; do
    mkdir -p "vllm/$(dirname "$rel")"
    if [ ! -e "vllm/$rel" ]; then
        ln -s "$EXTENSION_SOURCE/$rel" "vllm/$rel"
    fi
done

unset CUDA_HOME CUDA_PATH
if [ -x /usr/local/cuda/bin/nvcc ]; then
    export CUDA_HOME=/usr/local/cuda
elif [ -x /usr/local/cuda-13.1/bin/nvcc ]; then
    export CUDA_HOME=/usr/local/cuda-13.1
elif [ -x /usr/local/cuda-12.8/bin/nvcc ]; then
    export CUDA_HOME=/usr/local/cuda-12.8
fi
if [ -n "${CUDA_HOME:-}" ]; then
    export CUDA_PATH="$CUDA_HOME"
    export PATH="/home/daniel/git/vllm/.venv/bin:$CUDA_HOME/bin:$PATH"
    export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
fi

CUDAGRAPH_MODE="${CUDAGRAPH_MODE:-FULL_AND_PIECEWISE}"
SCENARIO="${SCENARIO:-recent64-plus-failure}"
OUT_DIR="/beegfs/outputs/vllm-nemotron-nano-swe-chat-window-${SLURM_JOB_ID}"
mkdir -p "$OUT_DIR"

{
    echo "job_id=$SLURM_JOB_ID"
    echo "node=$SLURM_JOB_NODELIST"
    echo "cwd=$PWD"
    echo "script=repro_nemotron_nano_swe_chat_window_standalone.py"
    echo "scenario=$SCENARIO"
    echo "cudagraph_mode=$CUDAGRAPH_MODE"
    echo "hf_home=$HF_HOME"
    echo "hf_hub_cache=$HF_HUB_CACHE"
    echo "vllm_use_deep_gemm=$VLLM_USE_DEEP_GEMM"
    echo "cuda_home=${CUDA_HOME:-}"
    echo "nvcc=$(command -v nvcc || true)"
    echo "ninja=$(command -v ninja || true)"
    echo "flashinfer_workspace_base=$FLASHINFER_WORKSPACE_BASE"
    /home/daniel/git/vllm/.venv/bin/python - <<'PY'
import importlib.metadata as md
import torch
import vllm

print(f"vllm={md.version('vllm')}")
print(f"vllm_module={vllm.__file__}")
print(f"torch={torch.__version__}")
print(f"cuda={torch.version.cuda}")
try:
    import flashinfer
except Exception as exc:
    print(f"flashinfer_import_error={type(exc).__name__}: {exc}")
else:
    print(f"flashinfer={flashinfer.__version__}")
    print(f"flashinfer_module={flashinfer.__file__}")
PY
} | tee "$OUT_DIR/run.info"

/home/daniel/git/vllm/.venv/bin/python repro_nemotron_nano_swe_chat_window_standalone.py \
    --scenario "$SCENARIO" \
    --cudagraph-mode "$CUDAGRAPH_MODE" \
    2>&1 | tee "$OUT_DIR/repro.log"
