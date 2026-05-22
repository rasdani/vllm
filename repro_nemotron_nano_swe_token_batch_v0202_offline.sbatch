#!/bin/bash

#SBATCH --job-name=vllm0202-nano-offline
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --partition=cluster
#SBATCH --exclusive
#SBATCH --output=/beegfs/outputs/vllm0202-nemotron-nano-offline-%j.log
#SBATCH --error=/beegfs/outputs/vllm0202-nemotron-nano-offline-%j.log

set -euo pipefail

VLLM0202_DIR="/home/daniel/git/vllm-nemotron-v0202-repro"
CLIENT_DIR="/home/daniel/git/vllm-nemotron-vllm-repro"
cd "$VLLM0202_DIR"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export VENV="$VLLM0202_DIR/.venv"
export PYTHONPATH="$VLLM0202_DIR:$CLIENT_DIR:${PYTHONPATH:-}"
export PATH="$VENV/bin:$PATH"
export HF_HOME=/beegfs/huggingface
export HF_HUB_CACHE=/beegfs/huggingface/hub
export VLLM_USE_DEEP_GEMM=0
export TRITON_CACHE_DIR=/tmp/triton-cache-$USER-$SLURM_JOB_ID
export VLLM_CACHE_ROOT=/tmp/.vllm_cache_${USER}_v0202
export FLASHINFER_WORKSPACE_BASE=/tmp/flashinfer-workspace-$USER-$SLURM_JOB_ID

EXTENSION_SOURCE="$VENV/lib/python3.12/site-packages/vllm"
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
    source_path="$EXTENSION_SOURCE/$rel"
    dest_path="vllm/$rel"
    mkdir -p "$(dirname "$dest_path")"
    if [ ! -e "$source_path" ]; then
        if [ -L "$dest_path" ] && [ "$(readlink "$dest_path")" = "$source_path" ]; then
            rm "$dest_path"
        fi
        continue
    fi
    if [ -L "$dest_path" ] && [ "$(readlink "$dest_path")" != "$source_path" ]; then
        rm "$dest_path"
    fi
    if [ ! -e "$dest_path" ] && [ ! -L "$dest_path" ]; then
        ln -s "$source_path" "$dest_path"
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
    export PATH="$VENV/bin:$CUDA_HOME/bin:$PATH"
    export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
fi

MODEL="${MODEL:-nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16}"
TP_SIZE="${TP_SIZE:-1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-256}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.95}"
CUDAGRAPH_MODE="${CUDAGRAPH_MODE:-FULL_AND_PIECEWISE}"

{
    echo "job_id=$SLURM_JOB_ID"
    echo "node=$SLURM_JOB_NODELIST"
    echo "cwd=$PWD"
    echo "client_dir=$CLIENT_DIR"
    echo "model=$MODEL"
    echo "tp_size=$TP_SIZE"
    echo "max_model_len=$MAX_MODEL_LEN"
    echo "max_num_seqs=$MAX_NUM_SEQS"
    echo "gpu_memory_utilization=$GPU_MEMORY_UTILIZATION"
    echo "cudagraph_mode=$CUDAGRAPH_MODE"
    echo "hf_home=$HF_HOME"
    echo "hf_hub_cache=$HF_HUB_CACHE"
    echo "vllm_use_deep_gemm=$VLLM_USE_DEEP_GEMM"
    echo "cuda_home=${CUDA_HOME:-}"
    echo "nvcc=$(command -v nvcc || true)"
    "$VENV/bin/python" - <<'PY'
import importlib.metadata as md
import torch
import vllm

print(f"vllm={md.version('vllm')}")
print(f"vllm_module={vllm.__file__}")
print(f"torch={torch.__version__}")
print(f"cuda={torch.version.cuda}")
PY
} | tee "/beegfs/outputs/vllm0202-nemotron-nano-offline-${SLURM_JOB_ID}.info"

"$VENV/bin/python" "$CLIENT_DIR/repro_nemotron_nano_swe_token_batch_offline.py" \
    --model "$MODEL" \
    --tensor-parallel-size "$TP_SIZE" \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --cudagraph-mode "$CUDAGRAPH_MODE"
