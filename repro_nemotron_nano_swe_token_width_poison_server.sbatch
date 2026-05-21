#!/bin/bash

#SBATCH --job-name=vllm-nano-width-poison
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --partition=cluster
#SBATCH --exclusive
#SBATCH --output=/beegfs/outputs/vllm-nemotron-nano-width-poison-%j.log
#SBATCH --error=/beegfs/outputs/vllm-nemotron-nano-width-poison-%j.log

set -euo pipefail

cd /home/daniel/git/vllm-nemotron-vllm-repro

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export PATH="/home/daniel/git/vllm/.venv/bin:$PATH"
export HF_HOME=/beegfs/huggingface
export HF_HUB_CACHE=/beegfs/huggingface/hub
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

MODEL="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
PORT="${PORT:-18000}"
CUDAGRAPH_MODE="${CUDAGRAPH_MODE:-FULL_AND_PIECEWISE}"
PAIRS="${PAIRS:-232:227,224:218,216:209,208:201,200:193}"
REPEATS="${REPEATS:-1}"
FILL_MAX_TOKENS="${FILL_MAX_TOKENS:-96}"
PROBE_MAX_TOKENS="${PROBE_MAX_TOKENS:-192}"
FILL_PROMPT_LEN="${FILL_PROMPT_LEN:-1024}"
PROBE_PROMPT_LEN="${PROBE_PROMPT_LEN:-0}"
OUT_DIR="/beegfs/outputs/vllm-nemotron-nano-width-poison-${SLURM_JOB_ID}"
mkdir -p "$OUT_DIR"

SERVER_LOG="$OUT_DIR/server.log"
CLIENT_LOG="$OUT_DIR/client.log"
COMPILATION_CONFIG="{\"cudagraph_mode\":\"$CUDAGRAPH_MODE\"}"

cleanup() {
    if [ -n "${SERVER_PID:-}" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

{
    echo "job_id=$SLURM_JOB_ID"
    echo "node=$SLURM_JOB_NODELIST"
    echo "cwd=$PWD"
    echo "model=$MODEL"
    echo "port=$PORT"
    echo "pairs=$PAIRS"
    echo "repeats=$REPEATS"
    echo "fill_prompt_len=$FILL_PROMPT_LEN"
    echo "probe_prompt_len=$PROBE_PROMPT_LEN"
    echo "fill_max_tokens=$FILL_MAX_TOKENS"
    echo "probe_max_tokens=$PROBE_MAX_TOKENS"
    echo "cudagraph_mode=$CUDAGRAPH_MODE"
    echo "hf_home=$HF_HOME"
    echo "hf_hub_cache=$HF_HUB_CACHE"
    echo "cuda_home=${CUDA_HOME:-}"
    echo "nvcc=$(command -v nvcc || true)"
    echo "ninja=$(command -v ninja || true)"
    /home/daniel/git/vllm/.venv/bin/python - <<'PY'
import importlib.metadata as md
import torch
import vllm

print(f"vllm={md.version('vllm')}")
print(f"vllm_module={vllm.__file__}")
print(f"torch={torch.__version__}")
print(f"cuda={torch.version.cuda}")
PY
} | tee "$OUT_DIR/run.info"

/home/daniel/git/vllm/.venv/bin/python -m vllm.entrypoints.openai.api_server \
    --host 127.0.0.1 \
    --port "$PORT" \
    --model "$MODEL" \
    --served-model-name "$MODEL" \
    --tensor-parallel-size 8 \
    --dtype bfloat16 \
    --max-model-len 131072 \
    --max-num-seqs 256 \
    --gpu-memory-utilization 0.90 \
    --logprobs-mode processed_logprobs \
    --enable-chunked-prefill \
    --no-enable-prefix-caching \
    --return-tokens-as-token-ids \
    --disable-uvicorn-access-log \
    --compilation-config "$COMPILATION_CONFIG" \
    >"$SERVER_LOG" 2>&1 &
SERVER_PID=$!

echo "server_pid=$SERVER_PID" | tee -a "$OUT_DIR/run.info"
echo "waiting_for_health=http://127.0.0.1:$PORT/health" | tee -a "$OUT_DIR/run.info"

READY=0
for attempt in $(seq 1 240); do
    if curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
        READY=1
        echo "server_ready_attempt=$attempt" | tee -a "$OUT_DIR/run.info"
        break
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "server_exited_before_ready" | tee -a "$OUT_DIR/run.info"
        tail -300 "$SERVER_LOG" || true
        exit 1
    fi
    sleep 10
done

if [ "$READY" != "1" ]; then
    echo "server_not_ready_after_timeout" | tee -a "$OUT_DIR/run.info"
    tail -300 "$SERVER_LOG" || true
    exit 1
fi

/home/daniel/git/vllm/.venv/bin/python repro_nemotron_nano_swe_token_width_poison_server.py \
    --base-url "http://127.0.0.1:$PORT/v1" \
    --pairs "$PAIRS" \
    --repeats "$REPEATS" \
    --fill-max-tokens "$FILL_MAX_TOKENS" \
    --probe-max-tokens "$PROBE_MAX_TOKENS" \
    --fill-prompt-len "$FILL_PROMPT_LEN" \
    --probe-prompt-len "$PROBE_PROMPT_LEN" \
    2>&1 | tee "$CLIENT_LOG"
