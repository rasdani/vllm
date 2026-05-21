#!/bin/bash

#SBATCH --job-name=vllm-nano-chat-ring
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --partition=cluster
#SBATCH --exclusive
#SBATCH --output=/beegfs/outputs/vllm-nemotron-nano-chat-ring-%j.log
#SBATCH --error=/beegfs/outputs/vllm-nemotron-nano-chat-ring-%j.log

set -euo pipefail

cd /home/daniel/git/vllm-nemotron-vllm-repro

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export HF_HOME=/beegfs/huggingface
export HF_HUB_CACHE=/beegfs/huggingface/hub
export TRITON_CACHE_DIR=/tmp/triton-cache-$USER-$SLURM_JOB_ID
export VLLM_CACHE_ROOT=/tmp/.vllm_cache_$USER
export FLASHINFER_WORKSPACE_BASE=/tmp/flashinfer-workspace-$USER-$SLURM_JOB_ID

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
    export PATH="$CUDA_HOME/bin:$PATH"
    export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
fi

MODEL="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
PORT="${PORT:-18001}"
CUDAGRAPH_MODE="${CUDAGRAPH_MODE:-FULL_AND_PIECEWISE}"
DUMP_PATH="${DUMP_PATH:-/beegfs/daniel/nemotron-nano-swe-step7-default-128k/chat_nan_diagnostics/1778796420266_chat_nonfinite_response_949b56786333e335.json}"
CHAT_REPEAT="${CHAT_REPEAT:-8}"
CHAT_CONCURRENCY="${CHAT_CONCURRENCY:-512}"
CHAT_LIMIT="${CHAT_LIMIT:-0}"
CHAT_SORT="${CHAT_SORT:-none}"
CHAT_FORCE_IGNORE_EOS="${CHAT_FORCE_IGNORE_EOS:-1}"
CHAT_MAX_COMPLETION_TOKENS="${CHAT_MAX_COMPLETION_TOKENS:-2048}"
OUT_DIR="/beegfs/outputs/vllm-nemotron-nano-chat-ring-${SLURM_JOB_ID}"
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
    echo "dump_path=$DUMP_PATH"
    echo "chat_repeat=$CHAT_REPEAT"
    echo "chat_concurrency=$CHAT_CONCURRENCY"
    echo "chat_limit=$CHAT_LIMIT"
    echo "chat_sort=$CHAT_SORT"
    echo "chat_force_ignore_eos=$CHAT_FORCE_IGNORE_EOS"
    echo "chat_max_completion_tokens=$CHAT_MAX_COMPLETION_TOKENS"
    echo "cudagraph_mode=$CUDAGRAPH_MODE"
    echo "hf_home=$HF_HOME"
    echo "hf_hub_cache=$HF_HUB_CACHE"
    echo "cuda_home=${CUDA_HOME:-}"
    echo "nvcc=$(command -v nvcc || true)"
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
    --max-num-seqs 512 \
    --gpu-memory-utilization 0.90 \
    --logprobs-mode processed_logprobs \
    --enable-chunked-prefill \
    --no-enable-prefix-caching \
    --return-tokens-as-token-ids \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
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

replay_args=()
if [ "$CHAT_FORCE_IGNORE_EOS" = "1" ]; then
    replay_args+=(--force-ignore-eos)
fi
if [ "$CHAT_LIMIT" != "0" ]; then
    replay_args+=(--limit "$CHAT_LIMIT")
fi

/home/daniel/git/vllm/.venv/bin/python repro_nemotron_nano_swe_chat_ring_server.py \
    --base-url "http://127.0.0.1:$PORT/v1" \
    --input "$DUMP_PATH" \
    --repeat "$CHAT_REPEAT" \
    --concurrency "$CHAT_CONCURRENCY" \
    --sort-by-content-length "$CHAT_SORT" \
    --max-completion-tokens "$CHAT_MAX_COMPLETION_TOKENS" \
    "${replay_args[@]}" \
    2>&1 | tee "$CLIENT_LOG"
