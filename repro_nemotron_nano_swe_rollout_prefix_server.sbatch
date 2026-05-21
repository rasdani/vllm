#!/bin/bash

#SBATCH --job-name=vllm-nano-rollout-prefix
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --partition=cluster
#SBATCH --exclusive
#SBATCH --output=/beegfs/outputs/vllm-nemotron-nano-rollout-prefix-%j.log
#SBATCH --error=/beegfs/outputs/vllm-nemotron-nano-rollout-prefix-%j.log

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
export VLLM_DEBUG_PADDED_INPUT_IDS="${VLLM_DEBUG_PADDED_INPUT_IDS:-0}"
export VLLM_DEBUG_PADDED_INPUT_IDS_LIMIT="${VLLM_DEBUG_PADDED_INPUT_IDS_LIMIT:-40}"
export VLLM_REPRO_SHAPE_TRACE_LIMIT="${VLLM_REPRO_SHAPE_TRACE_LIMIT:-20000}"
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
PORT="${PORT:-18002}"
CUDAGRAPH_MODE="${CUDAGRAPH_MODE:-FULL_AND_PIECEWISE}"
ROLLOUT_PATH="${ROLLOUT_PATH:-/beegfs/daniel/nemotron-nano-swe-step7-default-128k/run_default/rollouts/step_0/train_rollouts.jsonl}"
MAX_ROLLOUTS="${MAX_ROLLOUTS:-512}"
MAX_REQUESTS="${MAX_REQUESTS:-2048}"
MIN_TURN_INDEX="${MIN_TURN_INDEX:-0}"
MAX_TURN_INDEX="${MAX_TURN_INDEX:-9999}"
CHAT_CONCURRENCY="${CHAT_CONCURRENCY:-512}"
CHAT_MAX_COMPLETION_TOKENS="${CHAT_MAX_COMPLETION_TOKENS:-256}"
CHAT_ALLOWED_TOKEN_IDS="${CHAT_ALLOWED_TOKEN_IDS:-}"
CHAT_FORCE_IGNORE_EOS="${CHAT_FORCE_IGNORE_EOS:-0}"
OUT_DIR="/beegfs/outputs/vllm-nemotron-nano-rollout-prefix-${SLURM_JOB_ID}"
mkdir -p "$OUT_DIR"
export VLLM_REPRO_SHAPE_TRACE_DIR="$OUT_DIR/shape_trace"
mkdir -p "$VLLM_REPRO_SHAPE_TRACE_DIR"

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
    echo "rollout_path=$ROLLOUT_PATH"
    echo "max_rollouts=$MAX_ROLLOUTS"
    echo "max_requests=$MAX_REQUESTS"
    echo "min_turn_index=$MIN_TURN_INDEX"
    echo "max_turn_index=$MAX_TURN_INDEX"
    echo "chat_concurrency=$CHAT_CONCURRENCY"
    echo "chat_max_completion_tokens=$CHAT_MAX_COMPLETION_TOKENS"
    echo "chat_allowed_token_ids=$CHAT_ALLOWED_TOKEN_IDS"
    echo "chat_force_ignore_eos=$CHAT_FORCE_IGNORE_EOS"
    echo "cudagraph_mode=$CUDAGRAPH_MODE"
    echo "hf_home=$HF_HOME"
    echo "hf_hub_cache=$HF_HUB_CACHE"
    echo "vllm_use_deep_gemm=$VLLM_USE_DEEP_GEMM"
    echo "vllm_debug_padded_input_ids=$VLLM_DEBUG_PADDED_INPUT_IDS"
    echo "vllm_debug_padded_input_ids_limit=$VLLM_DEBUG_PADDED_INPUT_IDS_LIMIT"
    echo "vllm_repro_shape_trace_dir=$VLLM_REPRO_SHAPE_TRACE_DIR"
    echo "vllm_repro_shape_trace_limit=$VLLM_REPRO_SHAPE_TRACE_LIMIT"
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
    --max-num-seqs 1024 \
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

/home/daniel/git/vllm/.venv/bin/python repro_nemotron_nano_swe_rollout_prefix_server.py \
    --base-url "http://127.0.0.1:$PORT/v1" \
    --rollouts "$ROLLOUT_PATH" \
    --max-rollouts "$MAX_ROLLOUTS" \
    --max-requests "$MAX_REQUESTS" \
    --min-turn-index "$MIN_TURN_INDEX" \
    --max-turn-index "$MAX_TURN_INDEX" \
    --concurrency "$CHAT_CONCURRENCY" \
    --max-completion-tokens "$CHAT_MAX_COMPLETION_TOKENS" \
    --allowed-token-ids "$CHAT_ALLOWED_TOKEN_IDS" \
    "${replay_args[@]}" \
    2>&1 | tee "$CLIENT_LOG"
