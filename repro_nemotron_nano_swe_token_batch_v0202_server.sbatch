#!/bin/bash

#SBATCH --job-name=vllm0202-nano-token
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --partition=cluster
#SBATCH --exclusive
#SBATCH --output=/beegfs/outputs/vllm0202-nemotron-nano-token-batch-%j.log
#SBATCH --error=/beegfs/outputs/vllm0202-nemotron-nano-token-batch-%j.log

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
export VLLM_ALLOW_RUNTIME_LORA_UPDATING="${VLLM_ALLOW_RUNTIME_LORA_UPDATING:-True}"
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
PORT="${PORT:-18006}"
CUDAGRAPH_MODE="${CUDAGRAPH_MODE:-FULL_AND_PIECEWISE}"
ROLLOUT_PATH="${ROLLOUT_PATH:-/beegfs/daniel/nemotron-nano-swe-step7-default-128k/run_default/rollouts/step_0/train_rollouts.jsonl}"
TARGET_TRACE="${TARGET_TRACE:-/beegfs/daniel/nemotron-nano-swe-collector-20260521-235954/vllm_nan_trace/cudagraph_replay_nonfinite.1236436.jsonl}"
TARGET_WIDTH="${TARGET_WIDTH:-191}"
REPLAY_COUNT="${REPLAY_COUNT:-191}"
MAX_ROLLOUTS="${MAX_ROLLOUTS:-512}"
MAX_CANDIDATES="${MAX_CANDIDATES:-8192}"
MIN_TURN_INDEX="${MIN_TURN_INDEX:-4}"
MAX_TURN_INDEX="${MAX_TURN_INDEX:-9999}"
TOKEN_CONCURRENCY="${TOKEN_CONCURRENCY:-191}"
TOKEN_MAX_TOKENS="${TOKEN_MAX_TOKENS:-512}"
TOKEN_TEMPERATURE="${TOKEN_TEMPERATURE:-1.0}"
LORA_NAME="${LORA_NAME:-nemotron-step1-lora}"
LORA_PATH="${LORA_PATH:-/beegfs/daniel/nemotron-nano-swe-step7-default-128k/run_default/broadcasts/step_1}"
LORA_LOAD_INPLACE_CYCLES="${LORA_LOAD_INPLACE_CYCLES:-1}"
TP_SIZE="${TP_SIZE:-8}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-1024}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"

OUT_DIR="/beegfs/outputs/vllm0202-nemotron-nano-token-batch-${SLURM_JOB_ID}"
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
    echo "client_dir=$CLIENT_DIR"
    echo "model=$MODEL"
    echo "port=$PORT"
    echo "rollout_path=$ROLLOUT_PATH"
    echo "target_trace=$TARGET_TRACE"
    echo "target_width=$TARGET_WIDTH"
    echo "replay_count=$REPLAY_COUNT"
    echo "max_rollouts=$MAX_ROLLOUTS"
    echo "max_candidates=$MAX_CANDIDATES"
    echo "min_turn_index=$MIN_TURN_INDEX"
    echo "max_turn_index=$MAX_TURN_INDEX"
    echo "token_concurrency=$TOKEN_CONCURRENCY"
    echo "token_max_tokens=$TOKEN_MAX_TOKENS"
    echo "token_temperature=$TOKEN_TEMPERATURE"
    echo "lora_name=$LORA_NAME"
    echo "lora_path=$LORA_PATH"
    echo "lora_load_inplace_cycles=$LORA_LOAD_INPLACE_CYCLES"
    echo "tp_size=$TP_SIZE"
    echo "max_model_len=$MAX_MODEL_LEN"
    echo "max_num_seqs=$MAX_NUM_SEQS"
    echo "gpu_memory_utilization=$GPU_MEMORY_UTILIZATION"
    echo "cudagraph_mode=$CUDAGRAPH_MODE"
    echo "hf_home=$HF_HOME"
    echo "hf_hub_cache=$HF_HUB_CACHE"
    echo "vllm_use_deep_gemm=$VLLM_USE_DEEP_GEMM"
    echo "vllm_allow_runtime_lora_updating=$VLLM_ALLOW_RUNTIME_LORA_UPDATING"
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
} | tee "$OUT_DIR/run.info"

"$VENV/bin/python" -m vllm.entrypoints.openai.api_server \
    --host 127.0.0.1 \
    --port "$PORT" \
    --model "$MODEL" \
    --served-model-name "$MODEL" \
    --tensor-parallel-size "$TP_SIZE" \
    --dtype bfloat16 \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --logprobs-mode processed_logprobs \
    --enable-chunked-prefill \
    --no-enable-prefix-caching \
    --return-tokens-as-token-ids \
    --disable-uvicorn-access-log \
    --enable-lora \
    --max-loras 12 \
    --max-cpu-loras 100 \
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

"$VENV/bin/python" "$CLIENT_DIR/repro_nemotron_nano_swe_token_batch_server.py" \
    --base-url "http://127.0.0.1:$PORT/v1" \
    --rollouts "$ROLLOUT_PATH" \
    --target-trace "$TARGET_TRACE" \
    --target-width "$TARGET_WIDTH" \
    --replay-count "$REPLAY_COUNT" \
    --max-rollouts "$MAX_ROLLOUTS" \
    --max-candidates "$MAX_CANDIDATES" \
    --min-turn-index "$MIN_TURN_INDEX" \
    --max-turn-index "$MAX_TURN_INDEX" \
    --concurrency "$TOKEN_CONCURRENCY" \
    --max-tokens "$TOKEN_MAX_TOKENS" \
    --temperature "$TOKEN_TEMPERATURE" \
    --lora-name "$LORA_NAME" \
    --lora-path "$LORA_PATH" \
    --lora-load-inplace-cycles "$LORA_LOAD_INPLACE_CYCLES" \
    2>&1 | tee "$CLIENT_LOG"
