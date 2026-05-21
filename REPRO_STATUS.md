# Nemotron Nano SWE vLLM-only NaN repro

## Goal

Find the smallest vLLM-only reproduction for the Nemotron Nano SWE NaN/non-finite
response seen in training. This branch intentionally excludes prime-rl,
verifiers, Prime sandboxes, and Prime tunnels.

## Baseline

- vLLM base: `origin/main` at `a10d69116cb25c8137eeb3f320add71d4e04fda9`
- Model: `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`
- TP: 8
- dtype: `bfloat16`
- max model length: `131072`
- default CUDA graph mode under test: `FULL_AND_PIECEWISE`
- HF cache: `/beegfs/huggingface`
- Local sbatch wrappers reuse the compiled extension `.so` files from
  `/home/daniel/git/vllm/vllm` via untracked symlinks. This keeps Python imports
  pointed at this fresh worktree while avoiding a full local vLLM rebuild.

## Real data source

Primary captured failure:

```text
/beegfs/daniel/nemotron-nano-swe-step7-default-128k/chat_nan_diagnostics/1778796420266_chat_nonfinite_response_949b56786333e335.json
```

Important fields:

- `request_id`: `949b56786333e335`
- `model`: `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`
- `recent_requests`: 64 real chat requests captured near the failure boundary
- `response_payload.prompt_token_ids`: 12,853 token IDs for the failing request

Secondary shape evidence from the latest collector run:

```text
/beegfs/daniel/nemotron-nano-swe-collector-20260521-235954/vllm_nan_trace/cudagraph_replay_nonfinite.*.jsonl
```

That run tripped FULL CUDA graph replay with `actual_num_tokens=162`,
`descriptor_num_tokens=168`, and non-finite padded rows. Treat this as shape
evidence, not the primary standalone fixture.

## Repro scripts

- `repro_nemotron_nano_swe_chat_window_standalone.py`
    - Offline `LLM.chat` replay.
    - Embeds the failing request and 64 recent chat requests.
    - Cleanest standalone Python path, but weaker because it does not exercise the
    OpenAI server and continuous batching.
- `repro_nemotron_nano_swe_chat_window_server.py`
    - Client replay against a running vLLM OpenAI server.
    - Uses the embedded payload from the standalone script.
- `repro_nemotron_nano_swe_chat_ring_server.py`
    - Strongest current server replay attempt.
    - Replays the real diagnostic request ring with high concurrency and repeated
    waves to preserve serving pressure.
- `repro_nemotron_nano_swe_token_width_poison_server.py`
    - Synthetic width probe using real failing prompt token IDs.
    - Use only after real chat replay attempts, and label results as synthetic.

## Current next command

```bash
cd /home/daniel/git/vllm-nemotron-vllm-repro
HF_HOME=/beegfs/huggingface \
HF_HUB_CACHE=/beegfs/huggingface/hub \
CUDAGRAPH_MODE=FULL_AND_PIECEWISE \
CHAT_REPEAT=8 \
CHAT_CONCURRENCY=512 \
CHAT_MAX_COMPLETION_TOKENS=2048 \
sbatch repro_nemotron_nano_swe_chat_ring_server.sbatch
```

## Results

- `19359`: failed before model load because the fresh worktree did not have
  compiled flash-attention extensions. Fixed the sbatch wrappers to create
  untracked symlinks to the compiled extensions before importing vLLM.
- `19360`: passed model inspection and entered model execution, then failed
  while FlashInfer tried to JIT a CUTLASS MoE kernel because `ninja` was not
  visible in the worker PATH. Fixed the sbatch wrappers to prepend
  `/home/daniel/git/vllm/.venv/bin`, which contains the `ninja` binary.
