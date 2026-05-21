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

## Latest command

```bash
cd /home/daniel/git/vllm-nemotron-vllm-repro
HF_HOME=/beegfs/huggingface \
HF_HUB_CACHE=/beegfs/huggingface/hub \
CUDAGRAPH_MODE=FULL_AND_PIECEWISE \
PAIRS='168:162,176:168,160:152,232:227,224:218' \
REPEATS=2 \
FILL_MAX_TOKENS=96 \
PROBE_MAX_TOKENS=192 \
FILL_PROMPT_LEN=1024 \
PROBE_PROMPT_LEN=0 \
sbatch repro_nemotron_nano_swe_token_width_poison_server.sbatch
```

## Results

- `19359`: failed before model load because the fresh worktree did not have
  compiled flash-attention extensions. Fixed the sbatch wrappers to create
  untracked symlinks to the compiled extensions before importing vLLM.
- `19360`: passed model inspection and entered model execution, then failed
  while FlashInfer tried to JIT a CUTLASS MoE kernel because `ninja` was not
  visible in the worker PATH. Fixed the sbatch wrappers to prepend
  `/home/daniel/git/vllm/.venv/bin`, which contains the `ninja` binary.
- `19361`: progressed through weight load, torch compile, and FlashInfer MoE JIT,
  then failed during startup warmup with
  `DeepGEMM backend is not available or outdated`. This is the known Nemotron
  startup incompatibility; the hosted workaround is `VLLM_USE_DEEP_GEMM=0`.
  Updated all repro sbatch wrappers to set and log that environment variable.
- `19362`: faithful server chat-ring replay reached vLLM health and completed
  the full captured request ring: 65 base records repeated 8 times, 520 total
  `/v1/chat/completions` requests, concurrency 512, `ignore_eos=true`,
  `max_completion_tokens=2048`, `FULL_AND_PIECEWISE`, `VLLM_USE_DEEP_GEMM=0`.
  Result was `status_counts={'ok': 520}` and `RESULT no_nonfinite_observed`.
  This means captured chat payload plus high server concurrency is not by itself
  sufficient on current vLLM `origin/main`; continue with shape-focused probes.
- `19363`: synthetic token-width poison probe reached vLLM health and exercised
  the known-suspicious padded capture widths with real failing prompt token IDs:
  `168:162`, `176:168`, `160:152`, `232:227`, and `224:218`, two repeats each,
  `/v1/completions`, `FULL_AND_PIECEWISE`, `VLLM_USE_DEEP_GEMM=0`,
  `max_model_len=131072`. Result was
  `SUMMARY status_counts={'ok': 3757, 'exception': 17}` and
  `RESULT inconclusive: non-NaN request errors occurred`. The 17 failures were
  client-side `httpx.ReadError`s in long-prompt probe waves. There was no
  `REQUEST_NONFINITE`, no `Out of range float values`, and no server-side
  worker death or NaN traceback. This did not reproduce the target bug.
- `19364`: lightweight synthetic token-width probe with the
  `VLLM_DEBUG_PADDED_INPUT_IDS=1` diagnostic enabled. Command shape was
  `PAIRS='168:162,232:227'`, `REPEATS=1`, `FILL_MAX_TOKENS=32`,
  `PROBE_MAX_TOKENS=32`, `/v1/completions`, `FULL_AND_PIECEWISE`,
  `VLLM_USE_DEEP_GEMM=0`, `max_model_len=131072`. Result was
  `SUMMARY status_counts={'ok': 786, 'exception': 3}` and
  `RESULT inconclusive: non-NaN request errors occurred`; the exceptions were
  again client-side `httpx.ReadError`s. The important finding is that the
  external vLLM-only server workload does hit `GPUModelRunner._preprocess()` with
  nonzero stale padded token IDs:

  ```text
  PADDED_INPUT_IDS_DEBUG rank=0 scheduled=166 padded=168 pad_rows=2 any_nonzero=True sample=[1429, 8030]
  PADDED_INPUT_IDS_DEBUG rank=0 scheduled=150 padded=152 pad_rows=2 any_nonzero=True sample=[1729, 4460]
  PADDED_INPUT_IDS_DEBUG rank=0 scheduled=229 padded=232 pad_rows=3 any_nonzero=True ...
  ```

  Observed padded execution shapes included `150->152`, `157->160`,
  `164/166/167->168`, `174->176`, `222->224`, and `229/231->232`, all with
  `any_nonzero=True`. This confirms the stale padded-input mechanism is live in
  a vLLM-only OpenAI-server run, but this reduced workload still did not surface
  the final JSON NaN.
