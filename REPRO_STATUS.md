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
- `repro_nemotron_nano_swe_rollout_prefix_server.py`
    - Rebuilds chat requests from saved rollout transcripts.
    - Uses no prime-rl runtime, verifiers, tools, or sandboxes; it only reads
      persisted rollout JSONL and sends reconstructed `/v1/chat/completions`
      requests to vLLM.
- `repro_nemotron_nano_swe_token_batch_server.py`
    - Rebuilds real rollout prefixes, renders them through the model chat
      template locally, and sends the resulting prompt token IDs to
      `/v1/completions`.
    - This avoids chat tool-parser early exits and deliberately holds the
      suspicious `191 -> 192` one-token FULL CUDA graph decode shape.

## Latest command

```bash
cd /home/daniel/git/vllm-nemotron-vllm-repro
HF_HOME=/beegfs/huggingface \
HF_HUB_CACHE=/beegfs/huggingface/hub \
CUDAGRAPH_MODE=FULL_AND_PIECEWISE \
TARGET_WIDTH=191 \
MAX_CANDIDATES=8192 \
TOKEN_CONCURRENCY=191 \
TOKEN_MAX_TOKENS=512 \
VLLM_REPRO_SHAPE_TRACE_LIMIT=30000 \
sbatch repro_nemotron_nano_swe_token_batch_server.sbatch
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
- `19367`: faithful multi-diagnostic chat replay over all four captured
  `chat_nan_diagnostics/*chat_nonfinite_response*.json` files. This replay sent
  260 real captured chat bodies, concurrency 256, preserved `ignore_eos=false`
  from the bodies, used `max_completion_tokens=4096`, `FULL_AND_PIECEWISE`, and
  `VLLM_USE_DEEP_GEMM=0`. Result was `status_counts={'ok': 260}` and
  `RESULT no_nonfinite_observed`. The responses included 18 generations with at
  least 2048 logprob items and 5 length-capped 4096-token generations, but none
  sampled token id `0`.
- `19368`: same four-diagnostic chat replay, but with `allowed_token_ids=[0]`,
  `ignore_eos=true`, and `max_completion_tokens=64` to force the `<unk>` token
  path seen in every captured non-finite response. Result was
  `status_counts={'ok': 260}` and `RESULT no_nonfinite_observed`. This makes
  token id `0` look like a symptom or necessary downstream ingredient, not by
  itself sufficient to create the NaN from a fresh vLLM server.
- `19372`: first rollout-prefix replay attempt against the saved rollout corpus.
  It selected 2048 reconstructed assistant-turn prefixes from 512 saved
  rollouts, but the saved rollout `tool_calls` used a compact shape
  (`id`/`name`/`arguments`) while `/v1/chat/completions` expects the OpenAI
  tool-call shape (`id`/`type=function`/`function.name`/`function.arguments`).
  Result was `status_counts={'http_error': 2046, 'exception': 2}` with request
  validation errors, so this run did not exercise the target NaN path. Fixed by
  normalizing compact rollout tool calls before sending them to vLLM.
- `19374`: fixed rollout-prefix replay against the same saved rollout corpus.
  Command shape: 512 saved rollouts, 2048 selected assistant-turn prefixes,
  `MIN_TURN_INDEX=4`, `CHAT_CONCURRENCY=512`, `CHAT_MAX_COMPLETION_TOKENS=512`,
  `/v1/chat/completions`, `FULL_AND_PIECEWISE`, `VLLM_USE_DEEP_GEMM=0`,
  `max_model_len=131072`, and HF cache under `/beegfs/huggingface`. Startup was
  slow because FlashInfer fused-MoE kernels were JIT compiled during Mamba warmup,
  then the server became healthy at attempt 87. Result was
  `SUMMARY elapsed_seconds=103.24 status_counts={'ok': 2048}` and
  `RESULT no_nonfinite_observed`. This is the strongest current vLLM-only
  negative control: saved real rollout prefixes plus high server concurrency are
  still not sufficient to reproduce the hosted-training NaN outside the original
  training/server state.
- `19378`: same rollout-prefix replay as `19374`, but with the shape trace
  instrumentation enabled in the scheduler and GPU model runner. Result was
  still `status_counts={'ok': 2048}` and `RESULT no_nonfinite_observed`.
  Shape comparison against the failing training trace showed this replay did hit
  the padded-only `162 -> 168` FULL one-token decode shape once, but it did not
  hit the `191 -> 192` FULL shape where the training trace had real-row
  non-finite outputs. This ruled out the simpler "any padded FULL decode row is
  enough" explanation.
- `19380`: token-ID `/v1/completions` replay against the same saved rollout
  corpus. It selected 191 real rollout prefixes closest to the prompt-length
  distribution in
  `/beegfs/daniel/nemotron-nano-swe-collector-20260521-235954/vllm_nan_trace/cudagraph_replay_nonfinite.1236436.jsonl`,
  then sent all 191 requests concurrently with `ignore_eos=true`,
  `logprobs=1`, `max_tokens=512`, `FULL_AND_PIECEWISE`, `VLLM_USE_DEEP_GEMM=0`,
  and `max_model_len=131072`. Result was
  `TOKEN_BATCH_SUMMARY elapsed_seconds=23.22 status_counts={'ok': 191}` and
  `TOKEN_BATCH_RESULT no_nonfinite_observed`.

  The shape trace confirmed the intended target was exercised:

  ```text
  rows=746
  modes={'FULL': 511, 'NONE': 235}
  top shape: 277 x (191 actual tokens, 192 padded tokens, 191 reqs,
                    max scheduled tokens 1, FULL)
  also hit: 2 x 162 -> 168 FULL, 5 x 150 -> 152 FULL
  ```

  This is now the strongest negative control: real rollout-derived prompt token
  IDs, exact `191 -> 192` FULL decode shape, one padded row, and the same broad
  prompt-length distribution are still not sufficient on a fresh standalone vLLM
  server.
- `19383`: token-ID `/v1/completions` replay with request churn. The script kept
  the 191 target-matched real rollout prefixes first, then appended more real
  rollout-prefix token requests for a total of 2048 requests at concurrency 512.
  Result was `TOKEN_BATCH_SUMMARY elapsed_seconds=130.47
  status_counts={'ok': 2048}` and `TOKEN_BATCH_RESULT no_nonfinite_observed`.
  The shape trace saw 2742 GPU model-runner batches:

  ```text
  modes={'NONE': 1701, 'FULL': 1041}
  top shape: 351 x 512 -> 512 FULL
  target hits: 1 x 191 -> 192 FULL, 1 x 162 -> 168 FULL,
               2 x 157 -> 160 FULL, 1 x 227 -> 232 FULL
  scheduler waiting_count was nonzero in many mixed prefill/decode rows.
  ```

  This added training-like queue pressure and mixed prefill/decode churn, but it
  still did not reproduce the NaN.
- `19387`: same churn replay as `19383`, but with the failing deployment's LoRA
  server mode enabled: `--enable-lora --max-loras 12 --max-cpu-loras 100`.
  This did activate the relevant model path:

  ```text
  MoE model detected. Using fused MoE LoRA implementation.
  ```

  Result was `TOKEN_BATCH_SUMMARY elapsed_seconds=187.30
  status_counts={'ok': 2048}` and `TOKEN_BATCH_RESULT no_nonfinite_observed`.
  Shape trace saw 2551 GPU model-runner batches:

  ```text
  modes={'NONE': 1741, 'FULL': 810}
  top shapes: 191 x 512 -> 512 FULL, 62 x 511 -> 512 FULL
  target hits: 1 x 191 -> 192 FULL, 4 x 227 -> 232 FULL,
               1 x 150 -> 152 FULL
  ```

  This rules out the simple "fresh server plus LoRA-enabled Nemotron MoE path
  plus real tokenized rollout churn" hypothesis on the current vLLM main-based
  repro branch.
- `19392`: runtime LoRA replay using the actual adapter emitted by the training
  repro:
  `/beegfs/daniel/nemotron-nano-swe-step7-default-128k/run_default/broadcasts/step_1`.
  The client loaded the adapter once through `/v1/load_lora_adapter`, then sent
  2048 real rollout-prefix token requests at concurrency 512 using the adapter
  model alias `nemotron-step1-lora`. Result was
  `TOKEN_BATCH_SUMMARY elapsed_seconds=226.49 status_counts={'ok': 2048}` and
  `TOKEN_BATCH_RESULT no_nonfinite_observed`. This run exercised the real LoRA
  adapter path, but did not hit the exact `191 -> 192` target shape; observed
  LoRA FULL shapes included `226 -> 232`, `157 -> 160`, and many `512 -> 512`
  batches.
- `19393`: controlled runtime LoRA replay using the same actual adapter, but
  restricted to 191 selected real rollout-prefix token requests at concurrency
  191 to force the suspicious target width. The adapter loaded once through
  `/v1/load_lora_adapter`. Result was
  `TOKEN_BATCH_SUMMARY elapsed_seconds=39.38 status_counts={'ok': 191}` and
  `TOKEN_BATCH_RESULT no_nonfinite_observed`. The shape trace confirmed the
  target was exercised:

  ```text
  280 x 191 actual tokens, 192 padded tokens, 191 requests,
        max scheduled tokens 1, FULL, has_lora=true
  ```

  This is the strongest current vLLM-main negative control: fresh standalone
  vLLM server, actual Nemotron Nano adapter, real rollout-derived token IDs,
  exact `191 -> 192` FULL decode shape, and LoRA active in the batch descriptor
  are still not sufficient to reproduce the JSON NaN.
- `19394`: stateful runtime-LoRA variant of `19393`. It loaded the same adapter
  four times under the same adapter name before replaying the same 191
  target-width requests. This approximates repeated adapter reload state without
  involving prime-rl. The server registered both the base model and
  `nemotron-step1-lora`, accepted all four load cycles, and eventually reached
  the exact target shape:

  ```text
  14 x 191 actual tokens, 192 padded tokens, 191 requests,
       max scheduled tokens 1, FULL, has_lora=true
  ```

  There was no NaN, HTTP 400, or worker-side non-finite error. The run is still
  not a clean negative control because all 191 client requests hit the 1800s
  `httpx.ReadTimeout`, so the script reported
  `TOKEN_BATCH_SUMMARY elapsed_seconds=1800.25 status_counts={'exception': 191}`
  and `TOKEN_BATCH_RESULT inconclusive`. The important new signal is that
  repeated same-name runtime adapter loads caused a large throughput collapse
  compared with `19393` while still not surfacing the target JSON NaN.

## Current interpretation

- The previously suspected padded-width shape is real and easy to hit in a
  vLLM-only server process.
- Shape alone is not sufficient. `19380` hit the exact `191 -> 192` FULL decode
  shape hundreds of times and stayed finite.
- Generic server queue pressure is not sufficient. `19383` added 2048 real
  tokenized rollout-prefix requests at concurrency 512 and stayed finite.
- Enabling the deployment's LoRA server mode is not sufficient on current vLLM
  main. `19387` reached the fused MoE LoRA implementation and stayed finite.
- Loading and using the actual adapter once is not sufficient. `19393` reached
  the exact `191 -> 192` FULL LoRA decode shape 280 times and stayed finite.
- Re-loading the same actual adapter several times is still not sufficient to
  surface the target NaN on current vLLM main. `19394` reached the exact shape
  only 14 times before all client requests timed out, so treat it as an
  inconclusive performance-path signal, not as proof of correctness.
- The remaining high-signal differences from the failing training deployment are
  version and stateful runtime path:
    - failing deployment logs show vLLM `0.20.2`, while this standalone branch is
      current vLLM main (`0.21.1rc1.dev13+g6147c7022...`);
    - failing deployment used the prime-rl filesystem weight-update worker
      extension;
    - the real training server may have seen weight update / reload state before
      the first NaN, while all standalone runs above use a fresh static model.
