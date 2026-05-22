#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Offline vLLM-only replay for the Nemotron Nano SWE NaN investigation.

Exact default setting:
- model: nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16
- real rollout data:
  /beegfs/daniel/nemotron-nano-swe-step7-default-128k/run_default/rollouts/step_0/train_rollouts.jsonl
- real LoRA adapter:
  /beegfs/daniel/nemotron-nano-swe-step7-default-128k/run_default/broadcasts/step_1
- replay: 191 rollout-derived prompt-token batches selected to match the
  failing training trace width, max_tokens=512, logprobs=1, ignore_eos=true
- vLLM path: LLM.generate(), no OpenAI server, no prime-rl, no sandboxes
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import repro_nemotron_nano_swe_token_batch_server as token_batch
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt
from vllm.lora.request import LoRARequest

MODEL = token_batch.MODEL
LORA_PATH = Path(
    "/beegfs/daniel/nemotron-nano-swe-step7-default-128k/run_default/broadcasts/step_1"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL)
    parser.add_argument(
        "--rollouts", type=Path, default=token_batch.rollout_replay.ROLLOUT_PATH
    )
    parser.add_argument(
        "--tools-source",
        type=Path,
        default=token_batch.rollout_replay.TOOLS_SOURCE,
    )
    parser.add_argument("--target-trace", type=Path, default=token_batch.TARGET_TRACE)
    parser.add_argument("--target-width", type=int, default=191)
    parser.add_argument("--replay-count", type=int, default=191)
    parser.add_argument("--max-rollouts", type=int, default=512)
    parser.add_argument("--max-candidates", type=int, default=8192)
    parser.add_argument("--min-turn-index", type=int, default=4)
    parser.add_argument("--max-turn-index", type=int, default=9999)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=131072)
    parser.add_argument("--max-num-seqs", type=int, default=256)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.95)
    parser.add_argument("--cudagraph-mode", default="FULL_AND_PIECEWISE")
    parser.add_argument("--lora-name", default="nemotron-step1-lora")
    parser.add_argument("--lora-path", type=Path, default=LORA_PATH)
    return parser.parse_args()


def is_nonfinite(value: Any) -> list[tuple[str, float]]:
    def walk(item: Any, path: str, seen: set[int]) -> list[tuple[str, float]]:
        if isinstance(item, float):
            return [] if math.isfinite(item) else [(path, item)]
        if isinstance(item, (str, bytes, int, bool, type(None))):
            return []
        item_id = id(item)
        if item_id in seen:
            return []
        seen.add(item_id)
        if isinstance(item, dict):
            paths: list[tuple[str, float]] = []
            for key, value in item.items():
                paths.extend(walk(value, f"{path}.{key}", seen))
            return paths
        if isinstance(item, (list, tuple)):
            paths = []
            for index, value in enumerate(item):
                paths.extend(walk(value, f"{path}[{index}]", seen))
            return paths
        if hasattr(item, "__dict__"):
            return walk(vars(item), path, seen)
        return []

    return walk(value, "$", set())


def main() -> int:
    args = parse_args()
    records = token_batch.collect_token_records(args)
    prompts = [
        TokensPrompt(prompt_token_ids=record.prompt_token_ids) for record in records
    ]
    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        logprobs=1,
        ignore_eos=True,
    )
    lora_request = LoRARequest(
        args.lora_name,
        1,
        str(args.lora_path),
    )

    print(
        "OFFLINE_BATCH_START "
        f"records={len(records)} model={args.model} "
        f"tp={args.tensor_parallel_size} max_model_len={args.max_model_len} "
        f"max_num_seqs={args.max_num_seqs} lora_path={args.lora_path}",
        flush=True,
    )
    started = time.monotonic()
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype="bfloat16",
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_chunked_prefill=True,
        enable_prefix_caching=False,
        enable_lora=True,
        max_loras=12,
        max_cpu_loras=100,
        compilation_config={"cudagraph_mode": args.cudagraph_mode},
    )
    outputs = llm.generate(
        prompts,
        sampling_params,
        use_tqdm=True,
        lora_request=lora_request,
    )

    counts = {"ok": 0, "nonfinite": 0}
    for index, output in enumerate(outputs):
        bad_paths = is_nonfinite(output)
        if bad_paths:
            counts["nonfinite"] += 1
            print(
                "OFFLINE_BATCH_NONFINITE "
                f"index={index} request_id={output.request_id} "
                f"paths={bad_paths[:16]}",
                flush=True,
            )
        else:
            counts["ok"] += 1

    print(
        "OFFLINE_BATCH_SUMMARY "
        f"elapsed_seconds={time.monotonic() - started:.2f} "
        f"status_counts={json.dumps(counts, sort_keys=True)}",
        flush=True,
    )
    if counts["nonfinite"]:
        print("OFFLINE_BATCH_RESULT reproduced_nonfinite", flush=True)
        return 2
    print("OFFLINE_BATCH_RESULT no_nonfinite_observed", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
