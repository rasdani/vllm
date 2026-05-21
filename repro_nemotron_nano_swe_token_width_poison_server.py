#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Raw-token width-poison replay for the Nemotron Nano SWE NaN.

Exact local setting:
- Model: nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16
- Source diagnostic:
  /beegfs/daniel/nemotron-nano-swe-step7-default-128k/chat_nan_diagnostics/1778796420266_chat_nonfinite_response_949b56786333e335.json
- Captured failing prompt token count: 12,853
- Endpoint under test: /v1/completions with prompt token IDs, not chat templating
- Target shape: wide uniform decode wave followed by a narrower uniform decode
  wave that pads into the same FULL CUDA graph descriptor, e.g. 232 -> 227.

This is intentionally not a faithful chat workload. It is a shape-focused
negative/positive control: can stale padded decode rows be made visible by
running a full descriptor width first, then a shorter active width whose padded
rows reuse the same graph-sized input buffers?
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

MODEL = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
DIAGNOSTIC_PATH = Path(
    "/beegfs/daniel/nemotron-nano-swe-step7-default-128k/"
    "chat_nan_diagnostics/1778796420266_chat_nonfinite_response_949b56786333e335.json"
)
DEFAULT_PAIRS = "232:227,224:218,216:209,208:201,200:193"


@dataclass(frozen=True)
class WidthPair:
    fill_width: int
    probe_width: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--pairs", default=DEFAULT_PAIRS)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--fill-max-tokens", type=int, default=96)
    parser.add_argument("--probe-max-tokens", type=int, default=192)
    parser.add_argument("--fill-prompt-len", type=int, default=1024)
    parser.add_argument("--probe-prompt-len", type=int, default=0)
    parser.add_argument("--request-timeout", type=float, default=1800.0)
    parser.add_argument("--between-waves-sleep", type=float, default=2.0)
    parser.add_argument("--between-pairs-sleep", type=float, default=5.0)
    return parser.parse_args()


def parse_pairs(value: str) -> list[WidthPair]:
    pairs: list[WidthPair] = []
    for chunk in value.split(","):
        if not chunk.strip():
            continue
        fill, probe = chunk.split(":", 1)
        pairs.append(WidthPair(int(fill), int(probe)))
    if not pairs:
        raise ValueError("at least one width pair is required")
    return pairs


def load_failure_prompt_ids() -> list[int]:
    with DIAGNOSTIC_PATH.open() as f:
        diagnostic = json.load(f)
    prompt_ids = diagnostic["response_payload"]["prompt_token_ids"]
    if not prompt_ids:
        raise ValueError(f"no prompt_token_ids in {DIAGNOSTIC_PATH}")
    return [int(token_id) for token_id in prompt_ids]


def make_prompt(base: list[int], length: int, salt: int) -> list[int]:
    if length <= 0:
        return list(base)
    if length <= len(base):
        # Keep the suffix because it contains the final assistant/chat boundary.
        prompt = list(base[-length:])
    else:
        repeats = math.ceil(length / len(base))
        prompt = (base * repeats)[-length:]
    # Make otherwise identical requests slightly distinct without changing
    # prompt length. Token 0 is <unk> for this tokenizer and is valid input.
    if prompt:
        prompt[0] = (prompt[0] + salt) % 1024
    return prompt


def nonfinite_paths(value: Any, path: str = "$") -> list[tuple[str, float]]:
    if isinstance(value, float):
        return [] if math.isfinite(value) else [(path, value)]
    if isinstance(value, dict):
        paths: list[tuple[str, float]] = []
        for key, item in value.items():
            paths.extend(nonfinite_paths(item, f"{path}.{key}"))
        return paths
    if isinstance(value, list):
        paths = []
        for index, item in enumerate(value):
            paths.extend(nonfinite_paths(item, f"{path}[{index}]"))
        return paths
    return []


def response_summary(response_json: dict[str, Any]) -> str:
    choices = response_json.get("choices") or []
    if not choices:
        return "choices=0"
    choice = choices[0]
    text = choice.get("text") or ""
    token_ids = choice.get("token_ids") or []
    logprobs = choice.get("logprobs") or {}
    token_logprobs = logprobs.get("token_logprobs") or []
    zero_count = sum(1 for token_id in token_ids if token_id == 0)
    return (
        f"finish={choice.get('finish_reason')} text_chars={len(text)} "
        f"tokens={len(token_ids)} zeros={zero_count} "
        f"token_logprobs={len(token_logprobs)}"
    )


async def post_completion(
    client: httpx.AsyncClient,
    prompt: list[int],
    *,
    tag: str,
    index: int,
    max_tokens: int,
) -> str:
    body = {
        "model": MODEL,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 1.0,
        "top_p": 1.0,
        "stream": False,
        "ignore_eos": True,
        "logprobs": 1,
        "return_token_ids": True,
    }
    started = time.monotonic()
    try:
        response = await client.post(
            "/completions",
            json=body,
            headers={"X-Request-Id": f"{tag}-{index}"},
        )
    except Exception as exc:
        print(
            "REQUEST_EXCEPTION "
            f"tag={tag} index={index} prompt_len={len(prompt)} "
            f"elapsed_seconds={time.monotonic() - started:.2f} "
            f"exc={type(exc).__name__}: {exc}",
            flush=True,
        )
        return "exception"

    elapsed = time.monotonic() - started
    if response.status_code >= 400:
        text = response.text[:1200]
        status = "nonfinite" if "nan" in text.lower() else "http_error"
        print(
            "REQUEST_HTTP_ERROR "
            f"tag={tag} index={index} prompt_len={len(prompt)} "
            f"status={response.status_code} elapsed_seconds={elapsed:.2f} "
            f"text={text!r}",
            flush=True,
        )
        return status

    try:
        response_json = json.loads(response.text)
    except Exception as exc:
        text = response.text[:1200]
        status = "nonfinite" if "nan" in text.lower() else "json_error"
        print(
            "REQUEST_JSON_ERROR "
            f"tag={tag} index={index} prompt_len={len(prompt)} "
            f"elapsed_seconds={elapsed:.2f} exc={type(exc).__name__}: {exc} "
            f"text={text!r}",
            flush=True,
        )
        return status

    bad_paths = nonfinite_paths(response_json)
    if bad_paths:
        print(
            "REQUEST_NONFINITE "
            f"tag={tag} index={index} prompt_len={len(prompt)} "
            f"elapsed_seconds={elapsed:.2f} paths={bad_paths[:16]} "
            f"{response_summary(response_json)}",
            flush=True,
        )
        return "nonfinite"

    print(
        "REQUEST_OK "
        f"tag={tag} index={index} prompt_len={len(prompt)} "
        f"elapsed_seconds={elapsed:.2f} {response_summary(response_json)}",
        flush=True,
    )
    return "ok"


async def run_wave(
    client: httpx.AsyncClient,
    *,
    tag: str,
    width: int,
    prompt_len: int,
    max_tokens: int,
    base_prompt: list[int],
    salt_offset: int,
) -> dict[str, int]:
    print(
        "WAVE_START "
        f"tag={tag} width={width} prompt_len={prompt_len or len(base_prompt)} "
        f"max_tokens={max_tokens}",
        flush=True,
    )
    started = time.monotonic()
    tasks = [
        post_completion(
            client,
            make_prompt(base_prompt, prompt_len, salt_offset + index),
            tag=tag,
            index=index,
            max_tokens=max_tokens,
        )
        for index in range(width)
    ]
    statuses = await asyncio.gather(*tasks)
    counts: dict[str, int] = {}
    for status in statuses:
        counts[status] = counts.get(status, 0) + 1
    print(
        "WAVE_SUMMARY "
        f"tag={tag} width={width} elapsed_seconds={time.monotonic() - started:.2f} "
        f"status_counts={counts}",
        flush=True,
    )
    return counts


async def main_async() -> int:
    args = parse_args()
    pairs = parse_pairs(args.pairs)
    base_prompt = load_failure_prompt_ids()
    probe_prompt_len = args.probe_prompt_len or len(base_prompt)
    print(
        "TOKEN_WIDTH_POISON_SETTING "
        f"base_url={args.base_url} diagnostic={DIAGNOSTIC_PATH} "
        f"base_prompt_len={len(base_prompt)} pairs={pairs} repeats={args.repeats} "
        f"fill_prompt_len={args.fill_prompt_len} probe_prompt_len={probe_prompt_len} "
        f"fill_max_tokens={args.fill_max_tokens} probe_max_tokens={args.probe_max_tokens}",
        flush=True,
    )

    status_counts: dict[str, int] = {}
    timeout = httpx.Timeout(args.request_timeout)
    limits = httpx.Limits(max_connections=max(pair.fill_width for pair in pairs) + 32)
    async with httpx.AsyncClient(
        base_url=args.base_url,
        timeout=timeout,
        limits=limits,
        trust_env=False,
    ) as client:
        for repeat in range(args.repeats):
            for pair_index, pair in enumerate(pairs):
                fill_tag = f"r{repeat}-p{pair_index}-fill{pair.fill_width}"
                probe_tag = f"r{repeat}-p{pair_index}-probe{pair.probe_width}"
                fill_counts = await run_wave(
                    client,
                    tag=fill_tag,
                    width=pair.fill_width,
                    prompt_len=args.fill_prompt_len,
                    max_tokens=args.fill_max_tokens,
                    base_prompt=base_prompt,
                    salt_offset=repeat * 100000 + pair_index * 1000,
                )
                await asyncio.sleep(args.between_waves_sleep)
                probe_counts = await run_wave(
                    client,
                    tag=probe_tag,
                    width=pair.probe_width,
                    prompt_len=probe_prompt_len,
                    max_tokens=args.probe_max_tokens,
                    base_prompt=base_prompt,
                    salt_offset=repeat * 100000 + pair_index * 1000 + 500,
                )
                for counts in (fill_counts, probe_counts):
                    for status, count in counts.items():
                        status_counts[status] = status_counts.get(status, 0) + count
                await asyncio.sleep(args.between_pairs_sleep)

    print(f"SUMMARY status_counts={status_counts}", flush=True)
    if status_counts.get("nonfinite", 0):
        print("RESULT failed: observed non-finite completion response", flush=True)
        return 2
    non_ok = sum(count for status, count in status_counts.items() if status != "ok")
    if non_ok:
        print("RESULT inconclusive: non-NaN request errors occurred", flush=True)
        return 1
    print("RESULT passed: no non-finite completion response observed", flush=True)
    return 0


def main() -> int:
    return asyncio.run(main_async())


if __name__ == "__main__":
    raise SystemExit(main())
