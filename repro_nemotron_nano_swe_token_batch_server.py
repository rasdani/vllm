#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Replay real Nemotron Nano SWE rollout prefixes as token-id completions.

This is vLLM-only: it reads saved rollout transcripts, applies the model chat
template locally, and sends the resulting real prompt token IDs to
`/v1/completions`. Using completions with `ignore_eos=true` avoids tool-parser
early exits and lets us deliberately hold a target number of concurrent decode
requests, such as the 191-request FULL CUDA graph shape that failed in training.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import time
from contextlib import suppress
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
from transformers import AutoTokenizer

import repro_nemotron_nano_swe_rollout_prefix_server as rollout_replay

MODEL = rollout_replay.MODEL
TARGET_TRACE = Path(
    "/beegfs/daniel/nemotron-nano-swe-collector-20260521-235954/"
    "vllm_nan_trace/cudagraph_replay_nonfinite.1236436.jsonl"
)


@dataclass(frozen=True)
class TokenReplayRecord:
    rollout_index: int
    example_id: Any
    turn_index: int
    prompt_token_ids: list[int]
    target_prompt_len: int | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--rollouts", type=Path, default=rollout_replay.ROLLOUT_PATH)
    parser.add_argument(
        "--tools-source", type=Path, default=rollout_replay.TOOLS_SOURCE
    )
    parser.add_argument("--target-trace", type=Path, default=TARGET_TRACE)
    parser.add_argument("--target-width", type=int, default=191)
    parser.add_argument("--replay-count", type=int, default=None)
    parser.add_argument("--max-rollouts", type=int, default=512)
    parser.add_argument("--max-candidates", type=int, default=8192)
    parser.add_argument("--min-turn-index", type=int, default=4)
    parser.add_argument("--max-turn-index", type=int, default=9999)
    parser.add_argument("--concurrency", type=int, default=191)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--timeout-s", type=float, default=1800.0)
    return parser.parse_args()


def normalize_for_template(message: dict[str, Any]) -> dict[str, Any]:
    normalized = deepcopy(message)
    for tool_call in normalized.get("tool_calls") or []:
        function = tool_call.get("function") if isinstance(tool_call, dict) else None
        if isinstance(function, dict) and isinstance(function.get("arguments"), str):
            with suppress(json.JSONDecodeError):
                function["arguments"] = json.loads(function["arguments"])
    return normalized


def token_ids_from_template(
    tokenizer: Any,
    messages: list[dict[str, Any]],
    tools: Any,
) -> list[int]:
    rendered = tokenizer.apply_chat_template(
        [normalize_for_template(message) for message in messages],
        tools=tools,
        tokenize=True,
        add_generation_prompt=True,
    )
    if hasattr(rendered, "keys") and "input_ids" in rendered:
        return list(rendered["input_ids"])
    return list(rendered)


def load_target_prompt_lengths(path: Path, target_width: int) -> list[int]:
    if not path.exists():
        return []
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            scope = row.get("nonfinite_scope") or {}
            if scope.get("actual_num_tokens") != target_width:
                continue
            input_batch = (
                row.get("trace_info", {})
                .get("model_runner_context", {})
                .get("input_batch", {})
            )
            values = input_batch.get("row_values", {}).get("num_prompt_tokens")
            if values:
                return [int(value) for value in values]
    return []


def choose_closest_records(
    candidates: list[TokenReplayRecord],
    target_lengths: list[int],
    width: int,
    replay_count: int | None,
) -> list[TokenReplayRecord]:
    if len(candidates) < width:
        raise ValueError(f"Need at least {width} candidates, got {len(candidates)}")
    limit = replay_count if replay_count is not None else width
    if limit < width:
        raise ValueError(f"replay-count must be >= target-width ({width})")
    if len(candidates) < limit:
        raise ValueError(f"Need at least {limit} candidates, got {len(candidates)}")
    if not target_lengths:
        return candidates[:limit]

    unused = set(range(len(candidates)))
    selected: list[TokenReplayRecord] = []
    for target_len in target_lengths[:width]:
        index = min(
            unused,
            key=lambda i: abs(len(candidates[i].prompt_token_ids) - target_len),
        )
        unused.remove(index)
        candidate = candidates[index]
        selected.append(
            TokenReplayRecord(
                rollout_index=candidate.rollout_index,
                example_id=candidate.example_id,
                turn_index=candidate.turn_index,
                prompt_token_ids=candidate.prompt_token_ids,
                target_prompt_len=target_len,
            )
        )
    selected_keys = {(record.rollout_index, record.turn_index) for record in selected}
    for candidate in candidates:
        if len(selected) >= limit:
            break
        key = (candidate.rollout_index, candidate.turn_index)
        if key in selected_keys:
            continue
        selected.append(candidate)
        selected_keys.add(key)
    return selected


def collect_token_records(args: argparse.Namespace) -> list[TokenReplayRecord]:
    record_args = argparse.Namespace(
        rollouts=args.rollouts,
        tools_source=args.tools_source,
        max_rollouts=args.max_rollouts,
        max_requests=args.max_candidates,
        min_turn_index=args.min_turn_index,
        max_turn_index=args.max_turn_index,
    )
    records = rollout_replay.collect_records(record_args)
    tools = [tool["function"] for tool in rollout_replay.load_tools(args.tools_source)]
    tokenizer = AutoTokenizer.from_pretrained(MODEL)

    candidates: list[TokenReplayRecord] = []
    for record in records:
        token_ids = token_ids_from_template(tokenizer, record.messages, tools)
        candidates.append(
            TokenReplayRecord(
                rollout_index=record.rollout_index,
                example_id=record.example_id,
                turn_index=record.turn_index,
                prompt_token_ids=token_ids,
                target_prompt_len=None,
            )
        )

    target_lengths = load_target_prompt_lengths(args.target_trace, args.target_width)
    selected = choose_closest_records(
        candidates, target_lengths, args.target_width, args.replay_count
    )
    lengths = [len(record.prompt_token_ids) for record in selected]
    target_matched = sum(record.target_prompt_len is not None for record in selected)
    deltas = [
        abs(length - record.target_prompt_len)
        for length, record in zip(lengths, selected)
        if record.target_prompt_len is not None
    ]
    print(
        "TOKEN_BATCH_INPUTS "
        f"candidates={len(candidates)} selected={len(selected)} "
        f"target_width={args.target_width} target_matched={target_matched} "
        f"prompt_len_min={min(lengths)} prompt_len_max={max(lengths)} "
        f"prompt_len_head={lengths[:16]} prompt_len_tail={lengths[-16:]} "
        f"target_delta_max={max(deltas) if deltas else None}",
        flush=True,
    )
    return selected


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


def make_body(record: TokenReplayRecord, args: argparse.Namespace) -> dict[str, Any]:
    return {
        "model": MODEL,
        "prompt": record.prompt_token_ids,
        "stream": False,
        "echo": False,
        "logprobs": 1,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "ignore_eos": True,
        "request_id": (f"token-batch-r{record.rollout_index}-t{record.turn_index}"),
    }


def response_summary(response_json: dict[str, Any]) -> str:
    choices = response_json.get("choices") or []
    if not choices:
        return "choices=0"
    choice = choices[0]
    text = choice.get("text") or ""
    logprobs = choice.get("logprobs") or {}
    token_logprobs = logprobs.get("token_logprobs") or []
    return (
        f"finish={choice.get('finish_reason')} text_chars={len(text)} "
        f"logprob_items={len(token_logprobs)}"
    )


async def post_completion(
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    body: dict[str, Any],
    record: TokenReplayRecord,
) -> str:
    tag = f"rollout-{record.rollout_index}-turn-{record.turn_index}"
    started = time.monotonic()
    async with semaphore:
        try:
            response = await client.post(
                "/completions",
                json=body,
                headers={"X-Request-Id": tag},
            )
        except Exception as exc:
            print(
                "TOKEN_BATCH_EXCEPTION "
                f"tag={tag} elapsed_seconds={time.monotonic() - started:.2f} "
                f"exc={type(exc).__name__}: {exc}",
                flush=True,
            )
            return "exception"

    elapsed = time.monotonic() - started
    if response.status_code >= 400:
        text = response.text[:1600]
        status = "nonfinite" if "nan" in text.lower() else "http_error"
        print(
            "TOKEN_BATCH_HTTP_ERROR "
            f"tag={tag} status={response.status_code} "
            f"elapsed_seconds={elapsed:.2f} text={text!r}",
            flush=True,
        )
        return status

    try:
        response_json = json.loads(response.text)
    except Exception as exc:
        text = response.text[:1600]
        status = "nonfinite" if "nan" in text.lower() else "json_error"
        print(
            "TOKEN_BATCH_JSON_ERROR "
            f"tag={tag} elapsed_seconds={elapsed:.2f} "
            f"exc={type(exc).__name__}: {exc} text={text!r}",
            flush=True,
        )
        return status

    bad_paths = nonfinite_paths(response_json)
    if bad_paths:
        print(
            "TOKEN_BATCH_NONFINITE "
            f"tag={tag} elapsed_seconds={elapsed:.2f} "
            f"paths={bad_paths[:16]} {response_summary(response_json)}",
            flush=True,
        )
        return "nonfinite"

    print(
        "TOKEN_BATCH_OK "
        f"tag={tag} elapsed_seconds={elapsed:.2f} "
        f"{response_summary(response_json)}",
        flush=True,
    )
    return "ok"


async def main_async() -> int:
    args = parse_args()
    records = collect_token_records(args)
    print(
        "TOKEN_BATCH_START "
        f"records={len(records)} concurrency={args.concurrency} "
        f"max_tokens={args.max_tokens} temperature={args.temperature}",
        flush=True,
    )

    limits = httpx.Limits(
        max_connections=max(args.concurrency + 32, 256),
        max_keepalive_connections=max(args.concurrency + 32, 256),
    )
    timeout = httpx.Timeout(args.timeout_s, connect=60.0)
    semaphore = asyncio.Semaphore(args.concurrency)
    async with httpx.AsyncClient(
        base_url=args.base_url,
        timeout=timeout,
        limits=limits,
        trust_env=False,
    ) as client:
        tasks = [
            post_completion(client, semaphore, make_body(record, args), record)
            for record in records
        ]

        counts: dict[str, int] = {}
        started = time.monotonic()
        for task in asyncio.as_completed(tasks):
            status = await task
            counts[status] = counts.get(status, 0) + 1
            if status == "nonfinite":
                print(
                    "TOKEN_BATCH_SUMMARY "
                    f"elapsed_seconds={time.monotonic() - started:.2f} "
                    f"status_counts={counts}",
                    flush=True,
                )
                print("TOKEN_BATCH_RESULT reproduced_nonfinite", flush=True)
                return 2

    print(
        "TOKEN_BATCH_SUMMARY "
        f"elapsed_seconds={time.monotonic() - started:.2f} "
        f"status_counts={counts}",
        flush=True,
    )
    if any(status != "ok" for status in counts):
        print("TOKEN_BATCH_RESULT inconclusive", flush=True)
        return 1
    print("TOKEN_BATCH_RESULT no_nonfinite_observed", flush=True)
    return 0


def main() -> None:
    raise SystemExit(asyncio.run(main_async()))


if __name__ == "__main__":
    main()
