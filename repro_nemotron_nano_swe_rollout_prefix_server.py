#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Replay saved Nemotron Nano SWE rollout prefixes against a vLLM server.

Exact local setting:
- Model: nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16
- Source rollouts:
  /beegfs/daniel/nemotron-nano-swe-step7-default-128k/run_default/rollouts/step_0/train_rollouts.jsonl
- Endpoint under test: /v1/chat/completions

This stays vLLM-only. It does not run prime-rl, verifiers, sandboxes, tools, or
the training loop. It reconstructs the chat request prefixes that vLLM would
have seen at each assistant turn from already-saved real rollout transcripts.
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
ROLLOUT_PATH = Path(
    "/beegfs/daniel/nemotron-nano-swe-step7-default-128k/"
    "run_default/rollouts/step_0/train_rollouts.jsonl"
)
TOOLS_SOURCE = Path(
    "/beegfs/daniel/nemotron-nano-swe-step7-default-128k/"
    "chat_nan_diagnostics/1778796420266_chat_nonfinite_response_949b56786333e335.json"
)


@dataclass(frozen=True)
class ReplayRecord:
    rollout_index: int
    example_id: Any
    turn_index: int
    messages: list[dict[str, Any]]
    sampling_args: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--rollouts", type=Path, default=ROLLOUT_PATH)
    parser.add_argument("--tools-source", type=Path, default=TOOLS_SOURCE)
    parser.add_argument("--max-rollouts", type=int, default=512)
    parser.add_argument("--max-requests", type=int, default=2048)
    parser.add_argument("--min-turn-index", type=int, default=0)
    parser.add_argument("--max-turn-index", type=int, default=9999)
    parser.add_argument("--concurrency", type=int, default=512)
    parser.add_argument("--max-completion-tokens", type=int, default=256)
    parser.add_argument("--allowed-token-ids", default="")
    parser.add_argument("--force-ignore-eos", action="store_true")
    parser.add_argument("--timeout-s", type=float, default=1800.0)
    return parser.parse_args()


def parse_allowed_token_ids(value: str) -> list[int] | None:
    token_ids = [int(item) for item in value.split(",") if item.strip()]
    return token_ids or None


def load_tools(path: Path) -> list[dict[str, Any]]:
    with path.open() as f:
        diagnostic = json.load(f)
    tools = diagnostic.get("request", {}).get("body", {}).get("tools") or []
    if not tools:
        raise ValueError(f"no tools found in {path}")
    return tools


def normalize_message(message: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(message)
    tool_calls = normalized.get("tool_calls")
    if isinstance(tool_calls, list):
        parsed_tool_calls = []
        for tool_call in tool_calls:
            if isinstance(tool_call, str):
                parsed_tool_calls.append(json.loads(tool_call))
            else:
                parsed_tool_calls.append(tool_call)
        normalized["tool_calls"] = parsed_tool_calls
    return normalized


def collect_records(args: argparse.Namespace) -> list[ReplayRecord]:
    by_turn: dict[int, list[ReplayRecord]] = {}
    rollout_count = 0
    assistant_turn_count = 0

    with args.rollouts.open() as f:
        for rollout_index, line in enumerate(f):
            if rollout_index >= args.max_rollouts:
                break
            rollout = json.loads(line)
            rollout_count += 1
            prompt = [normalize_message(msg) for msg in rollout.get("prompt") or []]
            completion = [
                normalize_message(msg) for msg in rollout.get("completion") or []
            ]
            sampling_args = dict(rollout.get("sampling_args") or {})

            prefix = list(prompt)
            turn_index = 0
            for message in completion:
                if message.get("role") == "assistant":
                    assistant_turn_count += 1
                    if args.min_turn_index <= turn_index <= args.max_turn_index:
                        by_turn.setdefault(turn_index, []).append(
                            ReplayRecord(
                                rollout_index=rollout_index,
                                example_id=rollout.get("example_id"),
                                turn_index=turn_index,
                                messages=list(prefix),
                                sampling_args=sampling_args,
                            )
                        )
                    turn_index += 1
                prefix.append(message)

    records: list[ReplayRecord] = []
    for turn_index in sorted(by_turn):
        for record in by_turn[turn_index]:
            records.append(record)
            if len(records) >= args.max_requests:
                break
        if len(records) >= args.max_requests:
            break

    print(
        "ROLLOUT_REPLAY_INPUTS "
        f"path={args.rollouts} rollouts={rollout_count} "
        f"assistant_turns={assistant_turn_count} selected={len(records)} "
        f"turns={sorted(by_turn)[:8]}",
        flush=True,
    )
    if not records:
        raise ValueError("no replay records selected")
    return records


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


def make_body(
    record: ReplayRecord,
    *,
    tools: list[dict[str, Any]],
    max_completion_tokens: int,
    allowed_token_ids: list[int] | None,
    force_ignore_eos: bool,
) -> dict[str, Any]:
    extra_body = dict(record.sampling_args.get("extra_body") or {})
    body = {
        "model": MODEL,
        "messages": record.messages,
        "tools": tools,
        "tool_choice": "auto",
        "stream": False,
        "logprobs": True,
        "return_token_ids": True,
        "temperature": record.sampling_args.get("temperature", 1.0),
        "top_p": record.sampling_args.get("top_p", 1.0),
        "top_k": extra_body.get("top_k", -1),
        "min_p": extra_body.get("min_p", 0.0),
        "cache_salt": extra_body.get("cache_salt", "0"),
        "max_completion_tokens": max_completion_tokens,
        "request_id": (
            f"rollout-prefix-{record.rollout_index}-turn-{record.turn_index}"
        ),
    }
    if allowed_token_ids is not None:
        body["allowed_token_ids"] = allowed_token_ids
    if force_ignore_eos:
        body["ignore_eos"] = True
    return body


def response_summary(response_json: dict[str, Any]) -> str:
    choices = response_json.get("choices") or []
    if not choices:
        return "choices=0"
    choice = choices[0]
    message = choice.get("message") or {}
    content = message.get("content") or ""
    logprobs = choice.get("logprobs") or {}
    content_logprobs = logprobs.get("content") or []
    token_ids = []
    for item in content_logprobs:
        token_id = item.get("token_id") if isinstance(item, dict) else None
        if token_id is not None:
            token_ids.append(token_id)
    zero_count = sum(1 for token_id in token_ids if token_id == 0)
    return (
        f"finish={choice.get('finish_reason')} content_chars={len(content)} "
        f"logprob_items={len(content_logprobs)} zeros={zero_count}"
    )


async def post_chat(
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    body: dict[str, Any],
    record: ReplayRecord,
) -> str:
    started = time.monotonic()
    tag = f"rollout-{record.rollout_index}-turn-{record.turn_index}"
    body_len = len(json.dumps(body, separators=(",", ":")))
    async with semaphore:
        try:
            response = await client.post(
                "/chat/completions",
                json=body,
                headers={"X-Request-Id": tag},
            )
        except Exception as exc:
            print(
                "REQUEST_EXCEPTION "
                f"tag={tag} example_id={record.example_id} body_len={body_len} "
                f"elapsed_seconds={time.monotonic() - started:.2f} "
                f"exc={type(exc).__name__}: {exc}",
                flush=True,
            )
            return "exception"

    elapsed = time.monotonic() - started
    if response.status_code >= 400:
        text = response.text[:1600]
        status = "nonfinite" if "nan" in text.lower() else "http_error"
        print(
            "REQUEST_HTTP_ERROR "
            f"tag={tag} example_id={record.example_id} body_len={body_len} "
            f"status={response.status_code} elapsed_seconds={elapsed:.2f} "
            f"text={text!r}",
            flush=True,
        )
        return status

    try:
        response_json = json.loads(response.text)
    except Exception as exc:
        text = response.text[:1600]
        status = "nonfinite" if "nan" in text.lower() else "json_error"
        print(
            "REQUEST_JSON_ERROR "
            f"tag={tag} example_id={record.example_id} body_len={body_len} "
            f"elapsed_seconds={elapsed:.2f} exc={type(exc).__name__}: {exc} "
            f"text={text!r}",
            flush=True,
        )
        return status

    bad_paths = nonfinite_paths(response_json)
    if bad_paths:
        print(
            "REQUEST_NONFINITE "
            f"tag={tag} example_id={record.example_id} body_len={body_len} "
            f"elapsed_seconds={elapsed:.2f} paths={bad_paths[:16]} "
            f"{response_summary(response_json)}",
            flush=True,
        )
        return "nonfinite"

    print(
        "REQUEST_OK "
        f"tag={tag} example_id={record.example_id} body_len={body_len} "
        f"elapsed_seconds={elapsed:.2f} {response_summary(response_json)}",
        flush=True,
    )
    return "ok"


async def main_async() -> int:
    args = parse_args()
    tools = load_tools(args.tools_source)
    records = collect_records(args)
    allowed_token_ids = parse_allowed_token_ids(args.allowed_token_ids)
    print(
        "ROLLOUT_REPLAY_START "
        f"records={len(records)} concurrency={args.concurrency} "
        f"max_completion_tokens={args.max_completion_tokens} "
        f"allowed_token_ids={allowed_token_ids} "
        f"force_ignore_eos={args.force_ignore_eos}",
        flush=True,
    )
    for index, record in enumerate(records[:8]):
        print(
            "ROLLOUT_REPLAY_SOURCE "
            f"index={index} rollout={record.rollout_index} "
            f"example_id={record.example_id} turn={record.turn_index} "
            f"messages={len(record.messages)}",
            flush=True,
        )

    limits = httpx.Limits(
        max_connections=max(args.concurrency + 32, 1024),
        max_keepalive_connections=max(args.concurrency + 32, 1024),
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
            post_chat(
                client,
                semaphore,
                make_body(
                    record,
                    tools=tools,
                    max_completion_tokens=args.max_completion_tokens,
                    allowed_token_ids=allowed_token_ids,
                    force_ignore_eos=args.force_ignore_eos,
                ),
                record,
            )
            for record in records
        ]

        counts: dict[str, int] = {}
        started = time.monotonic()
        for task in asyncio.as_completed(tasks):
            status = await task
            counts[status] = counts.get(status, 0) + 1
            if status == "nonfinite":
                print(
                    f"SUMMARY elapsed_seconds={time.monotonic() - started:.2f} status_counts={counts}",
                    flush=True,
                )
                print(
                    "RESULT reproduced: non-finite rollout-prefix response observed",
                    flush=True,
                )
                return 2

    print(
        f"SUMMARY elapsed_seconds={time.monotonic() - started:.2f} status_counts={counts}",
        flush=True,
    )
    if any(status != "ok" for status in counts):
        print("RESULT inconclusive: non-NaN request errors occurred", flush=True)
        return 1
    print("RESULT no_nonfinite_observed", flush=True)
    return 0


def main() -> None:
    raise SystemExit(asyncio.run(main_async()))


if __name__ == "__main__":
    main()
