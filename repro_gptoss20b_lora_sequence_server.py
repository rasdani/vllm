#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Replay real GPT-OSS LoRA rollout prefixes against a standalone vLLM server.

Exact local setting:
- Model/tokenizer: unsloth/gpt-oss-20b-BF16
- Served base alias: unsloth/gpt-oss-20b-BF16
- Runtime LoRA request alias: openai/gpt-oss-20b
- Source run:
  /beegfs/daniel/gptoss20b-ptft-lora-nan-20260514-013919/run_default
- Default adapter sequence:
  broadcasts/step_1 -> broadcasts/step_2 -> broadcasts/step_3
- Default rollout replay source:
  rollouts/step_3/train_rollouts.jsonl

This is intentionally vLLM-only. It does not run prime-rl, verifiers, sandboxes,
tools, or the training loop. It reconstructs the chat request prefixes from
already-saved real rollout transcripts, loads real LoRA adapter checkpoints into
the vLLM server, then looks for non-finite values surfacing in OpenAI responses.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
from transformers import AutoTokenizer

MODEL = "unsloth/gpt-oss-20b-BF16"
LORA_NAME = "openai/gpt-oss-20b"
RUN_DIR = Path("/beegfs/daniel/gptoss20b-ptft-lora-nan-20260514-013919/run_default")
ROLLOUT_PATH = RUN_DIR / "rollouts/step_3/train_rollouts.jsonl"
DEFAULT_LORA_PATHS = [
    RUN_DIR / "broadcasts/step_1",
    RUN_DIR / "broadcasts/step_2",
    RUN_DIR / "broadcasts/step_3",
]


@dataclass(frozen=True)
class ReplayRecord:
    rollout_index: int
    example_id: Any
    turn_index: int
    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]]
    sampling_args: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--rollouts", type=Path, default=ROLLOUT_PATH)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--lora-name", default=LORA_NAME)
    parser.add_argument(
        "--lora-path",
        action="append",
        type=Path,
        dest="lora_paths",
        help="LoRA checkpoint path. Repeat to replay an update sequence.",
    )
    parser.add_argument("--max-rollouts", type=int, default=512)
    parser.add_argument("--max-requests", type=int, default=512)
    parser.add_argument("--min-turn-index", type=int, default=0)
    parser.add_argument("--max-turn-index", type=int, default=9999)
    parser.add_argument("--concurrency", type=int, default=128)
    parser.add_argument("--max-completion-tokens", type=int, default=192)
    parser.add_argument("--timeout-s", type=float, default=1800.0)
    parser.add_argument("--ignore-eos", action="store_true")
    parser.add_argument(
        "--endpoint",
        choices=("chat", "completions"),
        default="chat",
        help="Use chat/completions, or render chat locally and hit completions.",
    )
    parser.add_argument(
        "--repeat-to",
        type=int,
        default=0,
        help="Repeat selected records until this many requests are queued.",
    )
    parser.add_argument("--stop-on-nonfinite", action="store_true")
    return parser.parse_args()


def fill_missing_descriptions(schema: Any) -> None:
    if not isinstance(schema, dict):
        return
    if (
        "description" not in schema
        and {"type", "anyOf", "oneOf", "allOf", "properties"} & schema.keys()
    ):
        schema["description"] = ""
    for key in ("properties",):
        values = schema.get(key)
        if isinstance(values, dict):
            for value in values.values():
                fill_missing_descriptions(value)
    for key in ("anyOf", "oneOf", "allOf"):
        values = schema.get(key)
        if isinstance(values, list):
            for value in values:
                fill_missing_descriptions(value)
    fill_missing_descriptions(schema.get("items"))


def normalize_tool(tool: dict[str, Any]) -> dict[str, Any]:
    if tool.get("type") == "function":
        normalized = deepcopy(tool)
    else:
        normalized = {"type": "function", "function": deepcopy(tool)}
    function = normalized["function"]
    if function.get("strict") is None:
        function.pop("strict", None)
    fill_missing_descriptions(function.get("parameters"))
    return normalized


def normalize_message(message: dict[str, Any]) -> dict[str, Any]:
    role = message.get("role")
    normalized: dict[str, Any] = {"role": role}
    if "content" in message:
        normalized["content"] = message.get("content")
    if role == "tool" and message.get("tool_call_id"):
        normalized["tool_call_id"] = message["tool_call_id"]

    tool_calls = message.get("tool_calls")
    if isinstance(tool_calls, list):
        parsed_tool_calls = []
        for tool_call in tool_calls:
            if isinstance(tool_call, str):
                tool_call = json.loads(tool_call)
            if (
                isinstance(tool_call, dict)
                and "function" not in tool_call
                and "name" in tool_call
                and "arguments" in tool_call
            ):
                tool_call = {
                    "id": tool_call.get("id"),
                    "type": "function",
                    "function": {
                        "name": tool_call["name"],
                        "arguments": tool_call["arguments"],
                    },
                }
            if not isinstance(tool_call, dict):
                raise TypeError(f"unsupported tool call shape: {tool_call!r}")
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
            if not line.strip():
                continue
            rollout = json.loads(line)
            rollout_count += 1
            prompt = [normalize_message(msg) for msg in rollout.get("prompt") or []]
            completion = [
                normalize_message(msg) for msg in rollout.get("completion") or []
            ]
            tools = [normalize_tool(tool) for tool in rollout.get("tool_defs") or []]
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
                                tools=tools,
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
        "GPTOSS_REPLAY_INPUTS "
        f"path={args.rollouts} rollouts={rollout_count} "
        f"assistant_turns={assistant_turn_count} selected={len(records)} "
        f"turns={sorted(by_turn)[:8]}",
        flush=True,
    )
    if not records:
        raise ValueError("no replay records selected")
    return records


def repeat_records(records: list[ReplayRecord], repeat_to: int) -> list[ReplayRecord]:
    if repeat_to <= len(records):
        return records
    repeated: list[ReplayRecord] = []
    while len(repeated) < repeat_to:
        repeated.extend(records[: repeat_to - len(repeated)])
    return repeated


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


def prompt_token_ids(
    tokenizer: Any,
    record: ReplayRecord,
) -> list[int]:
    extra_body = dict(record.sampling_args.get("extra_body") or {})
    rendered = tokenizer.apply_chat_template(
        record.messages,
        tools=record.tools,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        **extra_body.get("chat_template_kwargs", {"enable_thinking": False}),
    )
    input_ids = (
        rendered["input_ids"]
        if hasattr(rendered, "keys") and "input_ids" in rendered
        else rendered
    )
    if input_ids and isinstance(input_ids[0], list):
        input_ids = input_ids[0]
    return list(input_ids)


def make_chat_body(
    record: ReplayRecord,
    args: argparse.Namespace,
    request_index: int,
) -> dict[str, Any]:
    sampling_args = record.sampling_args
    extra_body = dict(sampling_args.get("extra_body") or {})
    body: dict[str, Any] = {
        "model": args.lora_name,
        "messages": record.messages,
        "tools": record.tools,
        "tool_choice": extra_body.get("tool_choice", "required"),
        "stream": False,
        "logprobs": bool(sampling_args.get("logprobs", True)),
        "temperature": sampling_args.get("temperature", 1.0),
        "top_p": sampling_args.get("top_p", 1.0),
        "max_completion_tokens": args.max_completion_tokens,
        "return_token_ids": True,
        "top_k": extra_body.get("top_k", -1),
        "min_p": extra_body.get("min_p", 0.0),
        "chat_template_kwargs": extra_body.get(
            "chat_template_kwargs", {"enable_thinking": False}
        ),
        "cache_salt": extra_body.get("cache_salt", "1"),
        "request_id": (
            f"gptoss-{request_index}-rollout-{record.rollout_index}"
            f"-turn-{record.turn_index}"
        ),
    }
    if args.ignore_eos:
        body["ignore_eos"] = True
    return body


def make_completion_body(
    tokenizer: Any,
    record: ReplayRecord,
    args: argparse.Namespace,
    request_index: int,
) -> dict[str, Any]:
    sampling_args = record.sampling_args
    body: dict[str, Any] = {
        "model": args.lora_name,
        "prompt": prompt_token_ids(tokenizer, record),
        "stream": False,
        "echo": False,
        "logprobs": 1 if sampling_args.get("logprobs", True) else None,
        "temperature": sampling_args.get("temperature", 1.0),
        "top_p": sampling_args.get("top_p", 1.0),
        "max_tokens": args.max_completion_tokens,
        "ignore_eos": args.ignore_eos,
        "request_id": (
            f"gptoss-token-{request_index}-rollout-{record.rollout_index}"
            f"-turn-{record.turn_index}"
        ),
    }
    return body


def chat_response_summary(response_json: dict[str, Any]) -> str:
    choices = response_json.get("choices") or []
    if not choices:
        return "choices=0"
    choice = choices[0]
    message = choice.get("message") or {}
    content = message.get("content") or ""
    tool_calls = message.get("tool_calls") or []
    logprobs = choice.get("logprobs") or {}
    content_logprobs = logprobs.get("content") or []
    token_ids = []
    for item in content_logprobs:
        if isinstance(item, dict) and item.get("token_id") is not None:
            token_ids.append(item["token_id"])
    return (
        f"finish={choice.get('finish_reason')} content_chars={len(content)} "
        f"tool_calls={len(tool_calls)} logprob_items={len(content_logprobs)} "
        f"zero_token_ids={sum(1 for token_id in token_ids if token_id == 0)}"
    )


def completion_response_summary(response_json: dict[str, Any]) -> str:
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


async def load_lora_sequence(
    client: httpx.AsyncClient,
    args: argparse.Namespace,
) -> None:
    lora_paths = args.lora_paths or DEFAULT_LORA_PATHS
    for index, path in enumerate(lora_paths):
        body: dict[str, Any] = {
            "lora_name": args.lora_name,
            "lora_path": str(path),
        }
        if index > 0:
            body["load_inplace"] = True
        response = await client.post("/load_lora_adapter", json=body)
        if response.status_code >= 400:
            raise RuntimeError(
                "LoRA load failed "
                f"index={index + 1}/{len(lora_paths)} path={path} "
                f"status={response.status_code} text={response.text[:1600]!r}"
            )
        print(
            "GPTOSS_LORA_LOADED "
            f"index={index + 1}/{len(lora_paths)} name={args.lora_name} "
            f"path={path} load_inplace={index > 0} "
            f"response={response.text[:400]!r}",
            flush=True,
        )


async def post_chat(
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    body: dict[str, Any],
    record: ReplayRecord,
    request_index: int,
) -> str:
    started = time.monotonic()
    tag = (
        f"request-{request_index}-rollout-{record.rollout_index}"
        f"-turn-{record.turn_index}"
    )
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
                "GPTOSS_REQUEST_EXCEPTION "
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
            "GPTOSS_REQUEST_HTTP_ERROR "
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
            "GPTOSS_REQUEST_JSON_ERROR "
            f"tag={tag} example_id={record.example_id} body_len={body_len} "
            f"elapsed_seconds={elapsed:.2f} exc={type(exc).__name__}: {exc} "
            f"text={text!r}",
            flush=True,
        )
        return status

    bad_paths = nonfinite_paths(response_json)
    if bad_paths:
        print(
            "GPTOSS_REQUEST_NONFINITE "
            f"tag={tag} example_id={record.example_id} body_len={body_len} "
            f"elapsed_seconds={elapsed:.2f} paths={bad_paths[:16]} "
            f"{chat_response_summary(response_json)}",
            flush=True,
        )
        return "nonfinite"

    print(
        "GPTOSS_REQUEST_OK "
        f"tag={tag} example_id={record.example_id} body_len={body_len} "
        f"elapsed_seconds={elapsed:.2f} {chat_response_summary(response_json)}",
        flush=True,
    )
    return "ok"


async def post_completion(
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    body: dict[str, Any],
    record: ReplayRecord,
    request_index: int,
) -> str:
    started = time.monotonic()
    tag = (
        f"token-request-{request_index}-rollout-{record.rollout_index}"
        f"-turn-{record.turn_index}"
    )
    prompt_len = len(body["prompt"])
    async with semaphore:
        try:
            response = await client.post(
                "/completions",
                json=body,
                headers={"X-Request-Id": tag},
            )
        except Exception as exc:
            print(
                "GPTOSS_TOKEN_EXCEPTION "
                f"tag={tag} example_id={record.example_id} prompt_len={prompt_len} "
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
            "GPTOSS_TOKEN_HTTP_ERROR "
            f"tag={tag} example_id={record.example_id} prompt_len={prompt_len} "
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
            "GPTOSS_TOKEN_JSON_ERROR "
            f"tag={tag} example_id={record.example_id} prompt_len={prompt_len} "
            f"elapsed_seconds={elapsed:.2f} exc={type(exc).__name__}: {exc} "
            f"text={text!r}",
            flush=True,
        )
        return status

    bad_paths = nonfinite_paths(response_json)
    if bad_paths:
        print(
            "GPTOSS_TOKEN_NONFINITE "
            f"tag={tag} example_id={record.example_id} prompt_len={prompt_len} "
            f"elapsed_seconds={elapsed:.2f} paths={bad_paths[:16]} "
            f"{completion_response_summary(response_json)}",
            flush=True,
        )
        return "nonfinite"

    print(
        "GPTOSS_TOKEN_OK "
        f"tag={tag} example_id={record.example_id} prompt_len={prompt_len} "
        f"elapsed_seconds={elapsed:.2f} "
        f"{completion_response_summary(response_json)}",
        flush=True,
    )
    return "ok"


async def main_async() -> int:
    args = parse_args()
    records = repeat_records(collect_records(args), args.repeat_to)
    print(
        "GPTOSS_REPLAY_START "
        f"model={args.model} lora_name={args.lora_name} records={len(records)} "
        f"concurrency={args.concurrency} "
        f"max_completion_tokens={args.max_completion_tokens} "
        f"ignore_eos={args.ignore_eos} endpoint={args.endpoint}",
        flush=True,
    )
    for index, record in enumerate(records[:8]):
        print(
            "GPTOSS_REPLAY_SOURCE "
            f"index={index} rollout={record.rollout_index} "
            f"example_id={record.example_id} turn={record.turn_index} "
            f"messages={len(record.messages)} tools={len(record.tools)}",
            flush=True,
        )

    limits = httpx.Limits(
        max_connections=max(args.concurrency + 32, 512),
        max_keepalive_connections=max(args.concurrency + 32, 512),
    )
    timeout = httpx.Timeout(args.timeout_s, connect=60.0)
    semaphore = asyncio.Semaphore(args.concurrency)
    async with httpx.AsyncClient(
        base_url=args.base_url,
        timeout=timeout,
        limits=limits,
        trust_env=False,
    ) as client:
        await load_lora_sequence(client, args)
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        tasks = []
        for request_index, record in enumerate(records):
            if args.endpoint == "chat":
                body = make_chat_body(record, args, request_index)
                tasks.append(post_chat(client, semaphore, body, record, request_index))
            else:
                body = make_completion_body(tokenizer, record, args, request_index)
                tasks.append(
                    post_completion(client, semaphore, body, record, request_index)
                )

        counts: dict[str, int] = {}
        started = time.monotonic()
        for task in asyncio.as_completed(tasks):
            status = await task
            counts[status] = counts.get(status, 0) + 1
            if status == "nonfinite" and args.stop_on_nonfinite:
                print(
                    "GPTOSS_SUMMARY "
                    f"elapsed_seconds={time.monotonic() - started:.2f} "
                    f"status_counts={counts}",
                    flush=True,
                )
                print("GPTOSS_RESULT reproduced_nonfinite", flush=True)
                return 2

    print(
        "GPTOSS_SUMMARY "
        f"elapsed_seconds={time.monotonic() - started:.2f} "
        f"status_counts={counts}",
        flush=True,
    )
    if counts.get("nonfinite"):
        print("GPTOSS_RESULT reproduced_nonfinite", flush=True)
        return 2
    if any(status != "ok" for status in counts):
        print("GPTOSS_RESULT inconclusive", flush=True)
        return 1
    print("GPTOSS_RESULT no_nonfinite_observed", flush=True)
    return 0


def main() -> None:
    raise SystemExit(asyncio.run(main_async()))


if __name__ == "__main__":
    main()
