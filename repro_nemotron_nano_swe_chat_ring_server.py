#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Chat-ring replay for the Nemotron Nano SWE NaN.

Exact local setting:
- Model: nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16
- Source diagnostic:
  /beegfs/daniel/nemotron-nano-swe-step7-default-128k/chat_nan_diagnostics/1778796420266_chat_nonfinite_response_949b56786333e335.json
- Endpoint under test: /v1/chat/completions
- Workload: captured failing request plus its recent request ring, preserving
  chat messages/tools/logprobs and optionally forcing long decode tails.

This is a closer standalone replay than the raw-token width-poison control:
it keeps the production chat request shape and heterogeneous prompt lengths,
but it still does not recreate the full orchestrator/sandbox environment.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import time
from pathlib import Path
from typing import Any

import httpx

MODEL = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
DIAGNOSTIC_PATH = Path(
    "/beegfs/daniel/nemotron-nano-swe-step7-default-128k/"
    "chat_nan_diagnostics/1778796420266_chat_nonfinite_response_949b56786333e335.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--input", type=Path, default=DIAGNOSTIC_PATH)
    parser.add_argument("--repeat", type=int, default=8)
    parser.add_argument("--concurrency", type=int, default=512)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--sort-by-content-length", choices=["none", "asc", "desc"], default="none"
    )
    parser.add_argument("--force-ignore-eos", action="store_true")
    parser.add_argument("--max-completion-tokens", type=int, default=2048)
    parser.add_argument(
        "--allowed-token-ids",
        default="",
        help="Comma-separated token ids to constrain sampling, e.g. '0'.",
    )
    parser.add_argument("--timeout-s", type=float, default=1800.0)
    return parser.parse_args()


def parse_allowed_token_ids(value: str) -> list[int] | None:
    token_ids = [int(item) for item in value.split(",") if item.strip()]
    return token_ids or None


def body_content_length(body: dict[str, Any]) -> int:
    return len(json.dumps(body, sort_keys=True, separators=(",", ":")))


def collect_input_paths(path: Path) -> list[Path]:
    if path.is_dir():
        paths = sorted(path.glob("*chat_nonfinite_response*.json"))
    else:
        paths = [path]
    if not paths:
        raise ValueError(f"no diagnostic files found at {path}")
    return paths


def load_request_bodies(path: Path) -> list[tuple[str, dict[str, Any]]]:
    records: list[tuple[str, dict[str, Any]]] = []
    input_paths = collect_input_paths(path)
    print(
        "CHAT_REPLAY_INPUTS "
        f"path={path} files={len(input_paths)} "
        f"names={[p.name for p in input_paths]}",
        flush=True,
    )

    for input_path in input_paths:
        with input_path.open() as f:
            diagnostic = json.load(f)

        prefix = input_path.stem
        per_file_count = 0
        for index, recent in enumerate(diagnostic.get("recent_requests") or []):
            body = recent.get("body")
            if isinstance(body, dict):
                request_id = recent.get("request_id") or f"recent-{index}"
                records.append((f"{prefix}:recent:{request_id}", body))
                per_file_count += 1

        request_body = diagnostic.get("request", {}).get("body")
        if isinstance(request_body, dict):
            request_id = diagnostic.get("request_id") or "failing"
            records.append((f"{prefix}:failing:{request_id}", request_body))
            per_file_count += 1

        print(
            "CHAT_REPLAY_INPUT_FILE "
            f"name={input_path.name} records={per_file_count} "
            f"request_id={diagnostic.get('request_id')} "
            f"bad_paths={len(diagnostic.get('response_nonfinite_paths') or [])}",
            flush=True,
        )

    if not records:
        raise ValueError(f"no request bodies found in {path}")
    return records


def make_body(
    body: dict[str, Any],
    *,
    request_id: str,
    salt: int,
    force_ignore_eos: bool,
    max_completion_tokens: int,
    allowed_token_ids: list[int] | None,
) -> dict[str, Any]:
    replay_body = dict(body)
    replay_body["model"] = MODEL
    replay_body["stream"] = False
    replay_body["logprobs"] = True
    replay_body["return_token_ids"] = True
    replay_body["request_id"] = f"replay-{request_id}-{salt}"
    replay_body["cache_salt"] = f"replay-{salt}"
    if force_ignore_eos:
        replay_body["ignore_eos"] = True
    if max_completion_tokens > 0:
        replay_body["max_completion_tokens"] = max_completion_tokens
    if allowed_token_ids is not None:
        replay_body["allowed_token_ids"] = allowed_token_ids
    return replay_body


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
    *,
    tag: str,
    source_request_id: str,
    body_len: int,
) -> str:
    started = time.monotonic()
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
                f"tag={tag} source={source_request_id} body_len={body_len} "
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
            f"tag={tag} source={source_request_id} body_len={body_len} "
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
            f"tag={tag} source={source_request_id} body_len={body_len} "
            f"elapsed_seconds={elapsed:.2f} exc={type(exc).__name__}: {exc} "
            f"text={text!r}",
            flush=True,
        )
        return status

    bad_paths = nonfinite_paths(response_json)
    if bad_paths:
        print(
            "REQUEST_NONFINITE "
            f"tag={tag} source={source_request_id} body_len={body_len} "
            f"elapsed_seconds={elapsed:.2f} paths={bad_paths[:16]} "
            f"{response_summary(response_json)}",
            flush=True,
        )
        return "nonfinite"

    print(
        "REQUEST_OK "
        f"tag={tag} source={source_request_id} body_len={body_len} "
        f"elapsed_seconds={elapsed:.2f} {response_summary(response_json)}",
        flush=True,
    )
    return "ok"


async def main_async() -> int:
    args = parse_args()
    records = load_request_bodies(args.input)
    if args.sort_by_content_length != "none":
        reverse = args.sort_by_content_length == "desc"
        records = sorted(
            records, key=lambda item: body_content_length(item[1]), reverse=reverse
        )
    if args.limit > 0:
        records = records[: args.limit]

    allowed_token_ids = parse_allowed_token_ids(args.allowed_token_ids)
    expanded = records * args.repeat
    print(
        "CHAT_REPLAY_START "
        f"input={args.input} base_records={len(records)} total_requests={len(expanded)} "
        f"repeat={args.repeat} concurrency={args.concurrency} "
        f"force_ignore_eos={args.force_ignore_eos} "
        f"max_completion_tokens={args.max_completion_tokens} "
        f"allowed_token_ids={allowed_token_ids}",
        flush=True,
    )
    for index, (request_id, body) in enumerate(records[:8]):
        print(
            "CHAT_REPLAY_SOURCE "
            f"index={index} request_id={request_id} body_len={body_content_length(body)} "
            f"messages={len(body.get('messages') or [])}",
            flush=True,
        )

    limits = httpx.Limits(
        max_connections=max(args.concurrency + 32, 1024),
        max_keepalive_connections=max(args.concurrency + 32, 1024),
    )
    timeout = httpx.Timeout(args.timeout_s, connect=60.0)
    semaphore = asyncio.Semaphore(args.concurrency)
    async with httpx.AsyncClient(
        base_url=args.base_url, timeout=timeout, limits=limits
    ) as client:
        tasks = []
        for index, (request_id, body) in enumerate(expanded):
            replay_body = make_body(
                body,
                request_id=request_id,
                salt=index,
                force_ignore_eos=args.force_ignore_eos,
                max_completion_tokens=args.max_completion_tokens,
                allowed_token_ids=allowed_token_ids,
            )
            body_len = body_content_length(replay_body)
            tasks.append(
                post_chat(
                    client,
                    semaphore,
                    replay_body,
                    tag=f"chat-ring-{index}",
                    source_request_id=request_id,
                    body_len=body_len,
                )
            )

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
                    "RESULT reproduced: non-finite chat response observed", flush=True
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
