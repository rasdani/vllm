#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Concurrent OpenAI-server replay for the Nemotron Nano SWE NaN diagnostic.

This uses the embedded Nano/SWE chat requests from
`repro_nemotron_nano_swe_chat_window_standalone.py`, but sends them through a
running `/v1/chat/completions` server. That keeps the OpenAI serving path,
continuous batching, scheduler, CUDA graph dispatch, and JSON serialization in
scope, which the offline `LLM.chat()` replay intentionally does not.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import math
import time
from typing import Any

import httpx

import repro_nemotron_nano_swe_chat_window_standalone as replay_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument(
        "--scenario",
        default="recent64-plus-failure",
        choices=tuple(replay_data.SCENARIOS),
    )
    parser.add_argument("--concurrency", type=int, default=64)
    parser.add_argument("--waves", type=int, default=1)
    parser.add_argument("--request-timeout", type=float, default=900.0)
    parser.add_argument("--wave-sleep", type=float, default=0.0)
    parser.add_argument(
        "--max-tokens-cap", type=int, default=replay_data.REPLAY_MAX_TOKENS_CAP
    )
    return parser.parse_args()


def selected_rows(scenario: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = replay_data.load_payload()
    rows = replay_data.select_requests(payload, scenario)
    return payload, rows


def capped_body(row: dict[str, Any], max_tokens_cap: int) -> dict[str, Any]:
    body = copy.deepcopy(row["body"])
    requested_max = (
        body.get("max_completion_tokens") or body.get("max_tokens") or max_tokens_cap
    )
    body["max_completion_tokens"] = min(int(requested_max), max_tokens_cap)
    body["stream"] = False
    return body


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


def first_choice_summary(response_json: dict[str, Any]) -> str:
    choices = response_json.get("choices") or []
    if not choices:
        return "choices=0"
    choice = choices[0]
    logprobs = choice.get("logprobs") or {}
    content_logprobs = logprobs.get("content") or []
    message = choice.get("message") or {}
    text = message.get("content") or ""
    return (
        f"choices={len(choices)} finish={choice.get('finish_reason')} "
        f"content_chars={len(text)} logprob_entries={len(content_logprobs)}"
    )


async def post_one(
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    row: dict[str, Any],
    wave: int,
    index: int,
    max_tokens_cap: int,
) -> tuple[str, str]:
    request_id = str(row.get("request_id"))
    body = capped_body(row, max_tokens_cap)
    started = time.monotonic()
    async with semaphore:
        try:
            response = await client.post(
                "/chat/completions",
                json=body,
                headers={"X-Request-Id": request_id},
            )
        except Exception as exc:
            elapsed = time.monotonic() - started
            print(
                "REQUEST_EXCEPTION "
                f"wave={wave} index={index} request_id={request_id} "
                f"elapsed_seconds={elapsed:.2f} exc={type(exc).__name__}: {exc}",
                flush=True,
            )
            return request_id, "exception"

    elapsed = time.monotonic() - started
    if response.status_code >= 400:
        print(
            "REQUEST_HTTP_ERROR "
            f"wave={wave} index={index} request_id={request_id} "
            f"status={response.status_code} elapsed_seconds={elapsed:.2f} "
            f"text={response.text[:1000]!r}",
            flush=True,
        )
        return (
            request_id,
            "nonfinite" if "nan" in response.text.lower() else "http_error",
        )

    try:
        response_json = json.loads(response.text)
    except Exception as exc:
        print(
            "REQUEST_JSON_ERROR "
            f"wave={wave} index={index} request_id={request_id} "
            f"elapsed_seconds={elapsed:.2f} exc={type(exc).__name__}: {exc} "
            f"text={response.text[:1000]!r}",
            flush=True,
        )
        return (
            request_id,
            "nonfinite" if "nan" in response.text.lower() else "json_error",
        )

    bad_paths = nonfinite_paths(response_json)
    if bad_paths:
        print(
            "REQUEST_NONFINITE "
            f"wave={wave} index={index} request_id={request_id} "
            f"elapsed_seconds={elapsed:.2f} paths={bad_paths[:16]} "
            f"{first_choice_summary(response_json)}",
            flush=True,
        )
        return request_id, "nonfinite"

    print(
        "REQUEST_OK "
        f"wave={wave} index={index} request_id={request_id} "
        f"elapsed_seconds={elapsed:.2f} {first_choice_summary(response_json)}",
        flush=True,
    )
    return request_id, "ok"


async def main_async() -> int:
    args = parse_args()
    payload, rows = selected_rows(args.scenario)
    print(
        "SERVER_REPLAY_SETTING "
        f"base_url={args.base_url} scenario={args.scenario} selected_count={len(rows)} "
        f"waves={args.waves} concurrency={args.concurrency} max_tokens_cap={args.max_tokens_cap} "
        f"source={payload['source_file']} failure_request_id={replay_data.FAILURE_REQUEST_ID} "
        f"failure_present={any(row.get('request_id') == replay_data.FAILURE_REQUEST_ID for row in rows)} "
        f"captured_nonfinite_paths={len(payload.get('failure_nonfinite_paths') or [])}",
        flush=True,
    )
    for index, row in enumerate(rows):
        body = row["body"]
        print(
            "INPUT_ROW "
            f"index={index} request_id={row.get('request_id')} "
            f"messages={len(body.get('messages') or [])} "
            f"max_completion_tokens={body.get('max_completion_tokens')} "
            f"tool_choice={body.get('tool_choice')}",
            flush=True,
        )

    semaphore = asyncio.Semaphore(args.concurrency)
    timeout = httpx.Timeout(args.request_timeout)
    limits = httpx.Limits(max_connections=max(args.concurrency, 1) + 8)
    status_counts: dict[str, int] = {}
    nonfinite_ids: list[str] = []
    start = time.monotonic()
    async with httpx.AsyncClient(
        base_url=args.base_url,
        timeout=timeout,
        limits=limits,
        trust_env=False,
    ) as client:
        for wave in range(args.waves):
            wave_start = time.monotonic()
            results = await asyncio.gather(
                *[
                    post_one(client, semaphore, row, wave, index, args.max_tokens_cap)
                    for index, row in enumerate(rows)
                ]
            )
            for _, status in results:
                status_counts[status] = status_counts.get(status, 0) + 1
            wave_nonfinite = [
                request_id for request_id, status in results if status == "nonfinite"
            ]
            wave_errors = [
                request_id
                for request_id, status in results
                if status not in ("ok", "nonfinite")
            ]
            nonfinite_ids.extend(wave_nonfinite)
            print(
                "WAVE_SUMMARY "
                f"wave={wave} elapsed_seconds={time.monotonic() - wave_start:.2f} "
                f"nonfinite_count={len(wave_nonfinite)} nonfinite_ids={wave_nonfinite} "
                f"error_count={len(wave_errors)} error_ids={wave_errors[:16]}",
                flush=True,
            )
            if args.wave_sleep and wave + 1 < args.waves:
                await asyncio.sleep(args.wave_sleep)

    print(
        "SUMMARY "
        f"elapsed_seconds={time.monotonic() - start:.2f} "
        f"requests={len(rows) * args.waves} nonfinite_count={len(nonfinite_ids)} "
        f"nonfinite_ids={nonfinite_ids} status_counts={status_counts}",
        flush=True,
    )
    if nonfinite_ids:
        print(
            "RESULT failed: observed server non-finite response or JSON NaN error",
            flush=True,
        )
        return 2
    if any(status != "ok" and count for status, count in status_counts.items()):
        print(
            "RESULT inconclusive: server replay had non-NaN request errors", flush=True
        )
        return 1
    print("RESULT passed: no server non-finite response observed", flush=True)
    return 0


def main() -> int:
    return asyncio.run(main_async())


if __name__ == "__main__":
    raise SystemExit(main())
