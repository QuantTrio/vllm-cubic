# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Validate a running Kimi-K3 Cubic OpenAI-compatible server."""

import argparse
import http.client
import json
import sys
from collections import Counter

import regex as re

PROMPT = "背诵千字文全文"
QIANZIWEN_ANCHORS = (
    "天地玄黄",
    "寒来暑往",
    "云腾致雨",
    "金生丽水",
    "龙师火帝",
    "坐朝问道",
    "罔谈彼短",
    "矩步引领",
    "谓语助者",
    "焉哉乎也",
)


def has_repetition_loop(text: str, ngram_size: int = 32) -> bool:
    compact = "".join(text.split())
    if len(compact) < ngram_size * 3:
        return False
    counts = Counter(
        compact[index : index + ngram_size]
        for index in range(len(compact) - ngram_size + 1)
    )
    return any(count >= 3 for count in counts.values())


def recited_qianziwen(text: str) -> bool:
    chinese = "".join(
        character for character in text if "\u4e00" <= character <= "\u9fff"
    )
    return len(chinese) >= 950 and all(
        anchor in chinese for anchor in QIANZIWEN_ANCHORS
    )


def stream_request(
    host: str,
    port: int,
    model: str,
    max_tokens: int,
    thinking_effort: str | None,
    seed: int | None,
) -> tuple[str, str, str | None, dict | None]:
    request = {
        "model": model,
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if seed is not None:
        request["seed"] = seed
    if thinking_effort is not None:
        request["chat_template_kwargs"] = {
            "thinking": True,
            "thinking_effort": thinking_effort,
        }
    body = json.dumps(request, ensure_ascii=False).encode()
    connection = http.client.HTTPConnection(host, port, timeout=3600)
    connection.request(
        "POST",
        "/v1/chat/completions",
        body=body,
        headers={"Content-Type": "application/json"},
    )
    response = connection.getresponse()
    if response.status != 200:
        raise RuntimeError(
            f"HTTP {response.status}: {response.read().decode(errors='replace')}"
        )

    reasoning_parts: list[str] = []
    content_parts: list[str] = []
    finish_reason = None
    usage = None
    active_section = None
    for raw_line in response:
        line = raw_line.decode(errors="replace").strip()
        if not line.startswith("data: "):
            continue
        payload = line.removeprefix("data: ")
        if payload == "[DONE]":
            break
        chunk = json.loads(payload)
        usage = chunk.get("usage") or usage
        for choice in chunk.get("choices", []):
            delta = choice.get("delta", {})
            reasoning = delta.get("reasoning") or delta.get("reasoning_content") or ""
            content = delta.get("content") or ""
            if reasoning:
                if active_section != "reasoning":
                    print("\n[reasoning]", flush=True)
                    active_section = "reasoning"
                print(reasoning, end="", flush=True)
                reasoning_parts.append(reasoning)
            if content:
                if active_section != "content":
                    print("\n[content]", flush=True)
                    active_section = "content"
                print(content, end="", flush=True)
                content_parts.append(content)
            finish_reason = choice.get("finish_reason") or finish_reason
    print("\n[model_output_end]", flush=True)
    connection.close()
    return (
        "".join(reasoning_parts),
        "".join(content_parts),
        finish_reason,
        usage,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=30075)
    parser.add_argument("--model", default="Kimi-K3-Cubic-2.5bit")
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--thinking-effort",
        choices=("low", "high", "max"),
    )
    args = parser.parse_args()

    request = {
        "prompt": PROMPT,
        "temperature": "server/model default",
        "thinking": args.thinking_effort or "chat-template default",
        "max_tokens": args.max_tokens,
        "seed": args.seed if args.seed is not None else "server/model default",
    }
    print(json.dumps(request, ensure_ascii=False), flush=True)
    print("[model_output_begin]", flush=True)
    reasoning, content, finish_reason, usage = stream_request(
        args.host,
        args.port,
        args.model,
        args.max_tokens,
        args.thinking_effort,
        args.seed,
    )
    verdict = {
        "prompt_exact": PROMPT,
        "thinking_present": bool(reasoning),
        "recited_qianziwen": recited_qianziwen(content),
        "reasoning_not_repeating": not has_repetition_loop(reasoning),
        "content_not_repeating": not has_repetition_loop(content),
        "content_has_no_latin_text": re.search(r"[A-Za-z]", content) is None,
        "finish_reason": finish_reason,
        "stopped_normally": finish_reason == "stop",
        "usage": usage,
    }
    verdict["passed"] = all(
        value
        for key, value in verdict.items()
        if key
        not in {
            "prompt_exact",
            "finish_reason",
            "usage",
        }
    )
    print(json.dumps(verdict, ensure_ascii=False, indent=2), flush=True)
    if not verdict["passed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
