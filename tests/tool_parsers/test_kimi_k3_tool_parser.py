# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

from vllm.parser.kimi_k3 import KimiK3Parser
from vllm.reasoning.kimi_k3_reasoning_parser import KimiK3ReasoningParser
from vllm.tool_parsers.kimi_k3_tool_parser import KimiK3ToolParser


def test_streaming_response_drops_late_think_close_boundary():
    parser = KimiK3ToolParser(object())
    text = ".<|close|>think<|sep|><|open|>response<|sep|>天地玄黄"

    assert parser._extract_response_content(text) == "天地玄黄"


def test_streaming_response_holds_partial_think_close_boundary():
    parser = KimiK3ToolParser(object())

    partial_think_boundary = ".<|close|>think"[:-2]
    assert parser._extract_response_content(partial_think_boundary) is None
    assert (
        parser._extract_response_content(".<|close|>think<|sep|>天地玄黄") == "天地玄黄"
    )


class _Tokenizer:
    def encode(self, text, add_special_tokens=False):
        return list(text.encode())

    def get_vocab(self):
        return {}


class _ReasoningOnlyKimiK3Parser(KimiK3Parser):
    reasoning_parser_cls = KimiK3ReasoningParser
    tool_parser_cls = None


class _AutoToolKimiK3Parser(KimiK3Parser):
    reasoning_parser_cls = KimiK3ReasoningParser
    tool_parser_cls = KimiK3ToolParser


def _parse_stream(parser_cls):
    parser = parser_cls(
        _Tokenizer(),
        tools=None,
        chat_template_kwargs={"thinking": True},
    )
    request = SimpleNamespace(
        tools=None,
        tool_choice=None,
        include_reasoning=True,
    )
    chunks = [
        (
            "分析最后一句。<|close|>think<|sep|>"
            "<|open|>response<|sep|>天地玄黄，宇宙洪荒。<|"
        ),
        "close|>response<|sep|><|close|>message<|sep|>",
    ]
    reasoning = []
    content = []
    for index, chunk in enumerate(chunks):
        delta = parser.parse_delta(
            chunk,
            list(chunk.encode()),
            request,
            prompt_token_ids=(list(b"<|open|>think<|sep|>") if index == 0 else None),
            finished=index == len(chunks) - 1,
        )
        if delta is not None:
            if delta.reasoning:
                reasoning.append(delta.reasoning)
            if delta.content:
                content.append(delta.content)
    return "".join(reasoning), "".join(content)


def test_auto_tool_parser_preserves_plain_response_across_split_close_marker():
    expected = ("分析最后一句。", "天地玄黄，宇宙洪荒。")

    assert _parse_stream(_ReasoningOnlyKimiK3Parser) == expected
    assert _parse_stream(_AutoToolKimiK3Parser) == expected


def test_composed_auto_tool_parser_still_extracts_tool_call():
    parser = _AutoToolKimiK3Parser(
        _Tokenizer(),
        tools=[{"type": "function", "function": {"name": "search"}}],
        chat_template_kwargs={"thinking": True},
    )
    request = SimpleNamespace(
        tools=[{"type": "function"}],
        tool_choice="auto",
        include_reasoning=True,
    )
    output = (
        "分析。<|close|>think<|sep|>"
        "<|open|>response<|sep|>先查。<|close|>response<|sep|>"
        '<|open|>tools<|sep|><|open|>call tool="search" index="1"<|sep|>'
        '<|open|>argument key="q" type="string"<|sep|>天气'
        "<|close|>argument<|sep|><|close|>call<|sep|>"
        "<|close|>tools<|sep|><|close|>message<|sep|>"
    )

    delta = parser.parse_delta(
        output,
        list(output.encode()),
        request,
        prompt_token_ids=list(b"<|open|>think<|sep|>"),
        finished=True,
    )

    assert delta is not None
    assert delta.reasoning == "分析。"
    assert delta.content == "先查。"
    assert delta.tool_calls
    assert delta.tool_calls[0].function
    assert delta.tool_calls[0].function.name == "search"
    assert delta.tool_calls[0].function.arguments == '{"q": "天气"}'


def test_composed_content_strips_late_think_close_boundary():
    parser = KimiK3ReasoningParser(_Tokenizer())
    text = ".<|close|>think<|sep|><|open|>response<|sep|>天地玄黄"

    assert parser._content_ready_to_emit(text) == "天地玄黄"
