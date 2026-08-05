# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Any, cast

from vllm.config import VllmConfig
from vllm.entrypoints.chat_utils import (
    ChatCompletionMessageParam,
    ConversationMessage,
    parse_chat_messages,
    parse_chat_messages_async,
)
from vllm.exceptions import VLLMValidationError
from vllm.multimodal.media.connector import merge_media_io_kwargs
from vllm.tokenizers.hf import HfTokenizer
from vllm.utils.async_utils import make_async

from .base import BaseRenderer
from .inputs import DictPrompt
from .inputs.preprocess import parse_dec_only_prompt
from .params import ChatParams

_K3_MEDIA_IO_DEFAULTS: dict[str, dict[str, Any]] = {"image": {"image_mode": None}}
_K3_THINKING_EFFORTS = ("low", "high", "max")


def _merge_k3_media_io_kwargs(
    media_io_kwargs: dict[str, dict[str, Any]] | None,
) -> dict[str, dict[str, Any]] | None:
    return merge_media_io_kwargs(_K3_MEDIA_IO_DEFAULTS, media_io_kwargs)


def _dump_k3_template_value(value: Any) -> Any:
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return model_dump(mode="json", exclude_none=True)

    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return to_dict()

    return value


def _apply_k3_thinking_kwargs(kwargs: dict[str, Any]) -> None:
    if (enable_thinking := kwargs.pop("enable_thinking", None)) is not None:
        kwargs.setdefault("thinking", enable_thinking)

    reasoning_effort = kwargs.pop("reasoning_effort", None)
    if reasoning_effort == "none":
        kwargs.setdefault("thinking", False)
    elif reasoning_effort is not None:
        kwargs.setdefault("thinking_effort", reasoning_effort)

    thinking_effort = kwargs.get("thinking_effort")
    if thinking_effort is not None and thinking_effort not in _K3_THINKING_EFFORTS:
        supported = ", ".join(_K3_THINKING_EFFORTS)
        raise VLLMValidationError(
            f"Kimi K3 supports thinking_effort values: {supported}",
            parameter="thinking_effort",
            value=thinking_effort,
        )


def _normalize_k3_tool_messages(
    conversation: list[ConversationMessage],
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    i = 0

    while i < len(conversation):
        message = conversation[i]
        normalized.append(dict(message))
        i += 1

        if message.get("role") != "assistant":
            continue

        tool_calls = message.get("tool_calls")
        if not tool_calls:
            continue

        targets_by_id: dict[str, tuple[int, str]] = {}
        for position, tool_call in enumerate(tool_calls):
            function = tool_call.get("function")
            if not isinstance(function, dict):
                continue
            name = function.get("name")
            if not isinstance(name, str) or not name:
                continue

            call_target = (position, name)
            aliases = [f"{name}:{position}"]
            if tool_call_id := tool_call.get("id"):
                aliases.insert(0, str(tool_call_id))
            for alias in aliases:
                targets_by_id.setdefault(alias, call_target)

        block_start = i
        while i < len(conversation) and conversation[i].get("role") == "tool":
            i += 1
        if i == block_start:
            continue

        resolved: list[tuple[int, int, dict[str, Any]]] = []
        for order, tool_message in enumerate(conversation[block_start:i]):
            tool_call_id = tool_message.get("tool_call_id")
            resolved_target = (
                targets_by_id.get(str(tool_call_id))
                if tool_call_id is not None
                else None
            )
            if resolved_target is None:
                normalized.extend(dict(item) for item in conversation[block_start:i])
                break

            position, name = resolved_target
            enriched = dict(tool_message)
            enriched["tool"] = name
            enriched["index"] = position + 1
            resolved.append((position, order, enriched))
        else:
            resolved.sort(key=lambda item: (item[0], item[1]))
            normalized.extend(item for _, _, item in resolved)

    return normalized


class KimiK3Renderer(BaseRenderer[HfTokenizer]):
    """Render Kimi-K3 chats with the tokenizer's Python XTML encoding."""

    def __init__(self, config: VllmConfig, tokenizer: HfTokenizer | None) -> None:
        super().__init__(config, tokenizer)
        self._apply_chat_template_async = make_async(
            self._apply_chat_template, executor=self._executor
        )

    def _apply_chat_template(
        self,
        conversation: list[dict[str, Any]],
        params: ChatParams,
    ) -> list[int]:
        kwargs = params.get_apply_chat_template_kwargs()
        _apply_k3_thinking_kwargs(kwargs)
        if params.tool_choice not in (None, "auto"):
            kwargs["tool_choice"] = _dump_k3_template_value(params.tool_choice)
        if params.response_format is not None:
            kwargs["response_format"] = _dump_k3_template_value(params.response_format)
        kwargs["tokenize"] = True
        return self.get_tokenizer().apply_chat_template(conversation, **kwargs)

    def render_messages(
        self,
        messages: list[ChatCompletionMessageParam],
        params: ChatParams,
    ) -> tuple[list[ConversationMessage], DictPrompt]:
        conversation, mm_data, mm_uuids = parse_chat_messages(
            messages,
            self.model_config,
            content_format="string",
            media_io_kwargs=_merge_k3_media_io_kwargs(params.media_io_kwargs),
            mm_processor_kwargs=params.mm_processor_kwargs,
        )

        rendered_conversation = _normalize_k3_tool_messages(conversation)
        prompt = parse_dec_only_prompt(
            self._apply_chat_template(rendered_conversation, params)
        )
        if mm_data is not None:
            prompt["multi_modal_data"] = mm_data
        if mm_uuids is not None:
            prompt["multi_modal_uuids"] = mm_uuids

        return cast(list[ConversationMessage], rendered_conversation), prompt

    async def render_messages_async(
        self,
        messages: list[ChatCompletionMessageParam],
        params: ChatParams,
    ) -> tuple[list[ConversationMessage], DictPrompt]:
        conversation, mm_data, mm_uuids = await parse_chat_messages_async(
            messages,
            self.model_config,
            content_format="string",
            media_io_kwargs=_merge_k3_media_io_kwargs(params.media_io_kwargs),
            mm_processor_kwargs=params.mm_processor_kwargs,
        )

        rendered_conversation = _normalize_k3_tool_messages(conversation)
        token_ids = await self._apply_chat_template_async(rendered_conversation, params)
        prompt = parse_dec_only_prompt(token_ids)
        if mm_data is not None:
            prompt["multi_modal_data"] = mm_data
        if mm_uuids is not None:
            prompt["multi_modal_uuids"] = mm_uuids

        return cast(list[ConversationMessage], rendered_conversation), prompt
