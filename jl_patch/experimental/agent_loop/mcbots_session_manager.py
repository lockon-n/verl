from __future__ import annotations

import json
from copy import deepcopy
from typing import Any, Optional

from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast, ProcessorMixin

from verl.experimental.agent_loop.mcbots_bridge import McbotsBridge
from verl.tools.schemas import OpenAIFunctionToolCall, ToolResponse
from verl.workers.rollout.schemas import AsyncRolloutRequest, FinishReasonTypeEnum, Message, TokenizationSanityCheckModeEnum


class McbotsSessionManager:
    """Manage one incremental bridge per episode and sync upstream message suffixes.

    When a context reset occurs mid-episode, the current bridge is finalized and
    its rollout is stored.  ``finalize_episode`` returns **all** rollouts
    (from context resets + the final segment) so the caller can feed every
    segment into training under the same episode_id.
    """

    def __init__(
        self,
        processing_class: PreTrainedTokenizer | PreTrainedTokenizerFast | ProcessorMixin,
        *,
        multi_modal_keys: Optional[list[str]] = None,
        max_prompt_len: int = 8192,
        max_response_len: int = 8192,
        max_model_len: int = 32768,
        use_inference_chat_template: bool = False,
        tokenization_sanity_check_mode: TokenizationSanityCheckModeEnum = TokenizationSanityCheckModeEnum.STRICT,
    ) -> None:
        self.processing_class = processing_class
        self.multi_modal_keys = multi_modal_keys
        self.max_prompt_len = max_prompt_len
        self.max_response_len = max_response_len
        self.max_model_len = max_model_len
        self.use_inference_chat_template = use_inference_chat_template
        self.tokenization_sanity_check_mode = tokenization_sanity_check_mode
        self._bridges: dict[str, McbotsBridge] = {}
        self._completed: dict[str, list[AsyncRolloutRequest]] = {}

    def has_episode(self, episode_id: str) -> bool:
        return episode_id in self._bridges or episode_id in self._completed

    def get_bridge(self, episode_id: str) -> McbotsBridge:
        try:
            return self._bridges[episode_id]
        except KeyError as exc:
            raise KeyError(f"Unknown episode_id {episode_id!r}") from exc

    def prepare_generation(
        self,
        *,
        episode_id: str,
        messages: list[dict[str, Any]],
        context_reset: bool = False,
        request_id: Optional[str] = None,
        multi_modal_keys: Optional[list[str]] = None,
        multi_modal_data: Optional[dict[str, Any]] = None,
    ) -> list[int]:
        bridge = self._sync_messages(
            episode_id=episode_id,
            messages=messages,
            context_reset=context_reset,
            request_id=request_id,
            multi_modal_keys=multi_modal_keys,
            multi_modal_data=multi_modal_data,
        )
        return bridge.get_generation_prompt_ids()

    def record_assistant_turn(
        self,
        *,
        episode_id: str,
        content: str,
        content_ids=None,
        tool_calls: Optional[list[OpenAIFunctionToolCall | dict[str, Any]]] = None,
        is_final: bool = False,
        reward_scores: Optional[dict[str, list[float]]] = None,
        finish_reason_type: FinishReasonTypeEnum = FinishReasonTypeEnum.STOP,
    ) -> list[AsyncRolloutRequest] | None:
        bridge = self.get_bridge(episode_id)
        bridge.add_assistant_message(
            content=content,
            content_ids=content_ids,
            tool_calls=self._coerce_tool_calls(tool_calls),
        )
        if is_final:
            return self.finalize_episode(
                episode_id=episode_id,
                reward_scores=reward_scores,
                finish_reason_type=finish_reason_type,
            )
        return None

    def finalize_episode(
        self,
        *,
        episode_id: str,
        reward_scores: Optional[dict[str, list[float]]] = None,
        finish_reason_type: FinishReasonTypeEnum = FinishReasonTypeEnum.STOP,
    ) -> list[AsyncRolloutRequest]:
        bridge = self.get_bridge(episode_id)
        final_request = bridge.finalize(
            reward_scores=reward_scores,
            finish_reason_type=finish_reason_type,
        )
        completed = self._completed.pop(episode_id, [])
        completed.append(final_request)
        self._bridges.pop(episode_id, None)
        return completed

    def _sync_messages(
        self,
        *,
        episode_id: str,
        messages: list[dict[str, Any]],
        context_reset: bool,
        request_id: Optional[str],
        multi_modal_keys: Optional[list[str]],
        multi_modal_data: Optional[dict[str, Any]],
    ) -> McbotsBridge:
        normalized_messages = [self._normalize_incoming_message(message) for message in messages]
        if not normalized_messages:
            raise ValueError("messages must not be empty")

        if context_reset and episode_id in self._bridges:
            old_bridge = self._bridges.pop(episode_id)
            completed_request = old_bridge.finalize(
                reward_scores={},
                finish_reason_type=FinishReasonTypeEnum.CONTEXT_RESET,
            )
            self._completed.setdefault(episode_id, []).append(completed_request)

        if context_reset or episode_id not in self._bridges:
            bridge = McbotsBridge(
                processing_class=self.processing_class,
                episode_id=episode_id,
                request_id=request_id,
                messages=normalized_messages,
                multi_modal_keys=multi_modal_keys or self.multi_modal_keys,
                multi_modal_data=multi_modal_data,
                max_prompt_len=self.max_prompt_len,
                max_response_len=self.max_response_len,
                max_model_len=self.max_model_len,
                use_inference_chat_template=self.use_inference_chat_template,
                tokenization_sanity_check_mode=self.tokenization_sanity_check_mode,
            )
            self._bridges[episode_id] = bridge
            return bridge

        bridge = self._bridges[episode_id]
        tracked_signatures = [self._message_signature(message) for message in bridge.request.messages]
        incoming_prefix = [self._message_signature(message) for message in normalized_messages[: len(tracked_signatures)]]

        if len(normalized_messages) < len(tracked_signatures):
            raise ValueError(
                f"Incoming history for episode_id={episode_id!r} is shorter than tracked history. "
                "Use context_reset=True when upstream history was trimmed or rebuilt."
            )
        if tracked_signatures != incoming_prefix:
            raise ValueError(
                f"Incoming history for episode_id={episode_id!r} diverged from tracked history. "
                "Use context_reset=True to rebuild the bridge state."
            )

        for message in normalized_messages[len(tracked_signatures) :]:
            self._append_upstream_suffix_message(bridge, message)
        return bridge

    def _append_upstream_suffix_message(self, bridge: McbotsBridge, message: dict[str, Any]) -> None:
        role = message["role"]
        content = message["content"]

        if role == "user":
            bridge.add_user_message(content)
            return

        if role == "tool":
            bridge.add_tool_response_messages([self._tool_response_from_content(content)])
            return

        if role == "assistant":
            raise ValueError(
                "Assistant messages cannot be recovered from upstream history for an active episode. "
                "Call record_assistant_turn(...) with the generated content and token ids instead."
            )

        if role == "system":
            raise ValueError("System messages are only supported when creating or context-resetting an episode.")

        raise ValueError(f"Unsupported message role: {role!r}")

    @staticmethod
    def _coerce_tool_calls(
        tool_calls: Optional[list[OpenAIFunctionToolCall | dict[str, Any]]],
    ) -> Optional[list[OpenAIFunctionToolCall]]:
        if not tool_calls:
            return None

        normalized_tool_calls = []
        for tool_call in tool_calls:
            if isinstance(tool_call, OpenAIFunctionToolCall):
                normalized_tool_calls.append(tool_call)
                continue

            tool_call_dict = deepcopy(tool_call)
            function_dict = tool_call_dict.get("function") or {}
            arguments = function_dict.get("arguments", {})
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Tool call arguments must be valid JSON: {arguments!r}") from exc
            if arguments is None:
                arguments = {}
            if not isinstance(arguments, dict):
                raise ValueError(f"Tool call arguments must decode to a dict, got {type(arguments)}")
            function_dict["arguments"] = arguments
            tool_call_dict["function"] = function_dict
            normalized_tool_calls.append(OpenAIFunctionToolCall.model_validate(tool_call_dict))
        return normalized_tool_calls

    @classmethod
    def _normalize_incoming_message(cls, message: dict[str, Any] | Message) -> dict[str, Any]:
        if isinstance(message, Message):
            role = message.role
            content = deepcopy(message.content)
            tool_calls = message.tool_calls
        else:
            role = message["role"]
            content = deepcopy(message.get("content", ""))
            tool_calls = message.get("tool_calls")

        if content is None:
            content = ""

        normalized_message: dict[str, Any] = {"role": role, "content": content}
        if tool_calls:
            normalized_message["tool_calls"] = cls._coerce_tool_calls(tool_calls)
        return normalized_message

    @staticmethod
    def _tool_response_from_content(content: Any) -> ToolResponse:
        if isinstance(content, ToolResponse):
            return content
        if content is None:
            return ToolResponse(text="")
        if isinstance(content, str):
            return ToolResponse(text=content)

        items = content if isinstance(content, list) else [content]
        text_parts: list[str] = []
        for item in items:
            if not isinstance(item, dict):
                raise ValueError(f"Unsupported tool message item type: {type(item)}")
            item_type = item.get("type")
            if item_type == "text":
                text_parts.append(item.get("text", ""))
                continue
            raise ValueError(
                "Tool message suffixes in the session wrapper currently support text-only content. "
                f"Got item type {item_type!r}."
            )
        return ToolResponse(text="".join(text_parts))

    @classmethod
    def _message_signature(cls, message: dict[str, Any] | Message) -> dict[str, Any]:
        normalized_message = cls._normalize_incoming_message(message)
        tool_calls = normalized_message.get("tool_calls") or []
        return {
            "role": normalized_message["role"],
            "content": normalized_message["content"],
            "tool_calls": [tool_call.model_dump() for tool_call in tool_calls],
        }
