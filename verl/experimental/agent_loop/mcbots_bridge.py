# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from copy import deepcopy
from typing import Any, Optional
from uuid import uuid4

from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast, ProcessorMixin

from verl.tools.schemas import OpenAIFunctionToolCall, ToolResponse
from verl.workers.rollout.schemas import (
    AsyncRolloutRequest,
    AsyncRolloutRequestStateEnum,
    BASE_CHAT_HISTORY,
    FinishReasonTypeEnum,
    Message,
    TokenizationSanityCheckModeEnum,
)


class McbotsBridge:
    """Minimal bridge that keeps mcbots-style messages aligned with verl token state."""

    def __init__(
        self,
        processing_class: PreTrainedTokenizer | PreTrainedTokenizerFast | ProcessorMixin,
        messages: list[dict[str, Any]],
        *,
        episode_id: str,
        request_id: Optional[str] = None,
        multi_modal_keys: Optional[list[str]] = None,
        multi_modal_data: Optional[dict[str, Any]] = None,
        max_prompt_len: int = 8192,
        max_response_len: int = 8192,
        max_model_len: int = 32768,
        use_inference_chat_template: bool = False,
        tokenization_sanity_check_mode: TokenizationSanityCheckModeEnum = TokenizationSanityCheckModeEnum.STRICT,
    ) -> None:
        self.processing_class = processing_class
        self.episode_id = episode_id
        # Deepcopy to prevent processor/chat-template from mutating caller's dicts
        messages = deepcopy(messages)
        extracted_multi_modal_data = multi_modal_data
        if extracted_multi_modal_data is None:
            extracted_multi_modal_data = self._extract_multi_modal_data(messages)
        self.request = AsyncRolloutRequest(
            request_id=request_id or uuid4().hex,
            state=AsyncRolloutRequestStateEnum.PENDING,
            messages=messages,
            multi_modal_keys=multi_modal_keys,
            multi_modal_data=extracted_multi_modal_data,
            reward_scores={},
            max_prompt_len=max_prompt_len,
            max_response_len=max_response_len,
            max_model_len=max_model_len,
            metrics={},
            use_inference_chat_template=use_inference_chat_template,
            tokenization_sanity_check_mode=tokenization_sanity_check_mode,
            processing_class=processing_class,
        )

    def get_generation_prompt_ids(self) -> list[int]:
        return self.request.get_generation_prompt_ids(self.processing_class)

    def add_user_message(self, content: str | list[dict[str, Any]] | dict[str, Any]) -> None:
        if not self._content_has_multi_modal_payload(content):
            self.request.add_user_message(self.processing_class, content=content)
            return

        content_list = content if isinstance(content, list) else [content]
        delta_message = Message(role="user", content=content_list)
        delta_multi_modal_data = self._extract_multi_modal_data([delta_message.model_dump()])

        self.request.messages.append(delta_message)
        tools = [tool.model_dump() for tool in self.request.tool_schemas] if self.request.tool_schemas else None

        for key in self.request.multi_modal_keys:
            if delta_multi_modal_data.get(key):
                self.request.multi_modal_data[key].extend(delta_multi_modal_data[key])

        content_info = self.request._handle_apply_chat_template(
            self.processing_class,
            [*BASE_CHAT_HISTORY, delta_message],
            multi_modal_data=delta_multi_modal_data,
            tools=tools,
            add_generation_prompt=False,
            tokenize=True,
            return_dict=True,
        )
        content_ids = content_info["input_ids"][..., self.request.base_conv_wo_gen_prompt_end_pos :]

        from verl.workers.rollout.schemas import _PROCESSOR_PER_TOKEN_KEYS

        multi_modal_inputs = {k: v for k, v in content_info.items() if k not in _PROCESSOR_PER_TOKEN_KEYS}

        self.request._remove_generation_prompt_ids_if_present()
        self.request._update_input_ids(
            self.processing_class,
            content_ids,
            attention_mask=True,
            loss_mask=False,
            new_multi_modal_inputs=multi_modal_inputs,
        )

    def add_assistant_message(
        self,
        content: str,
        *,
        content_ids=None,
        tool_calls: Optional[list[OpenAIFunctionToolCall]] = None,
    ) -> None:
        self.request.add_assistant_message(
            self.processing_class,
            content=content,
            content_ids=content_ids,
            tool_calls=tool_calls,
        )

    def add_tool_response_messages(self, contents: list[ToolResponse]) -> None:
        self.request.add_tool_response_messages(self.processing_class, contents)

    def finalize(
        self,
        reward_scores: Optional[dict[str, list[float]]] = None,
        finish_reason_type: FinishReasonTypeEnum = FinishReasonTypeEnum.STOP,
    ) -> AsyncRolloutRequest:
        self.request.finalize(
            self.processing_class,
            reward_scores=reward_scores or {},
            finish_reason_type=finish_reason_type,
        )
        return self.request

    def _extract_multi_modal_data(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        if not isinstance(self.processing_class, ProcessorMixin):
            return {}
        if not any(self._message_has_multi_modal_payload(message) for message in messages):
            return {}

        from qwen_vl_utils import process_vision_info

        normalized = self._normalize_image_url_format(messages)
        images, videos = process_vision_info(normalized, return_video_metadata=True)
        multi_modal_data = {}
        if images:
            multi_modal_data["image"] = images
        if videos:
            multi_modal_data["video"] = videos
        return multi_modal_data

    @staticmethod
    def _normalize_image_url_format(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Flatten OpenAI nested image_url dicts for qwen_vl_utils compatibility.

        OpenAI format:  {"type": "image_url", "image_url": {"url": "data:..."}}
        qwen_vl_utils:  {"type": "image_url", "image_url": "data:..."}
        """
        out = []
        for msg in messages:
            content = msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", None)
            if not isinstance(content, list):
                out.append(msg)
                continue
            new_content = []
            for part in content:
                if (
                    isinstance(part, dict)
                    and part.get("type") == "image_url"
                    and isinstance(part.get("image_url"), dict)
                ):
                    new_content.append({"type": "image_url", "image_url": part["image_url"]["url"]})
                else:
                    new_content.append(part)
            out.append({**msg, "content": new_content})
        return out

    @staticmethod
    def _content_has_multi_modal_payload(content: Any) -> bool:
        if isinstance(content, dict):
            return content.get("type") in {"image", "image_url", "video"}
        if isinstance(content, list):
            return any(McbotsBridge._content_has_multi_modal_payload(item) for item in content)
        return False

    @classmethod
    def _message_has_multi_modal_payload(cls, message: dict[str, Any]) -> bool:
        content = message.get("content") if isinstance(message, dict) else getattr(message, "content", None)
        return cls._content_has_multi_modal_payload(content)
