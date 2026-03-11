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

from typing import Any, Optional
from uuid import uuid4

from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast, ProcessorMixin

from verl.tools.schemas import OpenAIFunctionToolCall, ToolResponse
from verl.workers.rollout.schemas import (
    AsyncRolloutRequest,
    AsyncRolloutRequestStateEnum,
    FinishReasonTypeEnum,
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
        max_prompt_len: int = 8192,
        max_response_len: int = 8192,
        max_model_len: int = 32768,
        use_inference_chat_template: bool = True,
        tokenization_sanity_check_mode: TokenizationSanityCheckModeEnum = TokenizationSanityCheckModeEnum.STRICT,
    ) -> None:
        self.processing_class = processing_class
        self.episode_id = episode_id
        self.request = AsyncRolloutRequest(
            request_id=request_id or uuid4().hex,
            state=AsyncRolloutRequestStateEnum.PENDING,
            messages=messages,
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

    def add_user_message(self, content: str) -> None:
        self.request.add_user_message(self.processing_class, content=content)

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
