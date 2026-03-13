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

"""OpenAI-compatible gateway that translates between mcbots HTTP requests and verl's token-level generate interface.

Usage::

    gateway = McbotsGateway(
        session_manager=session_manager,
        generate_fn=rollout_engine.generate,
        processing_class=processor,
    )
    base_url = gateway.start(port=0)  # auto-pick free port
    # mcbots: MCBOTS_BASE_URL=base_url

    # ... training loop ...
    rollouts = gateway.collect_rollouts()

    gateway.shutdown()
"""

from __future__ import annotations

import logging
import socket
import threading
import time
import uuid
from typing import Any, Awaitable, Callable, Optional

import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, Field

from jl_patch.experimental.agent_loop.mcbots_session_manager import McbotsSessionManager
from verl.workers.rollout.replica import TokenOutput
from verl.workers.rollout.schemas import AsyncRolloutRequest

logger = logging.getLogger(__name__)


# ── OpenAI-compatible request/response models ──


class ChatMessage(BaseModel):
    role: str
    content: Any = ""
    tool_calls: Optional[list[dict[str, Any]]] = None
    tool_call_id: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    model: str = ""
    messages: list[ChatMessage]
    temperature: float = 1.0
    top_p: float = 1.0
    max_tokens: Optional[int] = None
    stop: Optional[list[str] | str] = None
    # mcbots-specific fields via extra_body
    episode_id: Optional[str] = None
    context_reset: bool = False
    is_final: bool = False
    reward_scores: Optional[dict[str, list[float]]] = None

    model_config = {"extra": "allow"}


class ChatCompletionChoice(BaseModel):
    index: int = 0
    message: ChatMessage
    finish_reason: Optional[str] = "stop"


class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:12]}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str = ""
    choices: list[ChatCompletionChoice]
    usage: UsageInfo = Field(default_factory=UsageInfo)


# ── Generate function protocol ──

GenerateFn = Callable[
    [list[int], dict[str, Any], str, Optional[list[Any]], Optional[list[Any]], int],
    Awaitable[TokenOutput],
]


# ── Gateway ──


class McbotsGateway:
    """Thin translation layer: OpenAI chat completions <-> verl token generate."""

    def __init__(
        self,
        session_manager: McbotsSessionManager,
        generate_fn: GenerateFn,
        processing_class: Any,
        *,
        default_sampling_params: Optional[dict[str, Any]] = None,
    ) -> None:
        self.session_manager = session_manager
        self.generate_fn = generate_fn
        self.processing_class = processing_class
        self.default_sampling_params = default_sampling_params or {}
        self._pending_rollouts: dict[str, list[AsyncRolloutRequest]] = {}
        self.app = self._build_app()

    def start(self, host: str = "0.0.0.0", port: int = 0) -> str:
        """Start the gateway in a background thread and return the base_url.

        Args:
            host: Bind address. Defaults to all interfaces.
            port: Port to listen on. ``0`` means pick a free port automatically.

        Returns:
            The base_url string, e.g. ``"http://hostname:12345/v1"``.
        """
        if hasattr(self, "_server_thread") and self._server_thread.is_alive():
            raise RuntimeError("Gateway is already running")

        if port == 0:
            port = self._find_free_port()

        config = uvicorn.Config(self.app, host=host, port=port, log_level="warning")
        self._server = uvicorn.Server(config)

        self._server_thread = threading.Thread(target=self._server.run, daemon=True)
        self._server_thread.start()

        # Wait until uvicorn is fully ready (not just TCP-bound)
        self._wait_until_ready(self._server)

        hostname = socket.getfqdn() if host == "0.0.0.0" else host
        self.base_url = f"http://{hostname}:{port}/v1"
        logger.info(f"McbotsGateway serving at {self.base_url}")
        return self.base_url

    def shutdown(self) -> None:
        """Signal the server to stop."""
        if hasattr(self, "_server"):
            self._server.should_exit = True
            self._server_thread.join(timeout=5)

    @staticmethod
    def _find_free_port() -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            return s.getsockname()[1]

    @staticmethod
    def _wait_until_ready(server: uvicorn.Server, timeout: float = 10.0) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if server.started:
                return
            time.sleep(0.05)
        raise TimeoutError(f"Gateway did not become ready within {timeout}s")

    def collect_rollouts(self) -> dict[str, list[AsyncRolloutRequest]]:
        """Drain and return all completed rollouts, keyed by episode_id."""
        rollouts = self._pending_rollouts
        self._pending_rollouts = {}
        return rollouts

    def _build_app(self) -> FastAPI:
        app = FastAPI()

        @app.get("/v1/health")
        async def health():
            return {"status": "ok"}

        @app.post("/v1/chat/completions")
        async def chat_completions(request: ChatCompletionRequest) -> ChatCompletionResponse:
            return await self._handle_chat_completions(request)

        return app

    async def _handle_chat_completions(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        episode_id = request.episode_id or uuid.uuid4().hex
        messages = [msg.model_dump(exclude_none=True) for msg in request.messages]

        # 1. Messages → prompt_ids
        prompt_ids = self.session_manager.prepare_generation(
            episode_id=episode_id,
            messages=messages,
            context_reset=request.context_reset,
        )

        # 2. Prepare multi-modal data for generate
        bridge = self.session_manager.get_bridge(episode_id)
        mm_data = bridge.request.multi_modal_data or {}
        image_data = mm_data.get("image")
        video_data = mm_data.get("video")

        # 3. Build sampling params
        sampling_params = dict(self.default_sampling_params)
        if request.temperature is not None:
            sampling_params["temperature"] = request.temperature
        if request.top_p is not None:
            sampling_params["top_p"] = request.top_p
        if request.max_tokens is not None:
            sampling_params["max_tokens"] = request.max_tokens
        if request.stop is not None:
            sampling_params["stop"] = request.stop if isinstance(request.stop, list) else [request.stop]

        # 4. Generate
        request_id = uuid.uuid4().hex
        token_output = await self.generate_fn(
            prompt_ids,
            sampling_params,
            request_id,
            image_data,
            video_data,
            0,  # priority
        )

        # 5. Decode
        generated_ids = token_output.token_ids
        tokenizer = getattr(self.processing_class, "tokenizer", self.processing_class)
        content = tokenizer.decode(generated_ids, skip_special_tokens=True)

        # 6. Record assistant turn (content_ids must be a 2-d tensor matching input_ids shape)
        content_ids_tensor = torch.tensor([generated_ids], dtype=torch.long)
        rollouts = self.session_manager.record_assistant_turn(
            episode_id=episode_id,
            content=content,
            content_ids=content_ids_tensor,
            is_final=request.is_final,
            reward_scores=request.reward_scores,
        )

        # 7. Store completed rollouts
        if rollouts is not None:
            self._pending_rollouts.setdefault(episode_id, []).extend(rollouts)

        # 8. Build response (verl stop_reason is "completed"/"aborted", always map to "stop" for OpenAI)
        return ChatCompletionResponse(
            model=request.model,
            choices=[
                ChatCompletionChoice(
                    message=ChatMessage(role="assistant", content=content),
                    finish_reason="stop",
                )
            ],
            usage=UsageInfo(
                prompt_tokens=len(prompt_ids),
                completion_tokens=len(generated_ids),
                total_tokens=len(prompt_ids) + len(generated_ids),
            ),
        )
