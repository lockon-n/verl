import asyncio
import unittest
from typing import Any, Optional

import torch
from transformers import PreTrainedTokenizer

from jl_patch.experimental.agent_loop.mcbots_gateway import (
    ChatCompletionRequest,
    ChatMessage,
    McbotsGateway,
)
from jl_patch.experimental.agent_loop.mcbots_session_manager import McbotsSessionManager
from verl.workers.rollout.replica import TokenOutput


class ToyChatTokenizer(PreTrainedTokenizer):
    """Minimal tokenizer for CPU tests — same as in test_mcbots_session_manager_on_cpu."""

    def __init__(self):
        self._vocab = {"<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3}
        self._id_to_token = {idx: token for token, idx in self._vocab.items()}
        super().__init__(pad_token="<pad>", bos_token="<bos>", eos_token="<eos>", unk_token="<unk>")

    @property
    def vocab_size(self) -> int:
        return len(self._vocab)

    def get_vocab(self):
        return dict(self._vocab)

    def _tokenize(self, text, **kwargs):
        return list(text)

    def _convert_token_to_id(self, token):
        if token not in self._vocab:
            token_id = len(self._vocab)
            self._vocab[token] = token_id
            self._id_to_token[token_id] = token
        return self._vocab[token]

    def _convert_id_to_token(self, index):
        return self._id_to_token.get(index, self.unk_token)

    def convert_tokens_to_string(self, tokens):
        return "".join(tokens)

    def save_vocabulary(self, save_directory, filename_prefix=None):
        return ()

    def _encode_text(self, text: str) -> list[int]:
        return [self._convert_token_to_id(token) for token in self._tokenize(text)]

    def __call__(self, text=None, return_tensors=None, **kwargs):
        if isinstance(text, str):
            texts = [text]
        else:
            texts = list(text or [])
        encoded = [self._encode_text(item) for item in texts]
        max_length = max((len(item) for item in encoded), default=0)
        input_ids = []
        attention_mask = []
        for item in encoded:
            pad_size = max_length - len(item)
            input_ids.append(item + [self.pad_token_id] * pad_size)
            attention_mask.append([1] * len(item) + [0] * pad_size)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }

    def apply_chat_template(self, messages, tools=None, add_generation_prompt=False, tokenize=False, **kwargs):
        def format_content(content):
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                parts = []
                for item in content:
                    if item.get("type") == "text":
                        parts.append(item["text"])
                    else:
                        parts.append(f"[{item.get('type', 'item')}]")
                return "".join(parts)
            return str(content)

        def get_message_field(message, field):
            if isinstance(message, dict):
                return message[field]
            return getattr(message, field)

        rendered = "".join(
            f"<{get_message_field(msg, 'role')}>{format_content(get_message_field(msg, 'content'))}</{get_message_field(msg, 'role')}>"
            for msg in messages
        )
        if add_generation_prompt:
            rendered += "<assistant>"
        if tokenize:
            return self(rendered, return_tensors="pt")["input_ids"]
        return rendered


def _make_generate_fn(tokenizer: ToyChatTokenizer):
    """Return a mock generate_fn that echoes a fixed response."""

    async def generate_fn(
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        request_id: str,
        image_data: Optional[list[Any]] = None,
        video_data: Optional[list[Any]] = None,
        priority: int = 0,
    ) -> TokenOutput:
        response_text = "I will mine the block."
        token_ids = tokenizer._encode_text(response_text)
        return TokenOutput(token_ids=token_ids, stop_reason="completed")

    return generate_fn


class McbotsGatewayCpuTest(unittest.TestCase):
    def setUp(self):
        self.tokenizer = ToyChatTokenizer()
        self.session_manager = McbotsSessionManager(
            processing_class=self.tokenizer,
            max_prompt_len=512,
            max_response_len=512,
            max_model_len=1024,
        )
        self.generate_fn = _make_generate_fn(self.tokenizer)
        self.gateway = McbotsGateway(
            session_manager=self.session_manager,
            generate_fn=self.generate_fn,
            processing_class=self.tokenizer,
        )

    def _run(self, coro):
        return asyncio.get_event_loop().run_until_complete(coro)

    def test_single_turn(self):
        request = ChatCompletionRequest(
            messages=[
                ChatMessage(role="system", content="You are a Minecraft bot."),
                ChatMessage(role="user", content="Mine obsidian."),
            ],
            episode_id="ep-1",
        )
        response = self._run(self.gateway._handle_chat_completions(request))

        self.assertEqual(len(response.choices), 1)
        self.assertEqual(response.choices[0].message.role, "assistant")
        self.assertEqual(response.choices[0].message.content, "I will mine the block.")
        self.assertEqual(response.choices[0].finish_reason, "stop")
        self.assertGreater(response.usage.prompt_tokens, 0)
        self.assertGreater(response.usage.completion_tokens, 0)

        # Episode should still be active (not finalized)
        self.assertTrue(self.session_manager.has_episode("ep-1"))

    def test_multi_turn_then_finalize(self):
        # Turn 1
        req1 = ChatCompletionRequest(
            messages=[
                ChatMessage(role="system", content="You are a Minecraft bot."),
                ChatMessage(role="user", content="Mine obsidian."),
            ],
            episode_id="ep-1",
        )
        resp1 = self._run(self.gateway._handle_chat_completions(req1))
        self.assertEqual(resp1.choices[0].message.content, "I will mine the block.")

        # Turn 2 with is_final=True
        req2 = ChatCompletionRequest(
            messages=[
                ChatMessage(role="system", content="You are a Minecraft bot."),
                ChatMessage(role="user", content="Mine obsidian."),
                ChatMessage(role="assistant", content="I will mine the block."),
                ChatMessage(role="user", content="Observation: block mined successfully."),
            ],
            episode_id="ep-1",
            is_final=True,
        )
        resp2 = self._run(self.gateway._handle_chat_completions(req2))
        self.assertEqual(resp2.choices[0].message.content, "I will mine the block.")

        # Episode should be finalized
        self.assertFalse(self.session_manager.has_episode("ep-1"))

        # Rollouts should be collected
        rollouts = self.gateway.collect_rollouts()
        self.assertIn("ep-1", rollouts)
        self.assertEqual(len(rollouts["ep-1"]), 1)

    def test_context_reset_produces_multiple_rollouts(self):
        # Turn 1
        req1 = ChatCompletionRequest(
            messages=[
                ChatMessage(role="system", content="You are a Minecraft bot."),
                ChatMessage(role="user", content="Mine obsidian."),
            ],
            episode_id="ep-1",
        )
        self._run(self.gateway._handle_chat_completions(req1))

        # Turn 2 with context_reset
        req2 = ChatCompletionRequest(
            messages=[
                ChatMessage(role="system", content="You are a Minecraft bot."),
                ChatMessage(role="user", content="Summary of previous context. Now mine diamond."),
            ],
            episode_id="ep-1",
            context_reset=True,
        )
        self._run(self.gateway._handle_chat_completions(req2))

        # Finalize
        req3 = ChatCompletionRequest(
            messages=[
                ChatMessage(role="system", content="You are a Minecraft bot."),
                ChatMessage(role="user", content="Summary of previous context. Now mine diamond."),
                ChatMessage(role="assistant", content="I will mine the block."),
                ChatMessage(role="user", content="Observation: diamond found."),
            ],
            episode_id="ep-1",
            is_final=True,
        )
        self._run(self.gateway._handle_chat_completions(req3))

        rollouts = self.gateway.collect_rollouts()
        self.assertIn("ep-1", rollouts)
        self.assertEqual(len(rollouts["ep-1"]), 2, "Should have 2 rollouts: pre-reset + post-reset")

    def test_collect_rollouts_drains(self):
        req = ChatCompletionRequest(
            messages=[
                ChatMessage(role="system", content="You are a bot."),
                ChatMessage(role="user", content="Do something."),
            ],
            episode_id="ep-drain",
            is_final=True,
        )
        self._run(self.gateway._handle_chat_completions(req))

        rollouts = self.gateway.collect_rollouts()
        self.assertIn("ep-drain", rollouts)

        # Second call should be empty
        rollouts2 = self.gateway.collect_rollouts()
        self.assertEqual(len(rollouts2), 0)

    def test_auto_episode_id_when_not_provided(self):
        req = ChatCompletionRequest(
            messages=[
                ChatMessage(role="system", content="You are a bot."),
                ChatMessage(role="user", content="Hello."),
            ],
            # No episode_id provided
        )
        response = self._run(self.gateway._handle_chat_completions(req))
        self.assertEqual(response.choices[0].message.content, "I will mine the block.")

    def test_fastapi_app_has_endpoint(self):
        routes = [route.path for route in self.gateway.app.routes]
        self.assertIn("/v1/chat/completions", routes)

    def test_start_and_shutdown(self):
        base_url = self.gateway.start(host="127.0.0.1")
        try:
            self.assertTrue(base_url.startswith("http://"))
            self.assertTrue(base_url.endswith("/v1"))
            self.assertEqual(self.gateway.base_url, base_url)

            # Actually call the endpoint over HTTP (bypass any local proxy)
            import requests

            session = requests.Session()
            session.trust_env = False
            resp = session.post(
                f"{base_url}/chat/completions",
                json={
                    "messages": [
                        {"role": "system", "content": "You are a bot."},
                        {"role": "user", "content": "Hello."},
                    ],
                    "episode_id": "ep-http",
                    "is_final": True,
                },
            )
            self.assertEqual(resp.status_code, 200, f"Response body: {resp.text}")
            data = resp.json()
            self.assertEqual(data["choices"][0]["message"]["content"], "I will mine the block.")

            rollouts = self.gateway.collect_rollouts()
            self.assertIn("ep-http", rollouts)
        finally:
            self.gateway.shutdown()


if __name__ == "__main__":
    unittest.main()
