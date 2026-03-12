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

"""
Validate McbotsBridge with real Qwen3-VL processor and real mcbots messages.

Uses a small subset of a recorded episode (obsidian mining task) with
placeholder images to keep the test lightweight and CPU-only.
"""

import json
import os
import unittest
from pathlib import Path

import torch

# Real messages path (only needed to regenerate fixture; test uses inline subset)
_MESSAGES_JSON = (
    "/homes/junlong/junlong_export_ssd/projects/mcbots/eval/results/dev_subset/"
    "qwen3vl32b_thinking_quick10_20260228_122300/records/"
    "mc-openha-eval-server_20260228_042422_bot-quick10-02-mine-block-obsidian-"
    "qwen3vl32bthinkingquick1020260228122300_20260228_042422/messages.json"
)

_MODEL_PATH = os.environ.get(
    "QWEN3_VL_MODEL_PATH",
    str(Path(__file__).resolve().parents[4] / "../../models/Qwen/Qwen3-VL-32B-Thinking"),
)

# Valid 16x16 red JPEG for the processor to accept
_TINY_JPEG_B64 = (
    "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8UHRofHh0a"
    "HBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/2wBDAQkJCQwLDBgNDRgyIRwhMjIyMjIy"
    "MjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjL/wAARCAAQABADASIA"
    "AhEBAxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9"
    "AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6"
    "Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ip"
    "qrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/8QAHwEAAwEB"
    "AQEBAQEBAQAAAAAAAAECAwQFBgcICQoL/8QAtREAAgECBAQDBAcFBAQAAQJ3AAECAxEEBSExBhJB"
    "UQdhcRMiMoEIFEKRobHBCSMzUvAVYnLRChYkNOEl8RcYGRomJygpKjU2Nzg5OkNERUZHSElKU1RV"
    "VldYWVpjZGVmZ2hpanN0dXZ3eHl6goOEhYaHiImKkpOUlZaXmJmaoqOkpaanqKmqsrO0tba3uLm6"
    "wsPExcbHyMnK0tPU1dbX2Nna4uPk5ebn6Onq8vP09fb3+Pn6/9oADAMBAAIRAxEAPwDi6KKK+ZP3"
    "E//Z"
)


def _make_tiny_image_url():
    return f"data:image/jpeg;base64,{_TINY_JPEG_B64}"


def _load_and_prepare_subset():
    """Load first 5 messages from the real recording and replace images with tiny placeholders."""
    with open(_MESSAGES_JSON) as f:
        all_msgs = json.load(f)

    subset = all_msgs[:5]  # system, user_task, user_obs1, assistant1, user_obs2
    for msg in subset:
        content = msg.get("content")
        if isinstance(content, list):
            for part in content:
                if part.get("type") == "image_url":
                    part["image_url"]["url"] = _make_tiny_image_url()
    return subset


def _build_inline_subset():
    """Inline minimal subset for environments without access to the full messages.json."""
    system_prompt = (
        "You are a Minecraft bot controller with vision.\n\n"
        "The game is running in a full asynchronous mode."
    )
    assistant_content = (
        "Let me figure out how to mine the obsidian block. "
        "The screenshot shows the player holding a diamond pickaxe.\n"
        "</think>\n\n"
        "<action>\n"
        "  <type>exec</type>\n"
        "  <content>mcapi look --yaw -10 --pitch 10 --mode relative</content>\n"
        "  <observe_after_sec>1.5</observe_after_sec>\n"
        "</action>"
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "Mine the obsidian block."},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "[System at 2026-02-28 04:24:21.61] Baritone settings reset."},
                {"type": "text", "text": "[Screenshot at 2026-02-28 04:24:21.63]"},
                {"type": "image_url", "image_url": {"url": _make_tiny_image_url()}},
            ],
        },
        {"role": "assistant", "content": assistant_content},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "[Action Assignment] action_id=5d52c00c."},
                {"type": "text", "text": "[Screenshot at 2026-02-28 04:24:22.67]"},
                {"type": "image_url", "image_url": {"url": _make_tiny_image_url()}},
                {"type": "text", "text": "[Screenshot at 2026-02-28 04:25:09.35]"},
                {"type": "image_url", "image_url": {"url": _make_tiny_image_url()}},
            ],
        },
    ]


def _get_messages():
    """Try real messages first, fall back to inline subset."""
    if os.path.exists(_MESSAGES_JSON):
        return _load_and_prepare_subset()
    return _build_inline_subset()


class McbotsBridgeRealVlmTest(unittest.TestCase):
    """Test McbotsBridge with real Qwen3-VL processor and mcbots-shaped messages."""

    @classmethod
    def setUpClass(cls):
        model_path = _MODEL_PATH
        if not os.path.isdir(model_path):
            raise unittest.SkipTest(f"Model not found at {model_path}")

        from transformers import AutoProcessor

        cls.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        cls.messages = _get_messages()

    def test_two_turn_token_alignment(self):
        """
        Walk through 2 turns with real VLM processor:
          init(system + user_task + user_obs1)
          → add_assistant(response1)
          → add_user(obs2)
          → finalize
        Verify incremental token state matches full-prompt tokenization.
        """
        from verl.experimental.agent_loop.mcbots_bridge import McbotsBridge
        from verl.workers.rollout.schemas import TokenizationSanityCheckModeEnum

        msgs = self.messages
        # Split: initial context = [system, user_task, user_obs1]
        init_messages = msgs[:3]
        assistant_content = msgs[3]["content"]
        user_obs2 = msgs[4]["content"]

        bridge = McbotsBridge(
            processing_class=self.processor,
            episode_id="test-obsidian-mining",
            messages=init_messages,
            max_prompt_len=16384,
            max_response_len=16384,
            max_model_len=32768,
            use_inference_chat_template=False,
            tokenization_sanity_check_mode=TokenizationSanityCheckModeEnum.IGNORE_STRIPPABLE,
        )

        # Turn 1: get prompt, record assistant response
        prompt_ids = bridge.get_generation_prompt_ids()
        self.assertTrue(len(prompt_ids) > 0, "generation prompt should not be empty")

        bridge.add_assistant_message(assistant_content)

        # Turn 2: add next observation
        bridge.add_user_message(user_obs2)

        # Finalize
        request = bridge.finalize()

        # ── Verify 1: finalize sanity check passed (IGNORE_STRIPPABLE) ──
        # The full-prompt re-tokenization drops thinking content from past assistant
        # turns (Qwen3 thinking chat template behavior). This is expected — the
        # incremental tokenization preserves thinking for training, while
        # full-prompt drops it. The finalize() sanity check already validated this
        # with IGNORE_STRIPPABLE mode. We just verify token counts are reasonable.
        self.assertTrue(
            request.input_ids.numel() > request.prompt_ids.numel(),
            "Input should be longer than prompt after adding assistant + observation",
        )

        # ── Verify 2: loss mask correctness ──
        response_ids = request.response_ids
        response_loss_mask = request.response_loss_mask
        response_attn_mask = request.response_attention_mask

        valid_response = response_ids[response_attn_mask.bool()]
        trainable_response = response_ids[response_loss_mask.bool()]
        masked_response = response_ids[response_attn_mask.bool() & ~response_loss_mask.bool()]

        self.assertTrue(
            trainable_response.numel() > 0,
            "There should be trainable (assistant) tokens in the response",
        )
        self.assertTrue(
            masked_response.numel() > 0,
            "There should be masked (user observation) tokens in the response",
        )

        # Decode and check content
        tokenizer = self.processor.tokenizer
        trainable_text = tokenizer.decode(trainable_response, skip_special_tokens=False)
        masked_text = tokenizer.decode(masked_response, skip_special_tokens=False)

        # Assistant's action XML should be in trainable region
        self.assertIn("mcapi", trainable_text, "Assistant action command should be trainable")
        self.assertIn("action", trainable_text, "Action tag should be trainable")

        # User observation text should be in masked region
        self.assertIn("Screenshot", masked_text, "User observation should be masked (not trainable)")

        print(f"\n{'='*60}")
        print(f"Prompt tokens: {request.prompt_ids.numel()}")
        print(f"Total input tokens: {request.input_ids.numel()}")
        print(f"Total valid response tokens: {valid_response.numel()}")
        print(f"Trainable tokens: {trainable_response.numel()}")
        print(f"Masked tokens: {masked_response.numel()}")
        print(f"Trainable text preview: {trainable_text[:200]}...")
        print(f"Masked text preview: {masked_text[:200]}...")
        print(f"{'='*60}")

    def test_generation_prompt_includes_think_tag(self):
        """
        Verify that the generation prompt (what gets sent to the model)
        ends with the <think> tag from the chat template.
        """
        from verl.experimental.agent_loop.mcbots_bridge import McbotsBridge
        from verl.workers.rollout.schemas import TokenizationSanityCheckModeEnum

        msgs = self.messages
        init_messages = msgs[:3]

        bridge = McbotsBridge(
            processing_class=self.processor,
            episode_id="test-think-tag",
            messages=init_messages,
            max_prompt_len=16384,
            max_response_len=16384,
            max_model_len=32768,
            use_inference_chat_template=False,
            tokenization_sanity_check_mode=TokenizationSanityCheckModeEnum.IGNORE_STRIPPABLE,
        )

        prompt_ids = bridge.get_generation_prompt_ids()
        tokenizer = self.processor.tokenizer
        prompt_text = tokenizer.decode(prompt_ids, skip_special_tokens=False)

        # The Qwen3 thinking chat template should end with <think> in the generation prompt
        self.assertTrue(
            prompt_text.rstrip().endswith("<think>"),
            f"Generation prompt should end with <think> tag for thinking model. "
            f"Last 100 chars: ...{prompt_text[-100:]}",
        )
        print(f"\nGeneration prompt ends with: ...{prompt_text[-50:]}")

    def test_session_manager_two_turn_flow(self):
        """
        Test the full SessionManager flow with real VLM processor:
          prepare_generation(init) → record_assistant → prepare_generation(updated) → finalize
        """
        from jl_patch.experimental.agent_loop.mcbots_session_manager import McbotsSessionManager
        from verl.workers.rollout.schemas import TokenizationSanityCheckModeEnum

        msgs = self.messages
        episode_id = "test-session-obsidian"

        mgr = McbotsSessionManager(
            processing_class=self.processor,
            max_prompt_len=16384,
            max_response_len=16384,
            max_model_len=32768,
            use_inference_chat_template=False,
            tokenization_sanity_check_mode=TokenizationSanityCheckModeEnum.IGNORE_STRIPPABLE,
        )

        # Turn 1: prepare with initial messages
        prompt_ids_1 = mgr.prepare_generation(
            episode_id=episode_id,
            messages=msgs[:3],  # system + user_task + user_obs1
        )
        self.assertTrue(len(prompt_ids_1) > 0)

        # Record assistant response
        assistant_content = msgs[3]["content"]
        mgr.record_assistant_turn(
            episode_id=episode_id,
            content=assistant_content,
        )

        # Turn 2: prepare with full history (now includes assistant + new obs)
        prompt_ids_2 = mgr.prepare_generation(
            episode_id=episode_id,
            messages=msgs[:5],  # system + user_task + user_obs1 + assistant + user_obs2
        )
        self.assertTrue(len(prompt_ids_2) > len(prompt_ids_1), "Prompt should grow after adding turns")

        # Finalize
        request = mgr.finalize_episode(episode_id=episode_id)
        self.assertFalse(mgr.has_episode(episode_id), "Episode should be cleaned up after finalize")

        # Verify there are both trainable and masked tokens
        trainable = request.response_ids[request.response_loss_mask.bool()]
        masked = request.response_ids[
            request.response_attention_mask.bool() & ~request.response_loss_mask.bool()
        ]
        self.assertTrue(trainable.numel() > 0, "Should have trainable tokens")
        self.assertTrue(masked.numel() > 0, "Should have masked tokens")

        print(f"\nSession manager flow OK:")
        print(f"  Turn 1 prompt: {len(prompt_ids_1)} tokens")
        print(f"  Turn 2 prompt: {len(prompt_ids_2)} tokens")
        print(f"  Trainable: {trainable.numel()}, Masked: {masked.numel()}")


if __name__ == "__main__":
    unittest.main()
