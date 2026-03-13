import unittest

import torch
from transformers import PreTrainedTokenizer

from jl_patch.experimental.agent_loop.mcbots_session_manager import McbotsSessionManager
from verl.workers.rollout.schemas import AsyncRolloutRequestStateEnum


class ToyChatTokenizer(PreTrainedTokenizer):
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


class McbotsSessionManagerCpuTest(unittest.TestCase):
    def setUp(self):
        self.tokenizer = ToyChatTokenizer()
        self.manager = McbotsSessionManager(
            processing_class=self.tokenizer,
            max_prompt_len=512,
            max_response_len=512,
            max_model_len=1024,
        )
        self.initial_messages = [
            {"role": "system", "content": "You are a Minecraft assistant."},
            {"role": "user", "content": "Collect one wood block."},
        ]

    def test_reuses_episode_state_and_syncs_user_suffix(self):
        self.manager.prepare_generation(episode_id="episode-1", messages=self.initial_messages)
        tool_calls = [
            {
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "mine_block",
                    "arguments": '{"target": "oak_log", "count": 1}',
                },
            }
        ]
        self.manager.record_assistant_turn(
            episode_id="episode-1",
            content="<action><type>exec</type><content>mine tree</content></action>",
            tool_calls=tool_calls,
        )

        next_messages = [
            *self.initial_messages,
            {
                "role": "assistant",
                "content": "<action><type>exec</type><content>mine tree</content></action>",
                "tool_calls": tool_calls,
            },
            {"role": "user", "content": "Observation: inventory now contains 1 oak log."},
        ]
        prompt_ids = self.manager.prepare_generation(episode_id="episode-1", messages=next_messages)
        self.assertTrue(prompt_ids)

        rollouts = self.manager.finalize_episode(episode_id="episode-1")
        self.assertEqual(len(rollouts), 1)
        request = rollouts[0]
        self.assertEqual(request.messages[2].tool_calls[0].function.arguments, {"target": "oak_log", "count": 1})
        self.assertEqual(request.messages[-1].content, "Observation: inventory now contains 1 oak log.")

        full_prompt_ids = request._handle_apply_chat_template(
            self.tokenizer,
            [msg.model_dump() for msg in request.messages],
            multi_modal_data=request.multi_modal_data,
            tools=None,
            add_generation_prompt=False,
            tokenize=True,
        )
        self.assertTrue(full_prompt_ids.eq(request.input_ids).all())

    def test_requires_context_reset_for_diverged_history(self):
        self.manager.prepare_generation(episode_id="episode-1", messages=self.initial_messages)
        self.manager.record_assistant_turn(
            episode_id="episode-1",
            content="<action><type>exec</type><content>mine tree</content></action>",
        )

        with self.assertRaisesRegex(ValueError, "context_reset=True"):
            self.manager.prepare_generation(
                episode_id="episode-1",
                messages=[
                    {"role": "system", "content": "You are a Minecraft assistant."},
                    {"role": "user", "content": "Collect one dirt block instead."},
                ],
            )

    def test_context_reset_rebuilds_episode_state(self):
        self.manager.prepare_generation(episode_id="episode-1", messages=self.initial_messages)
        self.manager.record_assistant_turn(
            episode_id="episode-1",
            content="<action><type>exec</type><content>mine tree</content></action>",
        )

        reset_messages = [
            {"role": "system", "content": "You are a Minecraft assistant."},
            {"role": "user", "content": "Collect one dirt block instead."},
        ]
        self.manager.prepare_generation(
            episode_id="episode-1",
            messages=reset_messages,
            context_reset=True,
        )

        bridge = self.manager.get_bridge("episode-1")
        self.assertEqual([message.role for message in bridge.request.messages], ["system", "user"])
        self.assertEqual(bridge.request.messages[-1].content, "Collect one dirt block instead.")

    def test_context_reset_preserves_prior_rollout(self):
        # Segment 1: init → assistant
        self.manager.prepare_generation(episode_id="episode-1", messages=self.initial_messages)
        self.manager.record_assistant_turn(
            episode_id="episode-1",
            content="<action><type>exec</type><content>mine tree</content></action>",
        )

        # Context reset → segment 2
        reset_messages = [
            {"role": "system", "content": "You are a Minecraft assistant."},
            {"role": "user", "content": "Summary of previous context. Now collect dirt."},
        ]
        self.manager.prepare_generation(
            episode_id="episode-1",
            messages=reset_messages,
            context_reset=True,
        )
        self.manager.record_assistant_turn(
            episode_id="episode-1",
            content="<action><type>exec</type><content>mine dirt</content></action>",
        )

        # Finalize → should get 2 rollouts
        rollouts = self.manager.finalize_episode(episode_id="episode-1")
        self.assertEqual(len(rollouts), 2, "Should produce 2 rollouts (1 per segment)")

        # Both should be completed
        self.assertEqual(rollouts[0].state, AsyncRolloutRequestStateEnum.COMPLETED)
        self.assertEqual(rollouts[1].state, AsyncRolloutRequestStateEnum.COMPLETED)

        # Segment 1 should contain "mine tree", segment 2 should contain "mine dirt"
        tokenizer = self.manager.processing_class
        seg1_text = tokenizer.decode(rollouts[0].input_ids[0], skip_special_tokens=True)
        seg2_text = tokenizer.decode(rollouts[1].input_ids[0], skip_special_tokens=True)
        self.assertIn("mine tree", seg1_text)
        self.assertIn("mine dirt", seg2_text)

        self.assertFalse(self.manager.has_episode("episode-1"))

    def test_is_final_finalizes_and_cleans_up_episode(self):
        self.manager.prepare_generation(episode_id="episode-1", messages=self.initial_messages)
        rollouts = self.manager.record_assistant_turn(
            episode_id="episode-1",
            content="<action><type>done</type><content>finished</content></action>",
            is_final=True,
        )

        self.assertIsNotNone(rollouts)
        self.assertEqual(len(rollouts), 1)
        self.assertEqual(rollouts[0].state, AsyncRolloutRequestStateEnum.COMPLETED)
        self.assertFalse(self.manager.has_episode("episode-1"))

    def test_rejects_untracked_assistant_suffix_for_active_episode(self):
        self.manager.prepare_generation(episode_id="episode-1", messages=self.initial_messages)
        with self.assertRaisesRegex(ValueError, "record_assistant_turn"):
            self.manager.prepare_generation(
                episode_id="episode-1",
                messages=[
                    *self.initial_messages,
                    {
                        "role": "assistant",
                        "content": "<action><type>exec</type><content>mine tree</content></action>",
                    },
                ],
            )


if __name__ == "__main__":
    unittest.main()
