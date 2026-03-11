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

import unittest

import torch
from transformers import PreTrainedTokenizer

from verl.experimental.agent_loop.mcbots_bridge import McbotsBridge


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


class McbotsBridgeCpuTest(unittest.TestCase):
    def test_aligns_incremental_messages_on_cpu(self):
        tokenizer = ToyChatTokenizer()
        bridge = McbotsBridge(
            processing_class=tokenizer,
            episode_id="episode-1",
            messages=[
                {"role": "system", "content": "You are a Minecraft assistant."},
                {"role": "user", "content": "Collect one wood block."},
            ],
            max_prompt_len=512,
            max_response_len=512,
            max_model_len=1024,
        )

        prompt_ids = bridge.get_generation_prompt_ids()
        self.assertTrue(prompt_ids, "generation prompt ids should not be empty")

        bridge.add_assistant_message("<action><type>exec</type><content>mine tree</content></action>")
        bridge.add_user_message("Observation: inventory now contains 1 oak log.")

        request = bridge.finalize()

        messages = [msg.model_dump() for msg in request.messages]
        full_prompt_ids = request._handle_apply_chat_template(
            tokenizer,
            messages,
            multi_modal_data=request.multi_modal_data,
            tools=None,
            add_generation_prompt=False,
            tokenize=True,
        )
        self.assertTrue(full_prompt_ids.eq(request.input_ids).all())

        valid_response = request.response_ids[request.response_attention_mask.bool()]
        trainable_response = request.response_ids[request.response_loss_mask.bool()]
        masked_response = request.response_ids[request.response_attention_mask.bool() & ~request.response_loss_mask.bool()]

        valid_text = tokenizer.decode(valid_response, skip_special_tokens=False)
        trainable_text = tokenizer.decode(trainable_response, skip_special_tokens=False)
        masked_text = tokenizer.decode(masked_response, skip_special_tokens=False)

        self.assertIn("mine tree", valid_text)
        self.assertIn("inventory now contains 1 oak log", valid_text)
        self.assertIn("mine tree", trainable_text)
        self.assertNotIn("inventory now contains 1 oak log", trainable_text)
        self.assertIn("inventory now contains 1 oak log", masked_text)


if __name__ == "__main__":
    unittest.main()
