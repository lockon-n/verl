# mcbots <-> verl Handoff

## Goal

Integrate `/Users/bytedance/Projects/mcbots` into `verl` training while reusing the existing `mcbots` agent loop as much as possible.

Target split of responsibilities:

- `mcbots` keeps the agent/runtime logic:
  - environment interaction
  - message assembly
  - action parsing
  - action execution
  - deciding when an episode continues or ends
- `verl` owns token-level concerns:
  - chat template application
  - token-in/token-out generation
  - prompt/response/loss mask assembly
  - later reward/teacher labeling/training

## Key design decisions already made

1. Do **not** rewrite `mcbots` into a native `verl` agent loop first.
2. Do **not** treat plain OpenAI text responses as the training source of truth and retokenize later.
3. Keep chat-template ownership inside `verl`.
4. Use `AsyncRolloutRequest` as the core incremental token state object.
5. Start with a very small bridge and validate message-to-token alignment before touching gateway/Ray/trainer.

## Why this design

The main risk is not networking or Ray. The main risk is token alignment:

- assistant/model tokens must be trainable
- env/tool/user feedback tokens must be masked out
- incremental message appends must stay consistent with full chat-template tokenization

`verl/workers/rollout/schemas.py` already solves most of this with:

- `AsyncRolloutRequest`
- `add_user_message(...)`
- `add_assistant_message(...)`
- `add_tool_response_messages(...)`
- `finalize(...)`
- tokenization sanity checking

So the shortest path is to wrap that instead of inventing new token bookkeeping.

## Relevant references in this repo

- `verl/experimental/agent_loop/mcbots_bridge.py`
- `verl/workers/rollout/schemas.py`
- `verl/experimental/agent_loop/agent_loop.py`
- `verl/experimental/agent_loop/tool_agent_loop.py`
- `tests/workers/rollout/test_sglang_async_rollout_multimodal_delta.py`
- `jl_patch/tests/experimental/agent_loop/test_mcbots_bridge_on_cpu.py`

## Related external reference

`/Users/bytedance/Projects/rllm` is architecturally relevant.

The useful ideas from `rllm` are:

- sticky session/application id routing
- assistant tokens as trainable mask `1`
- env/tool/user tokens as mask `0`
- keeping agent/env execution above the verl training substrate

Do **not** directly copy the `rllm` `BaseAgent` abstraction. It assumes rewriting the agent into their framework, which is explicitly not the current goal.

## What has already been implemented

Minimal bridge scaffold:

- file: `verl/experimental/agent_loop/mcbots_bridge.py`
- exported in: `verl/experimental/agent_loop/__init__.py`

What it does:

- wraps `AsyncRolloutRequest`
- initializes request state from OpenAI-style messages
- exposes:
  - `get_generation_prompt_ids()`
  - `add_user_message(...)`
  - `add_assistant_message(...)`
  - `add_tool_response_messages(...)`
  - `finalize(...)`

Minimal CPU-only test:

- file: `jl_patch/tests/experimental/agent_loop/test_mcbots_bridge_on_cpu.py`

What it tests:

- create a toy tokenizer/chat template
- initialize bridge from messages
- append an assistant message
- append a user observation
- finalize request
- verify:
  - incremental token state matches full chat-template tokenization
  - assistant text is in trainable response span
  - observation text is in masked response span

## Commit / branch status

Already committed and pushed:

- branch: `main`
- commit: `faf918c7`
- message: `Add minimal mcbots bridge scaffold`

## Validation status

### Passed

- static syntax validation for:
  - `verl/experimental/agent_loop/mcbots_bridge.py`
  - `jl_patch/tests/experimental/agent_loop/test_mcbots_bridge_on_cpu.py`

### Blocked on current machine

Dynamic runtime test was **not** completed on this machine because:

- `uv run` is available
- local `.venv` does not contain `torch`
- the environment could not resolve/install missing packages from PyPI

This is an environment/dependency blocker, not a GPU blocker.

This first bridge test does **not** require GPU.

## Important environment notes

If using `uv run` in a restricted environment:

- prefer `uv run --no-sync ...` if a valid local `.venv` already exists
- if `uv` cache permissions are restricted, set:
  - `UV_CACHE_DIR=/tmp/uv-cache`

If `uv run` tries to build/sync and network is unavailable, dynamic testing will fail before reaching the bridge logic.

## Proposed extra_body protocol (not implemented yet)

These fields should eventually be sent from `mcbots` to `verl`:

- `episode_id`
  - required
  - stable for the whole rollout/episode
- `context_reset`
  - optional
  - true when local history has been trimmed/reset but it is still the same rollout
- `is_final`
  - optional
  - true on the last model request / explicit episode flush
- `agent_id`
  - optional
  - useful for multi-instance debugging
- `reward`
  - optional
  - if `mcbots` or env side wants to attach final reward directly

Note:

- `turn_id` does not need to come from `mcbots` in v1
- `verl` can assign per-episode turn indices internally

## Current recommended architecture direction

Near-term preferred path:

1. Keep `mcbots` runtime logic largely intact.
2. Use the minimal bridge to prove token alignment.
3. Add a thin layer that maps `extra_body` metadata into bridge/session state.
4. After that, decide whether model calls are exposed through:
   - a thin OpenAI-compatible gateway
   - or a Ray-managed internal entrypoint

Current bias:

- use Ray to manage `mcbots` lifecycle if needed
- but keep the first model-facing integration thin
- do not overengineer gateway/session infrastructure before the bridge is proven

## Immediate next steps for the next agent

1. Run the new bridge test on a machine with a working Python env that has:
   - `torch`
   - `transformers`
   - `pytest`
2. If the test passes, keep the bridge small and add the next thin layer:
   - a request/session wrapper that accepts `episode_id`
   - maps to one `McbotsBridge` instance
3. Do **not** build the full gateway yet if the bridge test is still unverified.
4. After the bridge is verified, add the smallest possible metadata ingestion path for:
   - `episode_id`
   - `context_reset`
   - `is_final`

## Suggested commands on the next machine

If the target machine already has a usable `uv` env:

```bash
cd /Users/bytedance/Projects/verl
uv run --no-sync python -m unittest jl_patch.tests.experimental.agent_loop.test_mcbots_bridge_on_cpu -v
```

If the env still needs syncing and network is available:

```bash
cd /Users/bytedance/Projects/verl
uv run python -m unittest jl_patch.tests.experimental.agent_loop.test_mcbots_bridge_on_cpu -v
```

## Things to avoid

- Do not re-tokenize plain text responses later and treat that as ground truth.
- Do not move chat-template ownership into `mcbots`.
- Do not rewrite `mcbots` into a new agent abstraction before the bridge is proven.
- Do not build a large gateway/Ray/session stack before one minimal end-to-end bridge test passes.
