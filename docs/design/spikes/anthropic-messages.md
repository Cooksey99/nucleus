# Spike: Anthropic Messages API, and the HTTP mock tool

Date: 2026-10-11. No API keys were used and no paid calls were made.

Versions and doc pages checked:
- Official docs (platform.claude.com, fetched 2026-10-11): [streaming](https://platform.claude.com/docs/en/build-with-claude/streaming), [thinking](https://platform.claude.com/docs/en/build-with-claude/thinking), [effort](https://platform.claude.com/docs/en/build-with-claude/effort), [prompt caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching), [handling stop reasons](https://platform.claude.com/docs/en/build-with-claude/handling-stop-reasons), [refusals and fallback](https://platform.claude.com/docs/en/build-with-claude/refusals-and-fallback), [structured outputs](https://platform.claude.com/docs/en/build-with-claude/structured-outputs), [define tools](https://platform.claude.com/docs/en/agents-and-tools/tool-use/define-tools), [handle tool calls](https://platform.claude.com/docs/en/agents-and-tools/tool-use/handle-tool-calls), [parallel tool use](https://platform.claude.com/docs/en/agents-and-tools/tool-use/parallel-tool-use), [fine-grained tool streaming](https://platform.claude.com/docs/en/agents-and-tools/tool-use/fine-grained-tool-streaming), [errors](https://platform.claude.com/docs/en/api/errors), [rate limits](https://platform.claude.com/docs/en/api/rate-limits), [vision](https://platform.claude.com/docs/en/build-with-claude/vision), [PDF support](https://platform.claude.com/docs/en/build-with-claude/pdf-support).
- SDK source: `anthropics/anthropic-sdk-python` v1.13.0 at commit `b4b7916` (2026-10-09), `anthropics/anthropic-sdk-typescript` at `2979ed5` (2026-10-09), `anthropics/anthropic-sdk-go` at `bdcae90`. All three are MIT licensed.
- The `claude-api` reference skill bundled with Claude Code 2.1.296 (model tables cached 2026-10-06). It is cited as "skill" below and was used only where it agrees with the docs or for detail the docs pages didn't cover.

How each claim was established: **[verified]** means run in spike code or read directly in a file or doc quoted here; **[docs]** means read in official docs or SDK source; **[inferred]** means my conclusion from those.

## Questions

From the roadmap (Phase 0.5 and Phase 3 "Messages protocol for Anthropic"), as assigned:
1. Streaming: the full SSE event list with exact JSON, including redacted thinking.
2. Tool use: `tool_use` / `tool_result`, `is_error`, parallel calls, `tool_choice`, message-ordering rules.
3. Thinking: adaptive thinking and effort on current models; replay rules during tool loops and across model switches; check the roadmap rule "tagged with provider and model, replayed when they match and dropped otherwise".
4. Prompt caching: automatic vs `cache_control`, limits, minimum tokens, TTLs, usage fields; a default for an agent loop.
5. `refusal` and the other stop reasons, and how the library should surface each.
6. Structured output vs tool-based extraction for `agent.extract::<T>()`.
7. Image and PDF input shapes; error and 529 shape; retry and rate-limit headers; the request-id header.
8. Fixture sources with realistic recorded SSE streams.
9. (Part B) Which HTTP mock tool, proven with a spike.

Found along the way:
10. Preserved thinking: thinking blocks are now bound to the conversation prefix (system, tools and earlier messages). This affects `with_system`, `with_tools`, handoff and RAG injection (see Q3).
11. Current models reject non-default `temperature`/`top_p`/`top_k`, assistant prefill, and (on Opus 5.5 / Sonnet 5.5 / Fable 5.1) forced `tool_choice`.

## Findings

### Current models (context for everything below)

| Model | Thinking when `thinking` omitted | Can thinking be turned off? | Effort default | Forced `tool_choice` | Cache minimum |
|---|---|---|---|---|---|
| `claude-opus-5-5` | adaptive | No: `disabled`, `enabled`, `between_tools` all 400 | `medium` | 400 | 512 |
| `claude-sonnet-5-5` | adaptive | `{"type":"between_tools"}` at effort ≤ `high`; `disabled` 400 | `high` | 400 | 512 |
| `claude-haiku-5-5` | adaptive | `disabled` at effort ≤ `high` | `medium` | accepted | 512 |
| `claude-fable-5-1` | adaptive | No | `high` | 400 | 512 |

Sources: thinking table in [thinking § Configuring thinking](https://platform.claude.com/docs/en/build-with-claude/thinking) [docs]; [effort § How effort works](https://platform.claude.com/docs/en/build-with-claude/effort) [docs]; [errors § Forced tool use not supported](https://platform.claude.com/docs/en/api/errors) [docs]; [prompt caching § Cache limitations](https://platform.claude.com/docs/en/build-with-claude/prompt-caching) [docs]. All four support effort levels `low` through `max` [docs, effort table], 128K max output [docs, thinking § Output limits] and structured outputs [docs, structured-outputs `supportedModels`].

On all four (and on Opus 5, Sonnet 5, Opus 4.7/4.8), non-default `temperature`, `top_p` or `top_k` returns 400 on every request. Sending `top_p: 1`, or both `temperature` and `top_p` even at defaults, is also a 400 [docs, thinking § Sampling parameters]. Claude 4.6 and later reject a prefilled final assistant turn: "This model does not support assistant message prefill. The conversation must end with a user message." [docs, errors § Prefill not supported].

### Q1. Streaming

**Event flow** [docs, streaming § Event types]: `message_start` (a `Message` with empty `content`), then for each content block `content_block_start`, one or more `content_block_delta`, `content_block_stop` (each with an `index` into the final `content` array), then one or more `message_delta`, then `message_stop`. Any number of `ping` events can appear. `error` events can arrive after the HTTP 200. "New event types may be added, and your code should handle unknown event types gracefully." Each event's SSE `event:` name equals the `type` in its JSON.

Exact examples (copied from the docs pages; the tool-use stream is a real recording with `model: claude-opus-5`):

```text
event: message_start
data: {"type":"message_start","message":{"id":"msg_014p7gG3wDgGV9EUtLvnow3U","type":"message","role":"assistant","model":"claude-opus-5","stop_sequence":null,"usage":{"input_tokens":472,"output_tokens":2},"content":[],"stop_reason":null}}

event: content_block_start
data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}

event: ping
data: {"type": "ping"}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Okay"}}

event: content_block_stop
data: {"type":"content_block_stop","index":0}

event: content_block_start
data: {"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"toolu_01T1x1fJ34qAmk2tNTrN7Up6","name":"get_weather","input":{}}}

event: content_block_delta
data: {"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":""}}

event: content_block_delta
data: {"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\"location\":"}}

event: content_block_delta
data: {"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":" \"San"}}

event: content_block_stop
data: {"type":"content_block_stop","index":1}

event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"tool_use","stop_sequence":null},"usage":{"output_tokens":89}}

event: message_stop
data: {"type":"message_stop"}
```

Thinking ([streaming § Streaming request with thinking](https://platform.claude.com/docs/en/build-with-claude/streaming)) [docs]:

```text
event: content_block_start
data: {"type": "content_block_start", "index": 0, "content_block": {"type": "thinking", "thinking": "", "signature": ""}}

event: content_block_delta
data: {"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": "I need to find the GCD of 1071 and 462 using the Euclidean algorithm.\n\n1071 = 2 × 462 + 147"}}

event: content_block_delta
data: {"type": "content_block_delta", "index": 0, "delta": {"type": "signature_delta", "signature": "EqQBCgIYAhIM1gbcDa9GJwZA2b3hGgxBdjrkzLoky3dl1pkiMOYds..."}}

event: content_block_stop
data: {"type": "content_block_stop", "index": 0}
```

Error event [docs, streaming § Error events]:

```text
event: error
data: {"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}}
```

Refusal stop with `stop_details` in `message_delta` (Python SDK fixture `tests/lib/streaming/fixtures/refusal_response.txt`) [verified, read file]:

```text
event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"refusal","stop_sequence":null,"stop_details":{"type":"refusal","category":"cyber","explanation":"This request was refused due to policy."}},"usage":{"output_tokens":0}}
```

Details that matter for the parser:
- **Delta types** are exactly `text_delta`, `input_json_delta`, `citations_delta`, `thinking_delta`, `signature_delta` (Python SDK `src/anthropic/types/raw_content_block_delta.py`) [docs, SDK source]. Treat an unknown delta type as a provider event, not an error.
- **`signature_delta`** is sent once, just before `content_block_stop` of a thinking block [docs, streaming § Thinking delta].
- **`display: "omitted"`** (the default on all current models) streams one empty `thinking_delta`, then the `signature_delta`. Thinking text is never streamed [docs, thinking § Controlling thinking display]. So a stream can look silent for a long time while the model thinks [inferred].
- **Progress updates**: on Opus 5.5, Sonnet 5.5 and Fable 5.1, notes the model writes between tool calls arrive as *separate thinking blocks* with their own signature, immediately before the `tool_use` they introduce. They are empty under `omitted`. Under `display: "updates"` (beta `thinking-display-updates-2026-08-18`) or Sonnet 5.5's `between_tools`, they have text [docs, thinking § Progress updates between tool calls]. Text that used to come back as `text` blocks between tool calls now comes back in these blocks [skill, model-migration § Opus 5.5].
- **`redacted_thinking`**: shape `{"type": "redacted_thinking", "data": "..."}` [docs, thinking § Redacted thinking blocks]. No delta type exists for it, so it must arrive complete in `content_block_start` and close with `content_block_stop` [inferred from the SDK delta union and from the SDK accumulator, which appends the `content_block_start` block as-is, `lib/streaming/_messages.py`]. No recorded stream with one exists in any SDK fixture, so this stays open. It must be round-tripped unchanged. Filtering on `type == "thinking"` alone breaks the protocol [docs].
- **`message_delta.usage` is cumulative** [docs, streaming warning]. The Python SDK *overwrites* snapshot fields with each `message_delta` value that is present, and keeps `message_start` values for fields the delta omits (`input_tokens`, cache fields, `server_tool_use`, `output_tokens_details`) [docs, SDK `_messages.py` lines 514-534; fixtures `message_delta_fields_response.txt`, `message_delta_omitted_usage_response.txt`].
- **`message_start` usage** carries `input_tokens`, `cache_creation_input_tokens`, `cache_read_input_tokens`, `cache_creation.{ephemeral_5m_input_tokens, ephemeral_1h_input_tokens}`, `output_tokens`, `service_tier`, and sometimes `inference_geo` [verified, SDK fixtures]. Final usage can add `output_tokens_details.thinking_tokens` and `iterations` (one entry per attempt when server-side fallback ran) [verified, `stream-a-refusal.sse`].
- **`input_transformations`** (beta `thinking-binding-controls-2026-08-01`) appears on the `message_start` message and again on `message_delta` after a mid-stream fallback [docs, streaming § Event types; verified, fixture `input_transformations_delta_response.txt`].
- **`fallback` content blocks** (server-side fallback beta) arrive as a `content_block_start`/`content_block_stop` pair with no deltas [docs, streaming § Event types].
- **Real streams pad JSON with trailing spaces** inside `data:` lines, for example `..."output_tokens":2}}      }`. This appears in the Go SDK's real VCR recordings and the Python fable-fallback fixture [verified, read files]. The parser must tolerate whitespace after the JSON value. serde_json does, but a hand-rolled fast path might not [inferred].
- **SSE framing**: the Python SDK splits frames on `\r\r`, `\n\n` or `\r\n\r\n`, decodes lines as UTF-8 only after splitting, joins multi-line `data:` with `\n`, and ignores frames for unknown event names (it only yields listed names, skips `ping`, raises on `error`) (`src/anthropic/_streaming.py`, `SSEDecoder`) [docs, SDK source]. The TS SDK's decoder tests cover "multi-byte characters across chunks", "multiple events in one chunk", "special new line characters" and "data json escaped double new line" (`tests/streaming.test.ts`) [docs]. Port these as proptest and unit cases.
- **Fine-grained tool streaming**: `eager_input_streaming: true` per tool (no beta header) streams tool input without server-side buffering or validation. The accumulated `partial_json` may then be invalid, especially on `max_tokens`. The docs recommend replying with an `is_error` `tool_result` whose content is `{"INVALID_JSON": "<raw>"}` [docs, fine-grained-tool-streaming]. Without the flag, the API buffers each key/value and then emits it in chunks [docs, streaming § Input JSON delta].
- **Resuming an interrupted stream** on Claude 4.6+: send a new *user* message containing the partial text and "continue" (prefill is gone). Tool-use and thinking blocks can't be partially recovered [docs, streaming § Error recovery].

### Q2. Tool use

- **Definition**: `name` (`^[a-zA-Z0-9_-]{1,128}$`), `description`, `input_schema`, optional `input_examples`, `cache_control`, `strict`, `defer_loading`, `allowed_callers`, `eager_input_streaming` [docs, define-tools]. With `strict: true` the schema needs `additionalProperties: false` and `required`, and tool inputs are then guaranteed to validate [docs, structured-outputs; skill].
- **`tool_use` block**: `{"type":"tool_use","id":"toolu_…","name":…,"input":{…}}` [docs, handle-tool-calls]. When streaming, `content_block_start` carries `input: {}` as a placeholder and the real value is the concatenated `partial_json` [docs, fine-grained-tool-streaming § Accumulating].
- **`tool_result` block**: `{"type":"tool_result","tool_use_id":…,"content": <string | [text|image|document|search_result blocks]>, "is_error": true?}`. `content` is optional [docs, handle-tool-calls].
- **Ordering rules** [docs, handle-tool-calls § Important formatting requirements; parallel-tool-use]:
  - Results must be in the user message *immediately* after the assistant `tool_use` message.
  - In that message, all `tool_result` blocks come *first*. Text may follow them, but text before them is a 400. A pending tool_use with no result gets "tool_use ids were found without tool_result blocks immediately after".
  - Return one result per `tool_use`, all in a single user message. Splitting them "silently trains Claude to stop making parallel calls" [skill].
  - If you skip a call, still return a result with `is_error: true`.
  - If the assistant turn also has an unresolved `server_tool_use`, the user message may contain *only* `tool_result` blocks.
  - Execution order is up to the client. Parallel execution is fine.
- **Empty `end_turn` responses** after tool results are usually caused by adding text blocks right after the results. Fix: send the `tool_result` blocks alone [docs, handling-stop-reasons § Empty responses]. That page says "never add text blocks immediately after tool results", while handle-tool-calls allows text after the results. The safe rule for nucleus is to send results only [inferred].
- **`tool_choice`**: `auto` (default with tools), `any`, `tool` (with `name`), `none` (default without tools). `disable_parallel_tool_use: true` goes *inside* `tool_choice`: with `auto` it means at most one call, with `any`/`tool` exactly one [docs, define-tools, parallel-tool-use]. **`any` and `tool` return 400 on Opus 5.5, Sonnet 5.5, Fable 5.1 and Mythos 5.1** ("tool_choice: type "tool" and "any" are not supported for this model."). Where forced use is accepted, it skips thinking [docs, errors, define-tools]. Changing `tool_choice` invalidates the messages cache only [docs, prompt-caching].
- **Errors**: a tool failure or a bad call (missing parameter, unknown tool) goes back as `is_error: true` with an instructive message. Claude retries 2-3 times [docs, handle-tool-calls].

### Q3. Thinking, effort, and replay

**Configuration.** Use `thinking: {"type": "adaptive"}` or omit it. Current models run adaptive thinking when it's omitted. `budget_tokens` (`"enabled"`) is a 400 on every Claude 4.7+ model [docs, thinking table; errors § Extended thinking not supported]. Turning thinking off differs per model (table above). Opus 4.7/4.8 run *without* thinking when `thinking` is omitted [docs]. `display`: `summarized` | `omitted` (default on current models) | `updates` (beta). Thinking is billed in full under every value [docs]. Interleaved thinking (thinking between tool calls) is automatic with adaptive thinking [docs, thinking § Interleaved thinking].

**Effort.** `output_config.effort` takes `low|medium|high|xhigh|max`. It's GA with no header [docs, effort]. It affects all output tokens, including tool calls. Changing the top-level value starts the cache over (messages always, tools and system depending on the model). Setting it explicitly to the model's default is the same as omitting it. Per-message effort, a `{"role":"system","content":[],"output_config":{"effort":…}}` message under beta `mid-conversation-output-config-2026-07-01`, changes effort without busting the cache on Opus 5.5, Sonnet 5.5, Haiku 5.5 and Fable 5.1 [docs, effort § Change effort mid-conversation].

**Replay inside a tool loop.** This is required. Pass every `thinking` and `redacted_thinking` block back "complete and unmodified", alongside its `tool_use`. In the latest assistant message the run of thinking blocks can't be reordered, edited or partially dropped. A modified block is a 400: "`thinking` or `redacted_thinking` blocks in the latest assistant message cannot be modified." Pass blocks back even when their `thinking` text is empty [docs, thinking § Preserving thinking blocks; errors § Thinking blocks cannot be modified]. Across turns the docs recommend passing everything back; outside tool use, earlier turns' thinking may be omitted. The API filters blocks itself and bills only the ones it keeps. Current models keep all prior turns' thinking in context [docs, thinking § Thinking block preservation by model].

**Replay across model switches.** Keep passing blocks back unchanged. "A thinking block is readable only by the model that produced it and certain other models, and the API ignores or drops the blocks the target model can't read." A dropped block is unbilled and does not cause an error [docs, thinking § Switching models; preserved-thinking]. The readers matrix [docs, same section]:
- Opus 5.5 reads Opus 5, earlier Opus/Sonnet/Haiku, and Haiku 5.5 (on the Claude API and Google Cloud). It does not read Fable or Mythos.
- Only Fable 5.1 and Mythos 5.1 read Opus 5.5 blocks (Claude API only).
- Only Opus 5.5 reads Sonnet 5.5 blocks (Claude API and Google Cloud).
- Only Opus 5.5 and Sonnet 5.5 read Haiku 5.5 blocks.
- No model other than Mythos 5.1 reads Fable 5.1 blocks.

The docs say to strip blocks yourself "only to save input tokens on models that ignore rather than drop them". The skill adds that removing blocks yourself "can trigger ordering/signature 400s" [skill, model-migration § Fable 5.1 breaking change 2].

**Preserved thinking (the conversation check)** [docs, thinking § Preserved thinking; errors § Thinking block no longer matches the conversation]. On Fable 5.1, Opus 5.5, Sonnet 5.5 and Haiku 5.5, each block's signature binds the top-level `system`, the `tools`, and every earlier message. If any of these changes, that block *and every later thinking block* become invalid. For accounts created on or after 2026-08-31 the request is then a 400:

```text
messages.{i}.content.{j}: Invalid `signature` in `thinking` block. The block is bound to a different conversation. Remove the block, or set `thinking.block_binding.prefix_mismatch_behavior` to "drop_block".
```

Older accounts are enforced only when they set `thinking.block_binding.prefix_mismatch_behavior` (`"error"` or `"drop_block"`, beta `thinking-binding-controls-2026-08-01`). `drop_block` drops the first mismatched block and all later ones, and reports `input_transformations` entries with reason `prefix_binding_mismatch` or `model_binding_mismatch`.

What counts as an edit [skill, model-migration § Fable 5.1 breaking change 3, consistent with the docs summary]:
- changing the system prompt or the tool set mid-conversation
- injecting per-request text into an earlier turn and removing it next time
- deleting or snipping old turns or tool results
- removing a thinking block other than a leading run
- a URL image or document whose bytes change

These do *not* count: appending, `role: "system"` messages appended mid-conversation, changing `max_tokens`/`effort`/`tool_choice`, moving `cache_control`, and server-side compaction or context editing. Sonnet 5.5 and Haiku 5.5 blocks are also bound to the producing account on Bedrock and Google Cloud [docs].

**Roadmap rule check: "tagged with provider and model, replayed when they match and dropped otherwise". Correct it.**
- The *provider* half is right. Anthropic thinking blocks mean nothing to OpenAI, xAI, Gloo or local, so drop them when the request goes to another provider [inferred].
- The *model* half is wrong for Anthropic. Within Anthropic, replay blocks to *any* Anthropic model, unchanged. The API decides which it can read and drops the rest unbilled. Dropping them client-side loses reasoning on upgrades the API would have kept (for example Opus 5.5 → Fable 5.1, Haiku 5.5 → Opus 5.5) and risks signature/ordering 400s [docs; skill].
- A new condition is missing: **the prefix must be unchanged.** nucleus changes the prefix in two first-class features: `with_system` / `with_tools` / handoff to an agent with a different system prompt or tools, and RAG text "added to every request" if it is injected into a past user turn and not persisted. After either, replayed blocks are invalid. On new accounts that is a hard 400 on Opus 5.5 / Sonnet 5.5 / Haiku 5.5 / Fable 5.1 [docs; inferred for nucleus features].

Corrected rule: *Provider-specific items are tagged with provider, model, and a fingerprint of the request prefix that produced them (hash of the rendered system + tools). They are replayed unchanged to the same provider, whatever the model. Within the Anthropic protocol, when the current system/tools fingerprint differs from a replayed block's, drop that block and every later thinking block (the same thing `drop_block` does), except inside the in-flight tool loop. Items go only to the provider that made them.* Also keep history append-only (see Impact).

### Q4. Prompt caching

- **Two mechanisms** [docs, prompt-caching § Automatic caching]:
  - *Automatic*: a top-level `"cache_control": {"type": "ephemeral"}` (optionally `"ttl": "1h"`). The breakpoint goes on the last cacheable block and moves forward as the conversation grows. It uses one of the 4 slots. It is a 400 if 4 explicit breakpoints already exist, or if the last block has an explicit marker with a different TTL. It is available on every platform except the legacy Bedrock InvokeModel integration.
  - *Explicit*: `cache_control` on tool definitions, system blocks, or message blocks (text, image, document, tool_use, tool_result) [skill].
  - Without any marker, nothing is cached, apart from thinking blocks cached with tool results during a loop [docs, thinking § Thinking and prompt caching].
- **Limits**: max 4 breakpoints. Minimum cacheable prefix is 512 tokens on all four current models (1,024 on Opus 4.8 and Sonnet 5; 2,048 on Opus 4.7; 4,096 on Opus 4.6/4.5 and Haiku 4.5). Shorter prompts silently aren't cached [docs]. The lookback is 20 positions per breakpoint; on the Claude API a run of consecutive `tool_use` blocks (or `tool_result` blocks) counts as one position [docs]. A cache entry becomes readable only once the first response *starts streaming* [skill].
- **TTL**: 5 minutes by default, `"ttl": "1h"` optional. Longer TTLs must come before shorter ones in the prompt. A read refreshes the TTL for free [docs; skill].
- **Prices**: write 1.25× base input (5m) or 2× (1h). Read 0.1× in general, but **0.05× on Opus 5.5 and Sonnet 5.5** and 0.025× on Fable 5.1 [docs, prompt-caching § Pricing]. Cache reads don't count toward ITPM rate limits on current models [docs, rate-limits § Cache-aware ITPM].
- **Usage fields**: `input_tokens` (uncached tail after the last breakpoint), `cache_creation_input_tokens` (equals the sum of `cache_creation.ephemeral_5m_input_tokens` + `ephemeral_1h_input_tokens`), `cache_read_input_tokens`. Total input = the sum of all three [docs].
- **What invalidates**:
  - Tool definitions or a model switch invalidate everything.
  - The system prompt invalidates system and messages.
  - `tool_choice`, images, or thinking/effort changes invalidate messages (thinking/effort sometimes more).
  - A dropped thinking block invalidates messages from its position onward [docs, prompt-caching § What invalidates the cache].

**Recommended default for an agent loop** [docs plus skill "robust combination"; inferred for nucleus]:
1. One explicit `cache_control: {type: ephemeral}` on the **last system block**, or on the last tool when there is no system prompt. Tools render before system, so this caches both.
2. Plus **top-level automatic caching** for the growing history.

That uses 2 of the 4 slots and leaves 2 for users via `provider_options`. Use a 5-minute TTL by default, with a `cache_ttl(1h)` option for human-paced chats where replies come more than 5 minutes apart. Serialize tools deterministically: stable order and stable JSON key order. Never interpolate volatile data into the system prompt. Keep `effort` and thinking settings constant per agent.

`Usage` should expose `cache_read_tokens` and `cache_write_tokens` (5m/1h split) separately, so cost can be computed with per-model multipliers.

### Q5. Stop reasons, and how to surface them

The full set is `end_turn`, `max_tokens`, `stop_sequence`, `tool_use`, `pause_turn`, `refusal`, `model_context_window_exceeded` [docs, handling-stop-reasons; SDK `types/stop_reason.py`]. `stop_details` is non-null only on `refusal`.

| `stop_reason` | Meaning (docs) | nucleus behaviour (recommended) |
|---|---|---|
| `end_turn` | finished naturally | `StopReason::EndTurn`; the run ends |
| `tool_use` | client tool call(s) | run tools (approval hook, parallel), append results, loop |
| `max_tokens` | hit `max_tokens` | `StopReason::MaxTokens`. Never execute an incomplete trailing `tool_use` (its JSON may be partial). Keep the assistant turn as-is (to continue, append a user message; prefill is gone). Return the partial output marked truncated |
| `stop_sequence` | custom stop sequence hit | `StopReason::StopSequence(String)` from `stop_sequence` |
| `pause_turn` | server-tool loop hit its iteration limit | send the assistant content back unchanged to continue; counts as a loop step. Only happens with server tools |
| `refusal` | HTTP 200, declined (classifier or model) | `StopReason::Refusal { category: Option<String>, explanation: Option<String> }`. Discard partial output and don't commit the turn to the `Conversation` (roll back, as `run_turn` does today). `complete()`/`extract()` return a typed `Error::Refused` so `?` works. The fallback chain treats refusal as a trigger: retry on the next model in the chain with the history unchanged, and stay on that model for the rest of the conversation |
| `model_context_window_exceeded` | output filled the context window | `StopReason::ContextWindowExceeded`; truncated, like `max_tokens`, with a hint to compact |
| unknown | | `StopReason::Other(String)` (the enum is `#[non_exhaustive]`) |

Refusal details [docs, refusals-and-fallback]:
- Categories so far: `cyber`, `bio`, `frontier_llm`, `reasoning_extraction`, `general_harms`. `category` and `explanation` may be `null` as a normal value. Branch on `stop_reason`, never on `content`.
- A refusal can arrive before any output (empty `content`) or mid-stream (discard the partial).
- Pre-output refusals in `bio`, `frontier_llm` and `reasoning_extraction` are billed; mid-stream refusals bill what was streamed.
- `reasoning_extraction` is triggered by prompts or schema fields that ask for "reasoning"/"thinking" in the output. Don't put a `reasoning` field in `extract` schemas [docs].
- Refusals are HTTP 200, so emit a `tracing` event for each one; error-rate monitoring won't see them [docs, "Instrument refusals as their own signal"].
- Server-side `fallbacks` (beta, Claude API only, not Batches, not Haiku 5.5) and fallback credit (`fallback_credit_token`, beta) exist. Leave them to `provider_options` for 0.2 and implement fallback generically in nucleus [inferred].

### Q6. Structured output for `extract::<T>()`

- **Native JSON output is GA** on all current models: `"output_config": {"format": {"type": "json_schema", "schema": {...}}}`. The result is valid JSON in the response's `text` block. No beta header; `output_format` is deprecated [docs, structured-outputs].
- **Schema restrictions** [docs]:
  - Every object must have `additionalProperties: false`.
  - No recursive schemas, no numeric or string length constraints, no `minItems` other than 0/1, no external `$ref`.
  - Limits: 24 optional parameters, 16 union-typed parameters, 20 strict tools per request.
  - Unsupported features are a 400. The first use compiles a grammar, which adds latency; grammars are cached 24h.
  - Changing `format` busts the messages cache.
  - Enum casing can vary, so compare case-insensitively.
  - Refusal and `max_tokens` can produce non-conforming output.
- **schemars output won't pass as-is** (for example `format: "uint32"`, `minimum: 0` on unsigned ints, `oneOf` for enums, `$schema`, `title`) [inferred]. The Python SDK has `transform_schema` (`src/anthropic/lib/_parse/_transform.py`). It rewrites `oneOf` → `anyOf`, keeps only supported keywords and string formats, and moves dropped constraints into `description`, for example `"A number\n\n{minimum: 1, maximum: 10}"` [docs, SDK source]. nucleus needs a Rust port of this, and should validate the parsed value locally against the original schema.
- **Tool-based extraction** (forcing a tool call) no longer works on Opus 5.5, Sonnet 5.5 or Fable 5.1, because forced `tool_choice` is a 400 [docs]. With `auto` + `strict: true` the call is likely but not guaranteed.
- **Recommendation**: `extract::<T>()` uses `output_config.format` on Anthropic. Run it as a single non-tool request: no other tools, the agent's system prompt, and the extraction schema. Check `stop_reason` before parsing: refusal → `Error::Refused`, max_tokens → `Error::Truncated`. Parse errors produce a typed error that includes the raw text [inferred]. Don't add a "reasoning" field [docs]. Citations are incompatible with `output_config.format` [skill].

### Q7. Images, PDFs, errors, headers

**Images**: `{"type":"image","source":{"type":"base64","media_type":"image/jpeg|png|gif|webp","data":…}}`, `{"type":"url","url":…}` or `{"type":"file","file_id":…}`. Limits:
- 10 MB per image (base64) on the Claude API, 8000×8000 px.
- More than 20 images in a request triggers a stricter dimension limit (resize to ≤2000 px).
- Up to 600 images per request (100 on 200k-context models).
- Images placed before text work best.
- URL and file sources only on the Claude API (Bedrock and Vertex: base64 only).

[docs, vision]. With preserved thinking, a URL whose bytes change between turns invalidates later thinking. Prefer base64 or `file_id` [skill].

**PDFs**: `{"type":"document","source":{"type":"base64","media_type":"application/pdf","data":…}}`, plus `url`, `file`, and `text` sources (`{"type":"text","media_type":"text/plain","data":…}`). Optional `citations: {enabled: true}`. Limits: 32 MB per request, 600 pages (100 under 1M context) [docs, pdf-support, handle-tool-calls; skill]. Images and documents may also appear inside `tool_result.content` [docs].

**Error shape** [docs, errors]:

```json
{"type":"error","error":{"type":"not_found_error","message":"The requested resource could not be found."},"request_id":"req_011CSHoEeqs5C35K2UUqR7Fy"}
```

| HTTP | `error.type` | nucleus `ErrorKind` (proposed) | Retry? |
|---|---|---|---|
| 400 | `invalid_request_error` (also used for spend limits you set, and other 4xx) | `InvalidRequest` | no |
| 401 | `authentication_error` | `Auth` | no |
| 402 | `billing_error` | `Billing` (or `Auth`) | no |
| 403 | `permission_error` | `Auth` | no |
| 404 | `not_found_error` | `InvalidRequest` | no |
| 409 | `conflict_error` | `Provider` | SDK retries it |
| 413 | `request_too_large` (from Cloudflare before the API) | `InvalidRequest` | no |
| 429 | `rate_limit_error` | `RateLimited { retry_after }` | yes, **unless** there is no `retry-after` and `error.details.error_code == "enforced_spend_limit_reached"` (monthly spend cap; fails until next month) |
| 500 | `api_error` | `Provider` (retryable) | yes |
| 504 | `timeout_error` | `Timeout` | yes |
| 529 | `overloaded_error` | `Overloaded` (retryable) | yes |
| SSE `error` after 200 | same `error` object, no `request_id` in the docs example | as above by `type` | only if no content was emitted yet [inferred] |

Sources: [docs, errors; rate-limits § Reaching your spend cap].

**Retry behaviour of the official SDK** (Python `_base_client.py`) [docs, SDK source]:
- Retries 408, 409, 429, all ≥500, and connection errors; 2 retries by default.
- Obeys the `x-should-retry: true|false` header.
- Waits for `retry-after-ms` (non-standard) first, then `retry-after` in seconds (floats accepted), then `retry-after` as an HTTP date.
- Otherwise uses exponential backoff from 0.5 s up to 8 s with jitter.
- Timeout is 10 min, connect timeout 5 s.

nucleus should match this.

**Headers** [docs, errors § Request ID; rate-limits § Response headers; verified in the Go SDK VCR cassette `toolrunner/testdata/cassettes/tool_runner_streaming_all.yaml`]:
- Request headers: `x-api-key`, `anthropic-version: 2023-06-01`, `content-type: application/json`, and `anthropic-beta: a,b` (comma-separated) for betas.
- Every response has `request-id` (`req_…`). It should go on `nucleus::Error` and on the raw response.
- `retry-after` in seconds.
- `anthropic-ratelimit-{requests,tokens,input-tokens,output-tokens}-{limit,remaining,reset}`, where reset is RFC 3339.
- Streaming responses have `content-type: text/event-stream; charset=utf-8`.

### Q8. Fixture sources

All three SDK repos are MIT; keep the notice if we copy fixtures.

1. **Python SDK `tests/lib/streaming/fixtures/`** (`b4b7916`):
   - `basic_response.txt`, `tool_use_response.txt`
   - `refusal_response.txt`
   - `incomplete_partial_json_response.txt`, `tool_use_invalid_json_response.txt` (truncated or invalid `input_json_delta`)
   - `message_delta_fields_response.txt`, `message_delta_omitted_usage_response.txt` (usage merge rules)
   - `input_transformations_*_response.txt` (preserved-thinking drops)
   - `server_tool_use_response.txt`, `compaction_response.txt`, `context_management_response.txt`
   - `fallback_response.txt`, `fallback_credit_response.txt`

   Also `tests/fixtures/fable-fallback/stream-a-refusal.sse`, `stream-a-toolrefusal.sse` and `stream-b-fallback.sse`: thinking + signature + refusal with `stop_details.fallback_credit_token`, `usage.iterations`, and real-style whitespace padding. [verified, downloaded and read]
2. **TypeScript SDK**: the same `tests/lib/fixtures/*.txt` and `tests/fixtures/fable-fallback/*.sse`, plus SSE-decoder edge cases in `tests/streaming.test.ts` [docs].
3. **Go SDK `toolrunner/testdata/cassettes/*.yaml`** (go-vcr): *real recorded* multi-turn tool loops with full request bodies, response headers (`Request-Id`, all rate-limit headers) and raw SSE bodies. Files include `tool_runner_streaming_all.yaml`, `tool_runner_concurrent.yaml`, `tool_runner_tool_call_error.yaml` and `tool_runner_max_iterations.yaml`. The model is `claude-3-7-sonnet` (old, but the wire format is the same). Also `lib/betafallback/testdata/fallbackstream/*.sse` [verified, read `tool_runner_streaming_all.yaml`]. These make the best request-matching fixtures because the second request contains the replayed assistant turn and `tool_result`.
4. **Docs pages**: the exact streams quoted in Q1 (basic, tool use, thinking, web search, error). Error and 529 bodies come from the errors page. The refusal JSON comes from refusals-and-fallback.
5. **Gaps** (to hand-write from the docs shapes): a `redacted_thinking` stream, an interleaved thinking → tool_use → thinking → tool_use response, progress-update blocks, `pause_turn`, `model_context_window_exceeded`, mid-stream `error`, HTTP 429/529 with headers, and a 4.7+ model id in the fixtures.

### Q9 (Part B). HTTP mock tool

Spike code: `spikes/mock-http/` (standalone crate, empty `[workspace]`, edition 2024). It uses reqwest 0.12 (`stream`, `json`, `rustls-tls`, same major version as nucleus-core) and a hand-written byte-level SSE parser. Fixtures are the docs' "Streaming request with tool use" (27 events) and "Streaming request with thinking" (13 events, includes the 2-byte `×`). Each fixture starts with a `: source:` SSE comment line, which also tests that the parser skips comments.

**Comparison** (crates.io API, GitHub API and crate source, 2026-10-11) [docs, verified where marked]:

| | wiremock | httpmock | mockito | hand-rolled tokio server |
|---|---|---|---|---|
| Latest | 0.6.5 (2025-08-24) | 0.8.3 (2026-02-04) | 1.7.2 (2026-02-02) | n/a |
| Last commit | 2025-08-24, none in ~14 months [verified, GitHub API] | 2026-10-07 [verified] | 2026-08-01 [verified] | n/a |
| Open issues + PRs | 32 | 27 | 7 | n/a |
| License | MIT/Apache-2.0 | MIT | MIT | ours |
| Declared MSRV | none (edition 2024, so ≥ 1.85) | 1.88 | 1.70 (master: 1.85) | tokio's |
| Unique crates (normal edges, default features) | 86 | 111 | 58 | tokio only: 6; hyper server: 14; axum: 47 |
| Added on top of reqwest + tokio | +20 | +43 | +23 (fewer with `default-features = false`) | +0 |
| Async/tokio | native | async variants | `*_async` APIs; each server runs on its own thread with a single-threaded runtime | the test's own runtime |
| **Body in chunks with delays** | **No.** The body is one `Full<Bytes>`; `set_delay` delays the whole response, headers included. PR #133 "Support body streams" has been open since 2023-11 [verified: test shows headers after 303 ms, `Content-Length: 3419`, one chunk] | **No.** One buffer; `delay` covers the whole response [docs, source] | **Yes.** `with_chunked_body(\|w\| …)`: each `write_all` is one HTTP chunk, `thread::sleep` between writes, and returning `Err` cuts the body off mid-stream [verified] | **Yes**, async `tokio::time::sleep` per chunk [verified] |
| JSON body matching | exact, partial, schema | exact, partial | `Matcher::Json`, `Matcher::PartialJson`, `match_request` closure [verified] | manual |
| Header matching | rich | rich | `match_header`; `Matcher::Any` (present) and `Matcher::Missing` (absent) [verified] | manual |
| Request verification | `expect(n)`, `verify`, `received_requests()` | `hits`, `calls`, record/playback | `expect(n)`, `expect_at_least`/`_at_most`, `assert_async()`. **No `received_requests()`**; capture bodies via `match_request` | ~15 lines |

**Recommendation: `mockito` 1.7.2 as a dev-dependency, behind a thin nucleus test helper** (for example `serve_sse(chunks, delay)` and `expect_request(partial_json, headers)`). Reasons:
- It is the only crate of the three that can stream a body in chunks with per-chunk delays, which the parser and idle-timeout tests need.
- Its matching and verification cover the shared suite's needs.
- It has the smallest dependency tree and the quickest build of the three, a permissive license, and recent maintenance.

wiremock has the nicest matching API but cannot stream and looks dormant. httpmock cannot stream and is the heaviest.

Keep the ~150-line hand-rolled tokio server from the spike (`src/raw_server.rs`) as a second backend behind the same helper. It handles faults mockito can't express (malformed chunk framing, slow headers, connection resets), and its stalls are async.

**What the spike proved** (`cargo test`: 17 passed, 0 failed; I re-ran it myself on 2026-10-11, and the subagent's 10-run loop passed every time) [verified]:
- **One-shot replay.** `mockito_one_shot_tool_use_stream` checks the exact event sequence and rebuilds the text and the tool input `{"location":"San Francisco, CA"}` from the `partial_json` fragments. It also checks stop_reason `tool_use`, usage 472/89, and that each SSE `event:` name equals `data.type`.
- **Chunked delivery.** `mockito_chunked_7_byte_chunks_with_delay` sent 489 chunks of 7 bytes with 1 ms delays. The client received exactly 489 `bytes_stream()` items with identical sizes, and the parse equals the one-shot parse.
- **Awkward split points.** `mockito_chunked_split_inside_utf8_char_and_data_field` ends chunks after byte `0xC3` of `×` and inside `da|ta:`. `mockito_chunked_crlf_line_endings` uses CRLF line endings with a chunk ending between `\r` and `\n`. Both parse identically to the one-shot case.
- **Key fact: with HTTP/1.1 chunked encoding, reqwest/hyper delivers one HTTP chunk as one stream item and never merges them,** even with zero delay and a 200 ms pile-up. Tests can therefore choose the exact boundaries the parser sees, and delays are only needed for timing tests. Very large chunks may still be split by TCP [inferred].
- **Idle timeout.** In `mockito_idle_timeout_fires_when_server_stalls`, the server sends 2 events and stalls. A per-`next()` `tokio::time::timeout` of 200 ms fires with exactly `[message_start, content_block_start]` received. So the timeout is per gap between chunks, not for the whole request.
- **Request matching and verification.** `mockito_matches_json_body_and_headers_then_verifies_once` matches `x-api-key` present, `anthropic-version`, content-type and partial JSON `{"model":…,"stream":true}`, with `expect(1)`. A `stream:false` request and one missing the key get 501. `assert_async` confirms one hit, and the captured body is checked.
- **Stream cut off mid-body.** `mockito_abort_mid_stream_via_closure_error` surfaces as a reqwest decode error ("unexpected EOF during chunk size line"). The raw server's `raw_connection_dropped_mid_stream_is_an_error` does the same for a dropped connection.
- The same scenarios pass on the hand-rolled server (`tests/raw_server.rs`, 4 tests), including recording requests.

**Gotchas for the test helper** [verified unless marked]:
1. **A blocking stall freezes the whole mockito server.** If a chunk closure blocks (`thread::sleep(1.5 s)`) and the client hangs up, another request to the same server on a new connection waits about 1.5 s (`mockito_gotcha_blocking_stall_freezes_server_after_client_hangs_up`). The likely cause is that `ChunkedStream::drop` joins the writer thread on the server's single-threaded runtime [docs, inferred]. Make stalls releasable (an `AtomicBool` polled every few ms) and release them when the test ends.
2. Each streaming mock uses one OS thread and runs on wall-clock time. `tokio::time::pause` can't speed it up [inferred], so keep delays in the low milliseconds.
3. Body capture uses a `match_request` closure, and that closure only sees requests that passed the earlier matchers.

The parser in `spikes/mock-http/src/sse.rs` is a good starting point for nucleus's. It splits on `\n`, `\r\n` or `\r`, holds a trailing lone `\r` until the next byte, decodes UTF-8 only once a line is complete, joins multi-line `data:` with `\n`, skips comments, and reports `Truncated` when the stream ends mid-event.

## Impact on the design

Proposed changes to ROADMAP.md, Phase 2 and Phase 3:

1. **Message model (Phase 2).** Replace "provider-specific data … tagged with provider and model, replayed when they match and dropped otherwise" with:
   - Assistant messages are an **ordered list of parts** (text, thinking/reasoning, tool call, opaque provider block), because Anthropic interleaves thinking → tool_use → thinking → tool_use and the order must be replayed exactly.
   - Provider-specific parts (thinking `{thinking, signature}`, `redacted_thinking {data}`, `fallback`, unknown block types as raw JSON) are tagged with provider, model and a prefix fingerprint.
   - They are replayed **unchanged to the same provider regardless of model** and dropped only for other providers.
   - Tool-call `input` is stored as raw JSON (`Box<RawValue>` or `preserve_order`) so the bytes replay identically [inferred].
2. **History is append-only (Phase 2, new design rule).**
   - Nothing that reaches the provider may be edited later: no per-request injection into past turns, no deletion of old tool results, no system/tools rewrites behind the user's back. This keeps both the prompt cache and preserved thinking valid.
   - RAG "added to every request" must append its text as part of the user turn that is persisted, or use the `search_knowledge` tool. It must not be spliced into an earlier turn and then removed. This changes the Phase 4 bullet and the `ChatManager` known issue ("retrieved RAG text is saved into history" is then the *intended* behaviour for the injected variant, or it moves to a tool).
3. **`with_system` / `with_tools` / handoff (Phase 2).** Document that changing system or tools mid-conversation (a) restarts the prompt cache and (b) invalidates earlier thinking on Anthropic's current models. The Anthropic protocol strips thinking blocks whose prefix fingerprint no longer matches, from the first mismatch onward, so these features never 400 on accounts created after 2026-08-31. Add a mock test asserting the stripped request body. Later optimisation: render a mid-conversation `role: "system"` message instead of editing top-level `system`.
4. **Tool loop (Phase 2).**
   - Send all results of a step in one user message, results only, no text.
   - Denials and errors go back as `is_error: true`; skipped calls get an `is_error` "not executed" result.
   - When the max-step limit stops a run with a pending `tool_use`, append `is_error` results ("step limit reached") or don't commit that turn. A dangling `tool_use` makes the next request a 400.
   - A conversation must always end with a user message when sent (prefill is a 400).
5. **`RunOptions` (Phase 2/3).**
   - `effort` (`Low…Max`) is first-class, held constant per agent.
   - `thinking` is `Auto` (omit the field) | `Adaptive` | `Off`. `Off` maps per model (Sonnet 5.5 → `between_tools`, Haiku 5.5 / Opus 5 / Sonnet 5 → `disabled`, Opus 5.5 / Fable → `Error::InvalidRequest` before sending), or a Models API capability lookup is used.
   - `thinking_display` (`Omitted` default / `Summarized` / `Updates`).
   - `temperature`/`top_p` are `Option` and omitted by default. If the user sets them, pass them through and let the 400 explain; don't drop silently.
6. **Prompt caching (Phase 3).** Replace "automatic prompt caching" with the concrete default: an explicit breakpoint on the last system block (or last tool) plus top-level automatic `cache_control`, 5-minute TTL, a `cache_ttl` option, and deterministic tool serialization. `Usage` gets `cache_read_tokens` and `cache_write_tokens` (5m/1h), and cost uses per-model cache multipliers. This answers part of the Phase 2 `Usage` cost question.
7. **`refusal` (Phase 3).** `StopReason::Refusal { category, explanation }`. The partial turn is not committed. `complete`/`extract` return `Error::Refused`. Fallback chains (Reliability) trigger on refusal as well as on retryable errors and stay on the fallback model. Emit a `tracing` event per refusal.
8. **`extract::<T>()` (Phase 2/3).** Use native JSON-schema output (`output_config.format` on Anthropic) plus a Rust port of the SDK's `transform_schema` and local validation. Tool-forcing is not an option on Opus 5.5 / Sonnet 5.5 / Fable 5.1.
9. **Errors (Phase 2).** Add `ErrorKind`s or fields for `RateLimited { retry_after }`, `Overloaded`, `Timeout`, `Refused`, `Truncated`, and `SpendCapReached` (non-retryable 429). Carry `request_id` on every provider error. Retry logic honours `x-should-retry`, `retry-after-ms`, `retry-after` (seconds or date), and 408/409/429/5xx. A mid-stream SSE `error` is retried only if no content was delivered.
10. **Streaming parser (Phase 3).** Handle unknown event and delta types as pass-through provider events; tolerate whitespace padding after JSON; usage merge = overwrite present fields, keep `message_start` values for absent ones; treat `ping` as liveness for the idle timeout. Proptest the frame splitter with arbitrary chunk boundaries (Part B shows how).
11. **Shared test suite (Phase 3).** The Phase 0.5 checkbox says "e.g. `wiremock`"; change it to **`mockito`** (dev-dependency), wrapped in a nucleus test helper (`serve_sse(chunks, delay)`, request matching and capture). The spike's hand-rolled tokio server sits behind the same helper for transport faults. Vendor the MIT fixtures listed in Q8 under `tests/fixtures/anthropic/` with a NOTICE, and hand-write the gaps listed there. Every SSE fixture is replayed both in one piece and with adversarial chunk boundaries, including a proptest over random boundaries.

## Cross-check

Checked on 2026-10-11 against the Claude API reference bundled with Claude Code (model data cached 2026-10-06). These findings above match it: preserved thinking and the 2026-08-31 enforcement date, sampling parameters and prefill rejected, forced `tool_choice` rejected on Opus 5.5, Sonnet 5.5 and Fable 5.1, per-model effort defaults, thinking always on for Opus 5.5, `output_config.format`, and `stop_details` only on refusals.

The reference adds three things that matter for the design [docs]:

- **Mid-conversation system messages.** Opus 5.5, Sonnet 5.5, Haiku 5.5 and Fable 5.1 accept a `{"role": "system"}` entry inside `messages`. Unlike editing the top-level `system`, it keeps the cached prefix and the thinking blocks valid. That is a natural way to implement `with_system` and handoff to another agent on Anthropic without invalidating earlier thinking.
- **`prefix_mismatch_behavior: "drop_block"`.** Under the beta `thinking-binding-controls-2026-08-01`, the API drops thinking blocks whose prefix no longer matches instead of returning a 400. That could replace client-side fingerprinting, or back it up.
- **Server-side refusal fallbacks.** The `fallbacks` parameter (beta `server-side-fallback-2026-07-01`, `fallbacks: "default"`) reroutes a refused request on the server. Phase 3 fallback chains should decide whether to use it on Anthropic or keep fallbacks client-side and provider-neutral.

Also: `eager_input_streaming: true` on a tool streams large tool inputs as they're generated. Without it the API buffers the input. With it, the API no longer validates the input, so the client must check the parsed input against the schema.

## Open questions

- **Ping cadence and longest silent gap** with `display: "omitted"` at high effort: needed to pick a default idle timeout. Only a live call answers it; until then, use a generous default (minutes, not seconds) and make it configurable.
- **`redacted_thinking` on the wire**: confirm it arrives complete in `content_block_start` with no deltas (inferred), and whether current models still emit it at all, given that Fable 5.1 "returns regular thinking blocks, not redacted_thinking" [skill]. Live call only.
- **Does the prefix fingerprint need to match bytes or semantics?** For example, does re-serializing a `tool_use.input` with different key order invalidate later blocks? Byte-faithful storage avoids the question. Confirming needs a live call with `drop_block` and `input_transformations`.
- **Whether `tool_result` content order or `cache_control` movement interact with the prefix check.** The skill says `cache_control` moves are fine. Unverified live.
- **Models API `capabilities`** (`GET /v1/models/{id}`): does it expose `between_tools` and forced-tool-choice support, so `Provider::capabilities()` can avoid a hard-coded per-model table? Needs a key, but the call costs no tokens.
- **Server-side `fallbacks` and fallback credit**: expose them as typed options in 0.2 or leave them to `provider_options`? A design decision, not a live-call question.
- **Per-message effort and mid-conversation system messages** (betas) could make `with_system` and effort changes cache- and thinking-friendly. Decide after 0.2.
