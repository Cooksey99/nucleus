# Spike: OpenAI and xAI Responses API

Date: 2026-10-11 to 2026-10-12. No API keys were used and no paid calls were made.

Versions and doc pages checked:
- OpenAI docs, fetched as Markdown (append `.md`) on 2026-10-11: [function calling](https://developers.openai.com/api/docs/guides/function-calling), [reasoning](https://developers.openai.com/api/docs/guides/reasoning), [conversation state](https://developers.openai.com/api/docs/guides/conversation-state), [structured outputs](https://developers.openai.com/api/docs/guides/structured-outputs), [streaming](https://developers.openai.com/api/docs/guides/streaming-responses), [error codes](https://developers.openai.com/api/docs/guides/error-codes), [rate limits](https://developers.openai.com/api/docs/guides/rate-limits).
- `openai/openai-openapi` `openapi.yaml` at `e95c0fe` (2026-10-09, MIT).
- `openai/openai-python` 3.28.0 at `b7a1990` (2026-10-11, Apache-2.0); `openai/openai-node` 7.32.0 at `3add470` (2026-10-11, Apache-2.0).
- `openai/codex` at `95db2cd` (2026-10-11, Apache-2.0): `codex-rs/codex-api/src/sse/responses.rs`, a production Rust Responses client.
- xAI docs, fetched as Markdown on 2026-10-11 (index: https://docs.x.ai/llms.txt): [generate text](https://docs.x.ai/developers/model-capabilities/text/generate-text), [reasoning](https://docs.x.ai/developers/model-capabilities/text/reasoning), [structured outputs](https://docs.x.ai/developers/model-capabilities/text/structured-outputs), [function calling](https://docs.x.ai/developers/tools/function-calling), [streaming](https://docs.x.ai/developers/model-capabilities/text/streaming), [Responses reference](https://docs.x.ai/developers/rest-api-reference/inference/responses), [migration](https://docs.x.ai/developers/model-capabilities/text/comparison), [rate limits](https://docs.x.ai/developers/rate-limits), [debugging](https://docs.x.ai/developers/debugging), [release notes](https://docs.x.ai/developers/release-notes), and the xAI OpenAPI file https://docs.x.ai/openapi.json.
- `xai-org/xai-sdk-python` 1.20.0 at `1d9e1df` (2026-09-28, Apache-2.0). This SDK uses **gRPC**, not REST Responses (`src/xai_sdk/client.py`), so it has no SSE fixtures. It is useful only for feature semantics.
- `vercel/ai` at `ba05943` (2026-10-10, Apache-2.0): `@ai-sdk/openai` 4.0.91 and `@ai-sdk/xai` 5.0.20. Their `__fixtures__/*.chunks.txt` files are **recorded live streams**. They turned out to be the best fixture source.
- Third-party clients of the Grok account proxy: `CherryHQ/cherry-studio` `src/main/ai/provider/grokCli.ts` at `d9c7839` (2026-10-02), and `openclaw/openclaw` `extensions/xai/provider-routing.ts` at `c6e6380` and `docs/providers/xai.md` at `77e2261`.
- Spike code: `spikes/responses/` (throwaway, not merged). It is a byte-level SSE parser plus an event accumulator. It replays 38 recorded fixture files (39 responses) from OpenAI and xAI, splitting each stream into random byte chunks with both LF and CRLF line endings, and checks invariants. Run it with `cargo run -- <fixture.chunks.txt>...`.

How each claim was established: **[verified]** means run in the spike code or read directly in a recorded fixture. **[docs]** means read in official docs, the OpenAPI spec or SDK source. **[inferred]** means my conclusion from those.

## Questions

From the roadmap (Phase 0.5, Phase 2 "Message model", Phase 3 "Responses protocol"), as assigned:
1. Streaming event shapes: the full list of SSE event types to handle, with exact JSON.
2. Function calling: tool definitions, `call_id` vs item `id`, parallel calls, `function_call_output`, `tool_choice`.
3. Reasoning with `store: false`: `include`, reasoning item shape, and replay rules. Check the roadmap rule "tagged with provider and model, replayed when they match and dropped otherwise".
4. Structured output (`text.format` json_schema, strict) and refusals.
5. Usage fields, and where they appear in the stream.
6. Errors: HTTP error body, in-stream errors, rate-limit headers, `retry-after`.
7. xAI differences from OpenAI.
8. Does the Grok account proxy (`cli-chat-proxy.grok.com`) speak the same protocol?
9. Review `grok.rs` against these findings.
10. Fixture sources that need no paid calls.

Found along the way:
11. OpenAI assistant messages carry a `phase` field that must be replayed. This is more provider-specific data, besides reasoning items.
12. OpenAI, xAI, the Grok proxy and Gloo diverge enough that the shared protocol layer needs small per-provider **dialect** settings (see "Impact on the design").

## Findings

### 1. Streaming events

**Wire format.** The stream is standard SSE. Each event is sent as `event: <type>` followed by `data: <json>`, and the JSON `type` field always equals the SSE event name [verified: the spike asserts this on re-encoded fixtures]. Some sources omit the `event:` line (the openai-node tests, and the docs examples), so the parser must dispatch on the JSON `type` field, not on the SSE event name [docs: `openai-node/tests/lib/ResponseStream.test.ts`]. xAI's OpenAPI says the stream "is terminated by a `data: [DONE]` message" [docs: openapi.json `ModelRequest.stream`]. None of the recorded xAI or OpenAI Responses streams contains `[DONE]` [verified: all fixtures]. The parser should ignore `[DONE]` if it appears, and must never depend on it.

**Every event type** in the OpenAI SDK's `ResponseStreamEvent` union [docs: `openai-python/src/openai/types/responses/response_stream_event.py`], sorted by what nucleus must do with it:

| Handling | Events |
|---|---|
| **Must handle** | `response.created` (response id, model), `response.output_item.added`, `response.output_item.done`, `response.output_text.delta`, `response.function_call_arguments.delta`, `response.reasoning_summary_text.delta`, `response.reasoning_text.delta`, `response.refusal.delta`, `response.completed`, `response.incomplete`, `response.failed`, `error` |
| Useful but redundant (the `*.done` text is repeated in `output_item.done`) | `response.output_text.done`, `response.function_call_arguments.done`, `response.reasoning_summary_text.done`, `response.reasoning_text.done`, `response.refusal.done`, `response.content_part.added/done`, `response.reasoning_summary_part.added/done`, `response.output_text.annotation.added` |
| Ignore (log at `trace`) | `response.in_progress`, `response.queued`; every hosted-tool event: `response.web_search_call.*`, `response.file_search_call.*`, `response.code_interpreter_call*`, `response.image_gen_call.*`, `response.mcp_call*`, `response.mcp_list_tools.*`, `response.custom_tool_call_input.*`, `response.shell_call_*`, `response.compaction.compacting`, `response.audio.*`; xAI adds `response.x_search_call.*`, `response.code_execution_call.*`, `response.mcp_call_output.*` [docs: `vercel-ai/packages/xai/src/responses/xai-responses-api.ts`] |
| Unknown | Ignore and log. The openai-python fixture `unknown_event_and_fields` requires unknown events and unknown fields to pass through without error [docs: `tests/lib/responses/fixtures/websocket_scenarios.json`]. |

Vercel's xAI client also accepts `response.done` as a terminal event, equivalent to `response.completed` [docs: `xai-responses-api.ts`]. No recorded fixture contains it. Treat it as an alias.

**Recorded OpenAI stream** with a reasoning item and a function call. This is `gpt-5.1-codex-max` with `store:false`, from `vercel-ai/packages/openai/src/responses/__fixtures__/openai-reasoning-encrypted-content.1.chunks.txt`. Long ids and ciphertext are shortened with `…`, and repeated deltas are omitted:

```json
{"type":"response.created","sequence_number":0,"response":{"id":"resp_01830d…","status":"in_progress","usage":null, "…":"…"}}
{"type":"response.output_item.added","sequence_number":2,"output_index":0,"item":{"id":"rs_01830d…","type":"reasoning","encrypted_content":"gAAAAABpPDIUph8c…","summary":[]}}
{"type":"response.reasoning_summary_part.added","sequence_number":3,"item_id":"rs_01830d…","output_index":0,"summary_index":0,"part":{"type":"summary_text","text":""}}
{"type":"response.reasoning_summary_text.delta","sequence_number":4,"item_id":"rs_01830d…","output_index":0,"summary_index":0,"delta":"**Calcul","obfuscation":"2ajSAGID"}
{"type":"response.reasoning_summary_text.done","sequence_number":36,"item_id":"rs_01830d…","output_index":0,"summary_index":0,"text":"**Calculating step-by-st…"}
{"type":"response.output_item.done","sequence_number":38,"output_index":0,"item":{"id":"rs_01830d…","type":"reasoning","encrypted_content":"gAAAAABpPDIVOKrs…","summary":[{"type":"summary_text","text":"**Calculating step-by-st…"}]}}
{"type":"response.output_item.added","sequence_number":39,"output_index":1,"item":{"id":"fc_01830d…","type":"function_call","status":"in_progress","arguments":"","call_id":"call_AB6AaRZ1FYZB2RwS6A5vbdqn","name":"calculator"}}
{"type":"response.function_call_arguments.delta","sequence_number":40,"item_id":"fc_01830d…","output_index":1,"delta":"{\"","obfuscation":"hKtg6o5cWUSdCa"}
{"type":"response.function_call_arguments.done","sequence_number":53,"item_id":"fc_01830d…","output_index":1,"arguments":"{\"a\":12,\"b\":7,\"op\":\"add\"}"}
{"type":"response.output_item.done","sequence_number":54,"output_index":1,"item":{"id":"fc_01830d…","type":"function_call","status":"completed","arguments":"{\"a\":12,\"b\":7,\"op\":\"add\"}","call_id":"call_AB6AaRZ1FYZB2RwS6A5vbdqn","name":"calculator"}}
{"type":"response.completed","sequence_number":55,"response":{"id":"resp_01830d…","status":"completed","output":["…"],"usage":{"input_tokens":134,"input_tokens_details":{"cached_tokens":0},"output_tokens":28,"output_tokens_details":{"reasoning_tokens":0},"total_tokens":162}}}
```

The final turn in the same file shows a text message:

```json
{"type":"response.output_item.added","sequence_number":2,"output_index":0,"item":{"id":"msg_01830d…","type":"message","status":"in_progress","content":[],"role":"assistant"}}
{"type":"response.content_part.added","sequence_number":3,"item_id":"msg_01830d…","output_index":0,"content_index":0,"part":{"type":"output_text","annotations":[],"logprobs":[],"text":""}}
{"type":"response.output_text.delta","sequence_number":4,"item_id":"msg_01830d…","output_index":0,"content_index":0,"delta":"The","logprobs":[],"obfuscation":"o6siFnnjPfOpq"}
{"type":"response.output_text.done","sequence_number":12,"item_id":"msg_01830d…","output_index":0,"content_index":0,"text":"The final result is **570**.","logprobs":[]}
{"type":"response.output_item.done","sequence_number":14,"output_index":0,"item":{"id":"msg_01830d…","type":"message","status":"completed","content":[{"type":"output_text","annotations":[],"logprobs":[],"text":"The final result is **570**."}],"role":"assistant"}}
```

**Recorded xAI stream** (`grok-code-fast-1`, `store:false`, from `vercel-ai/packages/xai/src/responses/__fixtures__/xai-text-with-reasoning-streaming-store-false.1.chunks.txt`). It uses the same event names and nesting as OpenAI, with uuid-style ids and no `obfuscation` field:

```json
{"sequence_number":2,"type":"response.output_item.added","item":{"id":"rs_0b824fe9-3250-2588-0bbf-0810402fc822","summary":[],"type":"reasoning","status":"in_progress"},"output_index":0}
{"sequence_number":4,"type":"response.reasoning_summary_text.delta","delta":"First","item_id":"rs_0b824fe9-…","output_index":0,"summary_index":0}
{"sequence_number":70,"type":"response.output_item.done","item":{"id":"rs_0b824fe9-…","summary":[{"text":"First, the question is: …","type":"summary_text"}],"type":"reasoning","status":"completed","encrypted_content":"WjtnRFdCMihgUoZ7…"}}
{"sequence_number":73,"type":"response.output_text.delta","content_index":0,"delta":"###","item_id":"msg_0b824fe9-…","output_index":1,"logprobs":[]}
{"sequence_number":654,"type":"response.completed","response":{"id":"0b824fe9-3250-2588-0bbf-0810402fc822","model":"grok-code-fast-1","store":false,"usage":{"input_tokens":216,"input_tokens_details":{"cached_tokens":192},"output_tokens":831,"output_tokens_details":{"reasoning_tokens":253},"total_tokens":1047,"num_sources_used":0,"num_server_side_tools_used":0}, "…":"…"}}
```

**Terminal and error events**:

```json
{"type":"response.incomplete","response":{"id":"resp_123","status":"incomplete","incomplete_details":{"reason":"max_output_tokens"},"usage":null, "…":"…"},"sequence_number":1}
{"type":"response.failed","sequence_number":3,"response":{"id":"resp_689bcf…","status":"failed","error":{"code":"rate_limit_exceeded","message":"Rate limit reached for gpt-5.1 … Please try again in 11.054s. …"},"usage":null, "…":"…"}}
{"type":"error","code":"ERR_SOMETHING","message":"Something went wrong","param":null,"sequence_number":1}
{"type":"error","sequence_number":2,"error":{"type":"insufficient_quota","code":"insufficient_quota","message":"You exceeded your current quota, …","param":null}}
```

Sources for these four: `response.incomplete` is the OpenAPI `x-oaiMeta` example [docs: openapi.yaml `ResponseIncompleteEvent`]. The `response.failed` shape was recorded by codex [docs: `codex-api/src/sse/responses.rs` test `rate_limit_error_preserves_retry_delay`]. The flat `error` is the OpenAPI example. The **nested** `error` is a recorded live OpenAI stream, followed by `response.failed` [verified: `openai-error.1.chunks.txt`]. The two `error` shapes disagree, so the parser must accept both: read `event.error` when it is an object, and the event itself otherwise. Codex and openai-python both read the nested form [docs: `_streaming.py` reads `data["error"]`].

**Rules the protocol layer should follow:**
- **`response.output_item.done` is the source of truth.** Deltas are only for live display. In the OpenAI SDK, the docs for `output_item.added` say: "For reasoning items, `encrypted_content` may be incomplete while the item is in progress. Use the reasoning item from the corresponding `response.output_item.done` event" [docs: `response_output_item_added_event.py`]. This is verified: in 2 recorded OpenAI streams, `encrypted_content` changed between `added` and `done`. Its length went from 844 to 1060 characters in `openai-reasoning-encrypted-content.1`, and it also changed in `programmatic-tool-calling.1` (`gpt-5.6-sol`) [verified: spike]. Codex ignores `function_call_arguments.delta` completely and builds tool calls only from `output_item.done` [docs: `responses.rs`, `process_responses_event`].
- **Order items by `output_index`, not by arrival.** In a recorded xAI stream with server-side search tools, the `output_item.done` events arrived in the order 0,4,5,2,3,1,6 [verified: `xai-x-search-tool.chunks.txt`, spike]. After sorting by `output_index`, the spike's ids matched `response.completed.output` exactly.
- **Final output: prefer `response.completed.output`; if it is null, missing or empty, rebuild it from the `output_item.done` items.** openai-python fixed a bug for this case ("preserve finalized output on null response completion", openai-python CHANGELOG, #3345) and tests null, missing and empty terminal outputs [docs: `tests/lib/responses/test_null_output.py`]. Gloo's `response.completed` carries no `output[]` at all (see `gloo.md`), so the rebuild path is required anyway.
- **A stream that ends without `response.completed`, `response.incomplete` or `response.failed` is an error.** Codex reports "stream closed before response.completed" [docs: `responses.rs`]. `grok.rs` treats it as success (see Q9).
- **Some fields can be null or missing.** Delta events can carry an `obfuscation` field (OpenAI only, on by default; `stream_options.include_obfuscation: false` turns it off) [docs: `response_create_params.py` `StreamOptions`]. `logprobs` is `[]` from OpenAI, but `null` in xAI's documented examples [docs: xAI openapi.json response example]. Reasoning items from OpenAI have no `status` field [verified: fixture]. Use lenient `Option` and `#[serde(default)]` fields everywhere.
- **Decode UTF-8 once per complete SSE event, never once per network chunk.** The spike's byte-level parser produced identical results for all 38 files across 4 chunkings, with both LF and CRLF [verified: spike, "chunking-independent: true" for 38/38].

### 2. Function calling

- **Tool definition (OpenAI and xAI): the flat form.** `{"type":"function","name":"get_weather","description":"…","parameters":{…JSON Schema…},"strict":true}` [docs: `function_tool_param.py`; xAI function calling page, xAI openapi.json `ModelTool`/`FunctionDefinition`]. OpenAI's SDK marks `strict` as required-but-nullable. Responses "will attempt to normalize your schema into strict mode when possible" and falls back to non-strict when it can't [docs: OpenAI function calling guide]. Send `strict` explicitly so behavior does not depend on that normalization. xAI: arguments "strictly conform to the tool's input JSON Schema (the `strict` flag is implicitly always `true`)", with a maximum of 350 tools [docs: xAI structured outputs and function calling pages]. **Gloo differs:** it requires the nested Chat Completions form `{"type":"function","function":{…}}` (see `gloo.md`).
- **`id` vs `call_id`.** `id` (`fc_…`) is the output item's id. `call_id` (`call_…`) is the id that pairs the call with its output [docs: function calling guide: "a `call_id` (used later to submit the function result)"]. Stream events refer to the item by `item_id`, which equals `id`. The `call_id` appears only on the item in `output_item.added/done` [verified: fixture]. **The `ToolCall.id` in nucleus should be `call_id`.** Keep the item `id` as provider data for replay. Don't assume id prefixes: xAI uses ids like `rs_<uuid>` and `msg_<uuid>_us-east-1`, and its response id has no `resp_` prefix [docs: xAI generate-text replay example; verified: xAI fixtures].
- **Parallel calls.** Several `function_call` items appear in one response, each with its own `output_index` and `call_id`. On both OpenAI and xAI, parallel calls are on by default; `parallel_tool_calls:false` limits a response to zero or one call [docs: OpenAI guide; xAI function calling page]. Two quirks:
  - OpenAI models can occasionally emit an internal wrapper call named `parallel` with `{"tool_uses":[{"recipient_name":"functions.weather","parameters":{…}},…]}`. Vercel expands it when no user tool has that name [docs: `@ai-sdk/openai` `expand-parallel-tool-call.ts`, CHANGELOG 6be0f51; the fixture `parallel-tool-call-wrapper.1.chunks.txt` is hand-made]. Treat this as a known quirk, handled in the agent loop as "unknown tool" unless it is seen in practice.
  - xAI docs: "With streaming, the function call is returned in whole in a single chunk, not streamed across chunks" [docs: xAI function calling page]. xAI does send `function_call_arguments.delta/done` events; Vercel had to add them after validation failures [docs: `@ai-sdk/xai` CHANGELOG 3.0.54]. A design based on `output_item.done` handles both behaviors.
- **Returning results.** Append the model's `function_call` item, then `{"type":"function_call_output","call_id":"call_…","output":"<string>"}` [docs: OpenAI guide; xAI openapi.json `FunctionToolCallOutput`]. `output` may also be a list of `input_text`/`input_image`/`input_file` parts on both providers [docs: `response_input_param.py` `FunctionCallOutput`; xAI openapi.json]. With `store:false`, the `function_call` item itself must be sent back, so it has to be kept in history (inferred from the stateless model; Gloo has the same rule).
- **`tool_choice`**: `"auto"` (default), `"none"`, `"required"`, `{"type":"function","name":"…"}`, and on OpenAI also `{"type":"allowed_tools","mode":"auto"|"required","tools":[…]}` [docs: `tool_choice_*.py`; OpenAI guide]. xAI's OpenAPI uses the same flat `{"type":"function","name"}` (`ModelToolChoice`). But the xAI function-calling page shows the Chat Completions form `{"type":"function","function":{"name":"…"}}`. This conflict is open. Gloo documents only the string forms.
- **Reasoning models + tools:** "any reasoning items returned in model responses with tool calls must also be passed back with tool call outputs" [docs: OpenAI function calling guide]. The reasoning guide is softer ("we highly recommend"), and adds: "ensure all items between the last user message and your function call output are passed into the next response untouched" [docs: reasoning guide, "Keeping reasoning items in context"].

### 3. Reasoning with `store: false`

**The reasoning item** (both providers): `{"type":"reasoning","id":"rs_…","summary":[{"type":"summary_text","text":"…"}],"content":[{"type":"reasoning_text","text":"…"}]?,"encrypted_content":"…"?,"status":…?}` [docs: `response_reasoning_item.py`; xAI openapi.json `Reasoning`]. `content` (raw reasoning text) appears on some models, streamed as `response.reasoning_text.delta`. `summary` is streamed as `reasoning_summary_text.delta`, and `summary_index` separates its parts. Map both to nucleus's "thinking" event, keeping a flag for summary vs raw.

**Getting `encrypted_content`:**
- **OpenAI (current docs):** "reasoning items in the response's `output` array include an `encrypted_content` property by default" in stateless mode (`store:false` or ZDR). "The API still accepts the legacy `reasoning.encrypted_content` value in `include` for compatibility, but doesn't require it" [docs: reasoning guide, "Encrypted reasoning items"]. Summaries need opt-in (`reasoning.summary: "auto"|"concise"|"detailed"`) [docs: same].
- **xAI:** Only `grok-4.7` returns `encrypted_content` without asking ("Always returned for grok-4.7 … whether or not `include` lists it"). Other models need `include: ["reasoning.encrypted_content"]` [docs: xAI reasoning page]. This is verified on `grok-code-fast-1`: the `store:true` fixture without include has no `encrypted_content`, and the `store:false` fixture has it (Vercel adds the include whenever `store:false`) [verified: fixtures; docs: `xai-responses-language-model.ts` lines 266–277].
- **Both default to `store: true`** (OpenAI: "Defaults to true when omitted"; xAI: `"store": … "default": true`, kept for 30 days) [docs: `response_create_params.py`; xAI openapi.json, generate-text page]. nucleus keeps history client-side, so it should **always send `store:false` and `include:["reasoning.encrypted_content"]`** on OpenAI, xAI and the Grok proxy. Gloo has neither field (see `gloo.md`).

**Replay rules:**
- **What to replay.** Send the reasoning item back **verbatim from `output_item.done`**: `id`, `summary`, `encrypted_content`. OpenAI: "For stateless reasoning-model requests, preserve every item in the response's `output` array … Replaying the complete output keeps reasoning items and assistant `phase` values intact" [docs: conversation-state guide]. xAI's replay example sends `{"id":"rs_…","summary":[],"type":"reasoning","status":"completed","encrypted_content":"…"}` followed by the assistant `message` item [docs: xAI generate-text, "Adding encrypted thinking content"].
- **Is it required?** It is required within a tool loop on OpenAI reasoning models ("must also be passed back", see Q2). Across user turns it is optional, but it decides quality. OpenAI's `reasoning.context` setting (`current_turn` | `all_turns`; the GPT-5.6 family defaults to `all_turns`) controls whether earlier turns' reasoning is used. It "has an effect only when the request has access to earlier response items" [docs: reasoning guide; `shared_params/reasoning.py`].
- **Same model?** "Persisted reasoning can be reused only within the same model family … reasoning does not carry between the GPT-5.6 and GPT-5.5 families. When you switch model families, the API omits incompatible reasoning from the model's context" [docs: reasoning guide]. So on OpenAI, a mismatched item is **silently ignored, not rejected**. xAI documents nothing about cross-model replay (open).
- **What if it is dropped?** OpenAI documents no error when it is dropped. The model just loses its prior reasoning. Its systems also "smartly ignore any reasoning items that aren't relevant" [docs: reasoning guide]. A reasoning item **without** `encrypted_content` cannot be replayed with `store:false`. Vercel skips it with a warning [docs: `convert-to-openai-responses-input.ts`, `convert-to-xai-responses-input.ts`], and Cherry Studio removes such items before calling the Grok proxy ("Unsigned reasoning from other providers cannot be replayed") [docs: `grokCli.ts`].
- **Other provider-specific output must also be replayed:** OpenAI assistant `message` items carry `phase: "commentary"|"final_answer"`. "For models like `gpt-5.3-codex` and beyond, when sending follow-up requests, preserve and resend phase on all assistant messages — dropping it can degrade performance" [docs: `response_output_message.py`; verified: `openai-phase.1.chunks.txt` has both phases]. `compaction` items (`encrypted_content`) and `configuration_update` input items are further OpenAI-only items [docs: `response_compaction_item.py`; reasoning guide, "Change reasoning mid-conversation"].

**The roadmap rule "tagged with provider and model, replayed when they match and dropped otherwise" holds, with four corrections:**
1. **The unit is the raw output item**, not reasoning text: reasoning items with `encrypted_content`, plus the `phase` on assistant messages, plus compaction items. Store each as opaque JSON exactly as received in `output_item.done`, and replay it unchanged. [inferred from the above]
2. **The provider tag must name the endpoint dialect, not just the vendor:** `xai` (api.x.ai) and `grok-account` (the proxy) may or may not accept each other's ciphertext. This is open. Exact provider+model matching is safe, but it is stricter than OpenAI needs: OpenAI accepts reasoning from the same model family and silently drops incompatible items. Dropping on any model mismatch costs only reasoning quality, never an error. Keep the simple rule. A per-provider "same family" relaxation can come later if needed. [inferred]
3. **Never drop `function_call` / `function_call_output` pairs.** Only the opaque provider data is dropped. Tool calls and results become ordinary messages for any provider. [inferred; Gloo note agrees]
4. **Drop any reasoning item that has no `encrypted_content`**, even when provider and model match (this happens with `store:true` responses, or when `include` was missing). [docs: Vercel and Cherry Studio behavior above]

### 4. Structured output and refusals

- **Request (OpenAI):** `"text":{"format":{"type":"json_schema","name":"math_reasoning","schema":{…},"strict":true}}`. `name` must match `[a-zA-Z0-9_-]`, at most 64 characters [docs: `response_format_text_json_schema_config_param.py`; structured outputs guide]. Strict mode requires `additionalProperties:false` on every object, **every property listed in `required`** (use `["string","null"]` for an optional property), a root that is an object and not `anyOf`, and no `allOf`/`not`/`if`/`then`/`else`/`dependent*`. An unsupported schema is an error [docs: structured outputs guide, "Supported schemas"]. `json_object` and `text` also exist.
- **Request (xAI):** the same `text.format` shape works on Responses (xAI's own example sends `type`, `name`, `schema`, `strict:true`) [docs: xAI structured outputs page]. But `name`, `description` and `strict` are "Only included for compatibility": the schema is always enforced [docs: xAI openapi.json `ModelResponseFormat`]. xAI's subset is different and broader:
  - `additionalProperties` **defaults to false**.
  - Fields not listed in `required` are optional.
  - `allOf` with a single subschema is supported.
  - `format` is enforced for date/time/email/uuid/ip/uri.
  - Some keywords are best-effort; zero-variant `enum`/`anyOf` and boolean schemas are rejected with `400` [docs: xAI structured outputs page].
- **Impact for `agent.extract::<T>()`:** a schemars schema needs a per-provider transform. For OpenAI strict mode, make every property required with optional properties as nullable, and add `additionalProperties:false`. xAI takes the schema mostly as-is. The `Option<T>` → nullable+required rewrite is safe on both [inferred].
- **Refusals** arrive as a content part inside the assistant `message` item: `{"type":"refusal","refusal":"…"}`, streamed as `response.refusal.delta` / `response.refusal.done` (with `item_id`, `output_index`, `content_index`, `delta`) [docs: `response_output_refusal.py`, `response_refusal_delta_event.py`; structured outputs guide, "Refusals with Structured Outputs"]. xAI's output schema also has the `refusal` content type [docs: xAI openapi.json `OutputMessageContent`]. The response status is still `completed`. Other ways a structured output can fail: `response.incomplete` with `incomplete_details.reason` of `max_output_tokens` or `content_filter` [docs: guide, "Handle edge cases"; `response.py` `IncompleteDetails`]. `extract` should return a typed error, not try to parse a refusal as JSON.

### 5. Usage

- **Fields:** `usage.{input_tokens, input_tokens_details.cached_tokens, output_tokens, output_tokens_details.reasoning_tokens, total_tokens}` [docs: `response_usage.py`; xAI openapi.json `ModelUsage`].
  - OpenAI 3.28.0 also has `input_tokens_details.cache_write_tokens`, but the late-2025 recorded streams lack it, so it must be optional [verified: fixtures].
  - xAI adds `cost_in_usd_ticks` (1 USD = 10^10 ticks) and `cost_in_nano_usd`, plus `num_sources_used`, `num_server_side_tools_used`, `server_side_tool_usage_details` and `context_details` [docs: xAI openapi.json `ModelUsage`; release notes "Cost Tracking"].
- **Where:** usage appears only in the terminal event's `response.usage` (`completed`, `incomplete`, and possibly `failed`, where it is often `null`). In `response.created` and `in_progress`, `usage` is `null` [verified: spike checked that no fixture has non-null usage before the terminal event; docs: Vercel xAI CHANGELOG 648c8f3].
- **Accounting:** in all 39 recorded responses (OpenAI and xAI), `total_tokens == input_tokens + output_tokens`. So **`output_tokens` includes `reasoning_tokens`**, and **`input_tokens` includes `cached_tokens`** [verified: spike output "total=in+out" 39/39]. Two caveats:
  - xAI's OpenAPI example contradicts this (`input 32, output 9, reasoning 110, total 151`) [docs: openapi.json]. Recorded data wins, but check it on grok-4.7.
  - Vercel's xAI usage converter handles `cached_tokens > input_tokens` ("non-inclusive reporting") [docs: `convert-xai-responses-usage.ts` and test, CHANGELOG 7ccb902]. nucleus's `Usage` should clamp, never subtract into negative numbers.
- **No `finish_reason`:** Responses has no finish reason. Derive it: `incomplete` with `max_output_tokens` → length; `content_filter` → content filter; `completed` with any `function_call` item → tool calls; otherwise stop [docs: Vercel `map-openai-responses-finish-reason.ts`]. xAI's `incomplete_details.reason` values differ: `max_output_tokens`, `max_prompt_tokens`, `max_time_limit` [docs: xAI openapi.json `IncompleteDetails`]. OpenAI's are `max_output_tokens`, `max_messages`, `content_filter`, `steered` [docs: `response.py`]. Map unknown reasons to "other" and keep the raw string.

### 6. Errors, rate limits, retries

- **OpenAI HTTP error body:** `{"error":{"message":"…","type":"invalid_request_error","param":"temperature","code":null}}` [docs: openapi.yaml `ErrorResponse`/`Error`; verified: Vercel fixtures `openai-error.1.json`, `reasoning-model-temperature-error.json`]. The SDK maps by status: 400 BadRequest, 401 Auth, 403 PermissionDenied, 404, 409, 422, 429 RateLimit, ≥500 InternalServer [docs: `openai-python/src/openai/_client.py` `_make_status_error`].
- **xAI HTTP error body** is different. Responses errors are `{"code":"…","error":"<message string>"}`; other endpoints use the OpenAI shape or `{"error":"…"}`. Vercel parses a union of all three [docs: `vercel-ai/packages/xai/src/xai-error.ts`]. xAI's debugging page lists 400/401/403/404/405/415/422/429 without body shapes [docs].
- **In-stream errors:** `error` events (flat or nested, see Q1) and `response.failed` with `response.error.{code,message}`. In-stream codes include `server_error`, `rate_limit_exceeded`, `invalid_prompt`, `vector_store_timeout`, `bio_policy`, `misalignment_policy_violation` and image errors [docs: `response_error.py`]. Recorded or tested codes beyond that list: `insufficient_quota`, `context_length_exceeded`, `cyber_policy`, `server_is_overloaded`, `slow_down` [docs: codex `responses.rs` tests; verified: `openai-error.1.chunks.txt`]. Codex reads the retry delay from the message text ("Please try again in 11.054s") for in-stream `rate_limit_exceeded` [docs: codex test `rate_limit_error_preserves_retry_delay`].
- **Headers (OpenAI):** `Retry-After` (seconds), sent on temporary 429s and on `503 server_is_overloaded`. Also `x-ratelimit-{limit,remaining,reset}-{requests,tokens}`, and `x-ratelimit-*-project-tokens` [docs: rate limits guide]. The SDK also honors `retry-after-ms` and an HTTP-date `retry-after`, and obeys `x-should-retry: true|false`. It retries 408/409/429/≥500 with exponential backoff (0.5 s → 8 s cap, 2 retries) and refuses a `Retry-After` above 120 s [docs: `openai-python/src/openai/_base_client.py`, `_constants.py`]. **A 429 is not always retryable:** `insufficient_quota`, `credit_balance_exhausted` and `*_spend_limit_exceeded` / `*_usage_limit_exceeded` are 429s that need user action [docs: error-codes guide; rate-limits guide: "It does not mean that quota, billing, or other errors that require user action can be resolved by retrying"].
- **Headers (xAI):** limits are per model, in RPS and TPM, and exceeding one returns 429. xAI documents only exponential backoff and **no rate-limit or `Retry-After` headers** [docs: xAI rate-limits page]. Whether it sends them is open.
- **Retrying after output has started is not safe** (the user has already seen the deltas). Retry only before the first output event; after that, surface a retryable error to the caller [inferred].

### 7. xAI differences from OpenAI

| Area | OpenAI | xAI | Source |
|---|---|---|---|
| Event names and nesting | as Q1 | same; also `response.done` alias and hosted `x_search`/`code_execution` events | [verified: fixtures; docs: Vercel xai schema] |
| Function-call streaming | argument deltas | arguments in one piece, but the delta/done events still exist | [docs: xAI function calling; Vercel CHANGELOG] |
| Encrypted reasoning | returned by default when stateless | needs `include` (grok-4.7: always) | Q3 |
| `store` default | true | true (30 days); xai-sdk gRPC defaults false | [docs] |
| Reasoning effort | `none, minimal, low, medium, high, xhigh, max`, model-dependent; `none` is a 400 on some models | `low, medium, high (default), xhigh` (grok-4.6+; on grok-4.5 `xhigh` is treated as `high`); "Reasoning cannot be disabled". xai-sdk 1.20.0 also has `EFFORT_NONE`. Also a non-standard top-level `reasoning_effort` | [docs: `shared/reasoning_effort.py`, OpenAI reasoning guide; xAI reasoning page; `xai_sdk/chat.py`; xAI openapi.json] |
| Reasoning summary | opt-in via `reasoning.summary` | "Only included for compatibility … always return `detailed`"; summaries exposed for grok-4.7 | [docs: xAI openapi.json `ReasoningConfiguration`; reasoning page] |
| Sampling limits on reasoning models | `temperature` rejected on some models (`Unsupported parameter: 'temperature'`) | `presence_penalty`, `frequency_penalty`, `stop` rejected | [verified: Vercel fixture; docs: xAI reasoning page] |
| Structured output | strict subset, all fields required | always enforced, broader subset, optional fields allowed | Q4 |
| `tool_choice` forced | `{"type":"function","name"}` | the same in the spec; the docs page shows the nested form | Q2 |
| Error body | `{"error":{…}}` | `{"code","error":"string"}` | Q6 |
| Rate-limit headers | documented | not documented | Q6 |
| Usage extras | `cache_write_tokens` | `cost_in_usd_ticks`, sources/tool counts | Q5 |
| `incomplete_details.reason` | `max_output_tokens, max_messages, content_filter, steered` | `max_output_tokens, max_prompt_tokens, max_time_limit` | Q5 |
| Assistant `phase` | yes (must be replayed) | not present | Q3 |
| `prompt_cache_key` | cache-routing hint | "Plumbed to x-grok-conv-id … used for routing" | [docs: xAI openapi.json] |
| `instructions` | yes | yes, "cannot be used alongside `previous_response_id`" | [docs: xAI openapi.json] |
| `max_output_tokens` | includes reasoning | "Only applies to visible output tokens"; defaults to 128,000 | [docs: xAI openapi.json] |
| Ignored fields | — | `truncation`, `metadata`, `background` accepted but unsupported; `logprobs` ignored on grok-4.20+ | [docs: xAI openapi.json] |

Stateful vs `store:false`: both support `previous_response_id` and stateless replay. nucleus should use stateless replay only, so `Conversation` stays plain data and works across providers [inferred].

### 8. The Grok account proxy (`cli-chat-proxy.grok.com`)

**It looks like the same protocol, with stricter request rules.** This can't be settled without a live call.
- **Repo evidence:** `grok.rs` sends a Responses body to `{PROXY_BASE}/responses` and parses `response.output_text.delta`, `response.reasoning_*_text.delta`, `response.completed` and `response.failed`. The commit `1525ece` ("grok working now", 2026-09-26) followed `8a6198f` ("Grok connected, but not routing the correct model"), which suggests text streaming worked live. The code comment says "The proxy dispatches from x-grok-model-override, not the JSON model field" [docs: repo history; inferred].
- **Cherry Studio** (`grokCli.ts`, commit `d9c7839`, 2026-10-02): "xAI's proxy speaks the OpenAI Responses surface but with stricter edges than the generic adapter emits". It:
  - moves `system`/`developer` turns into top-level `instructions`;
  - "requires encrypted reasoning on replay", dropping reasoning items without `encrypted_content`;
  - always sets `store ??= false` and `include += reasoning.encrypted_content`;
  - removes `prompt_cache_retention`;
  - drops `reasoning` entirely for the legacy `grok-build` and `grok-composer-2.5-fast` models.

  Its headers: `x-grok-client-identifier`, `x-grok-client-version`, `x-xai-token-auth: xai-grok-cli`, `x-authenticateresponse: authenticate-response`, `x-grok-client-mode: interactive`, `x-grok-model-override` [docs: third-party source].
- **OpenClaw** routes the proxy through its `"openai-responses"` transport and sends `prompt_cache_key` to it [docs: `provider-routing.ts`, `docs/providers/xai.md`].
- **What only a live call can settle:**
  - whether the stream events match api.x.ai exactly (including function calls);
  - whether `role:"system"` in `input` is accepted (`grok.rs` sends it today);
  - the error body shape;
  - whether proxy reasoning ciphertext replays on api.x.ai and the reverse;
  - whether the OAuth token (scope includes `api:access`) works directly on `api.x.ai`. OmniRoute documents an `xai-oauth` provider on api.x.ai separate from the proxy [docs: `diegosouzapw/OmniRoute` provider reference, found via GitHub code search].
- **Tag proxy data separately** (`grok-account`) so it can't be replayed against api.x.ai until this is known.

### 9. `grok.rs` review (beyond the Known issues already listed)

Each item is something the new protocol layer must do differently:
1. **UTF-8 is decoded per network chunk** (`String::from_utf8_lossy(&chunk)`), so a multi-byte character split across chunks becomes U+FFFD. The spike reproduced this on 8 of 38 recorded fixtures with random chunking (2 to 13 corrupted characters per file) [verified: spike]. Buffer bytes and decode per event.
2. **The SSE parsing is not spec-compliant:**
   - each `data:` line is parsed alone, so multi-line `data` is never joined;
   - JSON parse failures are silently skipped (`continue`), so data is lost with no error;
   - a CR at the end of one chunk followed by an LF at the start of the next is normalized to two line breaks (low impact: it only splits frames early).

   [inferred from code; the spike's parser handles CRLF across chunks]
3. **No terminal-event check:** EOF without `response.completed`/`incomplete`/`failed` is reported as success, with a synthesized `done` chunk. It must be an error [docs: codex behavior].
4. **`response.incomplete` is mapped to a normal finish.** The reason (length, content filter, xAI time limit) is lost [inferred].
5. **Errors lose their structure:** HTTP errors become `Api(format!("…{status}: {body}"))`, and in-stream errors keep only the message. There is no code, no status, no `retry-after`/`x-should-retry`, no request id (`x-request-id`), and no retryable classification [inferred; see Q6].
6. **Usage is never read** from `response.completed` [inferred].
7. **`output_item.done` is ignored**, so the layer cannot capture item ids, `call_id`, final arguments, reasoning `encrypted_content` or `phase`. The new layer must build `Message`s from `output_item.done` [inferred; see Q1, Q3].
8. **`store:false` is sent without `include:["reasoning.encrypted_content"]`.** On every xAI model except grok-4.7, no encrypted reasoning comes back, so replay is impossible [docs: Q3].
9. **`temperature` is always sent.** Some OpenAI reasoning models reject it [verified: fixture]. Optional sampling fields should be sent only when set.
10. **`REASONING_EFFORTS` is hard-coded to xAI's values** and lacks `none`, `minimal` and `max`. Effort should pass through to the provider, which validates it; or the per-provider dialect should hold the list [docs: Q7].
11. **`prompt_cache_key`, `x-grok-conv-id` and `x-grok-session-id` use one id per provider instance.** In the new design a provider is shared across users and conversations. These must come from a per-conversation id [inferred; ties to the roadmap's stateless `Agent`].
12. **The system prompt is sent as an `input` message with `role: system`.** Use top-level `instructions` instead: OpenAI and xAI accept it, the proxy reportedly requires it, and it matches the roadmap rule that "the system prompt is added to each request" [docs: Q8].
13. **Remove the Chat Completions fallbacks** (`choices[0].delta.content`) and the non-streaming full-payload fallback inside the stream parser. Each protocol should parse only its own shapes. Unknown events go to `trace` [inferred].
14. **Tools are sent without `strict`**, and there is no `tool_choice`, `parallel_tool_calls`, `max_output_tokens` or `text.format` [inferred].
15. **The `User-Agent` header** is `xai-grok-cli` on the proxy path. Whatever the final value, it must be a generic product string with no user identifiers (coordinator rule) [inferred].

### 10. Fixture sources (no paid calls)

Best sources: **recorded live streams** in the Vercel AI SDK (Apache-2.0; keep a NOTICE when copying). The `.chunks.txt` format has one JSON event per line; wrap each line as `event: <type>\ndata: <line>\n\n` (the spike's `encode()` does this).
- **OpenAI**, `https://github.com/vercel/ai/tree/main/packages/openai/src/responses/__fixtures__/`:
  - `openai-reasoning-encrypted-content.1.chunks.txt`: 4-turn tool loop with `store:false`, reasoning with encrypted content, and function-call deltas;
  - `openai-error.1.chunks.txt` + `openai-error.1.json`: nested `error` event and `insufficient_quota`;
  - `openai-phase.1.chunks.txt`: `phase` commentary and final answer. Its text deltas are **trimmed**: they don't add up to `output_text.done`; the same is true of `openai-shell-container.1`;
  - `openai-tool-search.1.chunks.txt`, `programmatic-tool-calling.{1,2}.chunks.txt` (`gpt-5.6-sol`), `openai-web-search-tool.1.chunks.txt`, `openai-compaction.1.chunks.txt`;
  - `reasoning-model-temperature-error.json` (an HTTP 400 body).
  - Avoid `parallel-tool-call-wrapper.1.chunks.txt`: it is hand-made and contains invalid JSON lines [verified: spike].
- **xAI**, `https://github.com/vercel/ai/tree/main/packages/xai/src/responses/__fixtures__/`: `xai-text-streaming.1`, `xai-text-with-reasoning-streaming.1` (store true, no ciphertext), `xai-text-with-reasoning-streaming-store-false.1` (ciphertext), `xai-web-search-tool.1`, `xai-x-search-tool` (out-of-order items), `xai-image-generation-tool.1` (`.chunks.txt`). There is **no recorded xAI function-call stream**: the Vercel test inlines a synthetic one (`xai-responses-language-model.test.ts` around line 3990). Build one from the docs request and response examples plus the OpenAI shape, and mark it synthetic.
- **OpenAI docs:**
  - function-calling stream (10 lines, "Output events" block in `https://developers.openai.com/api/docs/guides/function-calling.md`);
  - per-event `x-oaiMeta.example` blocks in `openapi.yaml` (`ResponseErrorEvent`, `ResponseIncompleteEvent`, `ResponseCompletedEvent`, …);
  - the rate-limit header table (`rate-limits.md`).
- **xAI docs:** `https://docs.x.ai/openapi.json` (the `/v1/responses` 200 example), and the replay request in the generate-text page.
- **openai-python:**
  - `tests/lib/responses/fixtures/websocket_scenarios.json`: failed, incomplete, nested-error, unknown-event and malformed-JSON scenarios. They are WebSocket frames, but the event JSON is identical.
  - `tests/lib/responses/test_responses.py`: a non-streaming response recorded from the live API.
  - `tests/lib/responses/test_null_output.py`: null, missing and empty terminal output.
- **codex** `codex-rs/codex-api/src/sse/responses.rs` tests: realistic `response.failed` bodies (`rate_limit_exceeded` with "try again in", `context_length_exceeded`, `insufficient_quota`, `cyber_policy`, `misalignment_policy_violation`), and the "stream closed before response.completed" case.
- **openai-node** `tests/lib/streaming-named-error-events.test.ts`: `event: error` with a flat body, and an error after a successful delta.

## Impact on the design

Proposed ROADMAP.md changes:

1. **Phase 2, Message model:** replace the bullet with: "Tool calls carry the provider's `call_id`. Provider-specific output (reasoning items with encrypted content, the assistant-message `phase`, compaction items, Anthropic thinking blocks) is kept as opaque JSON, exactly as received, tagged with the provider dialect and model. It is replayed unchanged when both match and dropped otherwise; an item that can't be replayed (for example reasoning without `encrypted_content`) is dropped even on a match. Tool calls and tool results are never dropped." Also add "`Conversation` has a stable id", used for `prompt_cache_key` and the Grok proxy's conversation headers.
2. **Phase 3, Responses protocol:** replace "(built from the existing Grok code), configured for OpenAI, Grok and Gloo" with a list:
   - **One parser, configured by a `Dialect`.** OpenAI and xAI already diverge in places (forced `tool_choice` shape in the docs, error body, structured-output subset, effort values, incomplete reasons, encrypted-reasoning default, usage extras). With Gloo (nested tools, flattened stream payloads, no `store`/`include`) and the Grok proxy (`instructions` required, extra headers, model-override header), a small private `Dialect` struct of knobs keeps one implementation:
     - tool serialization: flat or nested;
     - stream payload: wrapped or flat;
     - whether to send `store`/`include`;
     - system prompt as `instructions` or as a message;
     - extra headers;
     - request fields to strip;
     - an error-body parser.

     The **response parser stays lenient** for the union of known shapes (both `error` shapes, `response.done`, `[DONE]`, null `logprobs`, missing `status`), so most differences need no knob.
   - **Request rules:** always `stream:true`, `store:false`, `include:["reasoning.encrypted_content"]` where the dialect supports them. The system prompt goes in `instructions`. Optional sampling fields are sent only when set. `reasoning.effort` passes through unvalidated.
   - **Stream rules:** a byte-level SSE parser; `output_item.done` (ordered by `output_index`) is the source of truth; deltas are only for live `Event`s; a terminal event is required; usage comes from the terminal event; the finish reason is derived as in Q5.
   - **Structured output:** a per-dialect schema transform for `extract::<T>()` (OpenAI strict: all properties required, optionals nullable, `additionalProperties:false`). Refusal and `incomplete` become typed errors.
3. **Phase 3, Reliability:** classify retryable errors by code as well as status (a 429 can be quota or billing, which is not retryable). Honor `retry-after-ms`, `retry-after` (seconds or HTTP-date) and `x-should-retry`, and cap long delays. Treat in-stream `response.failed` with `rate_limit_exceeded`/`server_is_overloaded`/`server_error` as retryable, parsing "try again in Ns" when there is no header. Retry only before the first output event.
4. **Phase 2 design gate, `Usage` cost question:** prefer cost reported by the provider (xAI `cost_in_usd_ticks`) when present; fall back to a price table. `Usage` must define `output_tokens` as including reasoning, and clamp a cached count larger than the input count.
5. **Known issues, `grok.rs`:** add Q9 items 1–3 (UTF-8 per chunk, non-compliant SSE, no terminal-event check) and items 8, 11 and 12 (`include` missing, per-instance cache key, system prompt as a message).
6. **Testing:** the SSE parser proptest should assert chunking independence (same events for any split of the bytes, LF or CRLF), as the spike does. Fixtures come from the Q10 list; vendored Vercel fixtures need an Apache-2.0 NOTICE.
7. **Phase 0.5 checkbox:** "xAI and OpenAI Responses API" can be ticked once this note is merged. The proxy question stays open (below) and does not block.

## Open questions

Only a live call can answer these:
- **Grok proxy:** does the stream match api.x.ai exactly, including function calls and the error body? Is `role:system` in `input` accepted, or is `instructions` required? Does proxy ciphertext replay on api.x.ai (and the reverse)? Does the OAuth token work directly on api.x.ai?
- **xAI:**
  - Does grok-4.7 send function-call arguments as one delta or several?
  - Which forced `tool_choice` shape is accepted on `/v1/responses` (flat or nested)?
  - Is `effort: "none"` accepted?
  - Does replaying reasoning from another Grok model, or from OpenAI, fail or get ignored?
  - What is the in-stream `error` event shape?
  - Are `x-ratelimit-*` or `Retry-After` headers sent?
  - Does grok-4.7's `output_tokens` include reasoning (the OpenAPI example says no; the recorded grok-code-fast-1 and grok-4-fast streams say yes)?
  - Do `response.done` or `[DONE]` ever appear?
- **OpenAI:** what exactly does the API return when a reasoning item is sent with an `id` but without `encrypted_content` under `store:false`? (Believed to be a 4xx; nucleus will never send it.)
- **Gloo:** whether reasoning items can be replayed (see `gloo.md`). It does not change the rule above, because unreplayable items are dropped.
