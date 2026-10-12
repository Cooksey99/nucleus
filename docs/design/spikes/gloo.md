# Spike: Gloo AI

Date: 2026-10-11. Versions and doc pages checked: the docs.gloo.com pages listed in its [`llms.txt`](https://docs.gloo.com/llms.txt) index, fetched 2026-10-11 (Mintlify has no doc versioning, so the date is the version). The OpenAPI specs [`openapi.json`](https://docs.gloo.com/openapi/openapi.json) and [`openapi.full.json`](https://docs.gloo.com/openapi/openapi.full.json), fetched the same day. Gloo's official sample repo [`GlooDeveloper/gloo-ai-docs-cookbook`](https://github.com/GlooDeveloper/gloo-ai-docs-cookbook) at commit `2457ed2`. One unauthenticated `GET https://platform.ai.gloo.com/platform/v2/models`, which the docs label "No authentication required" (179 models returned). No credentials were used and no paid calls were made. Local copies of everything are in `spikes/gloo/docs/`, which is not merged.

How each claim was established: **docs** means read in Gloo's docs or OpenAPI spec. **sample** means read in Gloo's official cookbook code. **verified** means checked with the one unauthenticated catalog call. **inferred** means my reading of the above, not stated by Gloo.

Warning about sources: web search summaries and the older `docs.ai.gloo.com` site still say Gloo uses OAuth2 client credentials only, with no API keys. That is out of date. The current docs and the [changelog (August 2026)](https://docs.gloo.com/changelog) say API keys replaced it.

## Questions

From the roadmap:
1. Auth: static API key, OAuth2 client credentials, or both? Token endpoint, grant type, scopes, lifetime, refresh, and how the credential is sent.
2. Base URLs and API surface: Responses, Chat Completions or proprietary? How OpenAI-compatible are the request fields, response fields and SSE event names?
3. Tools, streaming (including streamed tool calls), structured output, image and PDF input.
4. Models, how they are named, and whether there is a models-list endpoint.
5. Embeddings endpoint, models and dimensions.
6. Batch API.
7. Rate limits, error format, `Retry-After`, usage reporting, pricing.
8. Gloo-specific features a provider abstraction must allow for.

Found along the way:
9. Gloo has three endpoint families (guarded, direct, grounded). Which one should `gloo/` use?
10. How do Gloo's reasoning items fit the "replay provider-specific data" rule, given that `previous_response_id` is not supported?

## Findings

### 1. Auth: static API key. OAuth client credentials are deprecated

- **Current method: a static API key** (`sk_...`), created in Gloo AI Studio and sent as `Authorization: Bearer sk_...` on every request. There is no token exchange and no refresh. Keys don't expire. They can be revoked (takes effect within 60 seconds), deactivated and reactivated. Docs: [Generate API Keys](https://docs.gloo.com/studio/manage-api-credentials), [Authentication tutorial](https://docs.gloo.com/tutorials/authentication), [Libraries & SDKs](https://docs.gloo.com/api-guides/sdks-and-libraries) ("Your API key doesn't expire").
- The OpenAPI security schemes are both `http`/`bearer` with `bearerFormat: api-key`. The spec has no OAuth scheme. Docs: [openapi.json](https://docs.gloo.com/openapi/openapi.json) `components.securitySchemes`.
- The conventional env var is `GLOO_API_KEY`. Docs: [Quickstart](https://docs.gloo.com/getting-started/quickstart-developers).
- **OAuth2 client credentials are deprecated and will be "sunset by December 2026"**. Existing Client ID / Secret pairs keep working until then. Docs: [changelog, Aug 2026](https://docs.gloo.com/changelog), [Generate API Keys](https://docs.gloo.com/studio/manage-api-credentials).
- How the legacy flow works, for the record. Docs: [Authentication tutorial, "Deprecated: OAuth2 Client Credentials"](https://docs.gloo.com/tutorials/authentication), [API overview service map](https://docs.gloo.com/api-reference/general/overview).
  - Token endpoint: `POST https://platform.ai.gloo.com/oauth2/token`
  - Body: `application/x-www-form-urlencoded` with `grant_type=client_credentials&scope=api/access`
  - Client authentication: HTTP Basic `client_id:client_secret`
  - Response: `access_token`, `token_type`, `expires_in`. Tokens "expire after one hour".
  - Refresh: there is no refresh token. The samples request a new token about 60 s before expiry. Gloo's own [IDE proxy `src/auth.py`](https://github.com/GlooDeveloper/gloo-ai-ide-proxy) does the same, defaulting `expires_in` to 3600 (sample).

### 2. Base URLs and API surface: Responses is first-class, with real deviations from OpenAI

Gloo serves both an OpenAI-shaped **Responses API** (Gloo's recommended surface) and **Chat Completions**. Both come in three endpoint families. Docs: [Endpoint Types](https://docs.gloo.com/api-guides/endpoint-types), [API overview](https://docs.gloo.com/api-reference/general/overview).

| Family | Responses | Chat Completions | What Gloo adds |
|---|---|---|---|
| Guarded (Gloo's default) | `POST /ai/v2/guarded/responses` | `POST /ai/v2/guarded/chat/completions` | input guardrails, output moderation, `tradition`, "ministry-aware prompting"; a 6.5% platform fee |
| Direct | `POST /ai/v2/direct/responses` | `POST /ai/v2/direct/chat/completions` | nothing; the prompt reaches the model verbatim; no fee |
| Grounded | `POST /ai/v1/grounded/responses` | `POST /ai/v2/grounded/chat/completions` | RAG over the user's uploaded Gloo content |

- Host: `https://platform.ai.gloo.com`. For the OpenAI SDKs, the base URL is `https://platform.ai.gloo.com/ai/v2/guarded` or `.../ai/v2/direct`. Docs: [Responses](https://docs.gloo.com/api-guides/responses), [Direct Responses](https://docs.gloo.com/api-guides/direct-responses).
- Old paths `/ai/v2/responses` and `/ai/v2/chat/completions` still work as aliases. `/ai/v1/responses` is deprecated. Docs: [changelog, Sept/July 2026](https://docs.gloo.com/changelog).
- Gloo publishes no SDK of its own. It tells users to use the official OpenAI SDKs. Docs: [Libraries & SDKs](https://docs.gloo.com/api-guides/sdks-and-libraries).
- Auto-routing (`auto_routing`, `model_family`) exists **only on Chat Completions V2**. Responses takes a pinned `model` only. Docs: [Responses request format](https://docs.gloo.com/api-guides/responses).

**What matches OpenAI Responses** (docs: [Responses](https://docs.gloo.com/api-guides/responses), [V2ResponsesRequest / V3Request / ResponsesApiResponse schemas](https://docs.gloo.com/openapi/openapi.json)):
- Request fields: `input` (a string or an item array), `instructions`, `max_output_tokens`, `temperature`, `top_p`, `tools`, `tool_choice`, `parallel_tool_calls`, `stream`, `reasoning`, `prompt_cache_key`.
- Response: `id`, `object: "response"`, `created_at`, `model`, `status` (`completed|failed|in_progress|incomplete`), a typed `output[]` (`message`, `function_call`, `reasoning`, `image_generation_call`), and `usage`.
- Tool round trips: `function_call` items carrying `id`, `call_id`, `name` and `arguments` (a JSON string), and `function_call_output` items matched by `call_id`. Docs: [Tool Use](https://docs.gloo.com/api-guides/tool-use).
- Both request schemas set `additionalProperties: true`, so unknown fields are accepted. Whether Gloo forwards them or ignores them is not documented.

**Where Gloo differs from OpenAI.** Each of these matters for a shared Responses protocol layer:
1. **Tool definitions use the nested Chat Completions schema.** Gloo says to send `{"type":"function","function":{name,description,parameters}}` on the Responses endpoint. It warns that "the flat Responses tool schema is not reliably accepted across providers". Docs: [Tool Use, warning box](https://docs.gloo.com/api-guides/tool-use). OpenAI and xAI Responses use the flat `{"type":"function","name":...}` form, so the shared protocol needs a per-provider switch for tool serialization (inferred).
2. **No `previous_response_id`.** History is always client-side. The spec has no `store` or `include` field. Docs: [Responses](https://docs.gloo.com/api-guides/responses), [Direct Responses](https://docs.gloo.com/api-guides/direct-responses), [V2ResponsesRequest schema](https://docs.gloo.com/openapi/openapi.json). This fits nucleus, which already keeps state client-side (inferred).
3. **Stream payloads are flattened, not OpenAI's shape.** Gloo's tutorial says: "The live API sends **flattened payloads**: each `data:` line is the event object itself … with no nested `response` wrapper. The `response.completed` payload carries `usage` at the top level and does **not** include the `output[]` array." Docs: [Using the Guarded Responses API, Example 3 note](https://docs.gloo.com/tutorials/responses). Sample: [cookbook `responses/python/main.py`](https://github.com/GlooDeveloper/gloo-ai-docs-cookbook/tree/main/responses), which accepts both shapes.
   - The OpenAPI stream schemas agree with the tutorial. For example, the `response.output_item.added/done` payload *is* the item, with `type` set to `message|function_call|image_generation_call`. In OpenAI's shape the item is wrapped under `item`, and `type` holds the event name. Docs: `ResponsesStreamOutputItemEvent`, `ResponsesStreamCompletedEvent` in [openapi.json](https://docs.gloo.com/openapi/openapi.json).
   - The guide pages contradict this. They show OpenAI-style nested examples (`"item":{...}`, `"response":{...}`). Docs: [Responses streaming](https://docs.gloo.com/api-guides/responses), [Direct Responses streaming](https://docs.gloo.com/api-guides/direct-responses).
   - Consequence (inferred): the parser must dispatch on the SSE `event:` line, never on `data.type`. It must accept both payload shapes, and it must build the final output from the deltas and `output_item.done` events, because `response.completed` may carry no output.
4. **The event set is smaller.** The spec lists `response.output_item.added/done`, `response.content_part.added/done`, `response.output_text.delta`, `response.function_call_arguments.delta` and `response.completed`. The guides also mention `response.created`, `response.in_progress` and `response.output_text.done`. No `function_call_arguments.done`, reasoning-summary or refusal-delta events are documented. Docs: [openapi.json](https://docs.gloo.com/openapi/openapi.json) `x-event-payload-schemas`, [tutorial event table](https://docs.gloo.com/tutorials/responses). The `function_call_arguments.delta` payload is keyed by `output_index` only, with no `item_id` (docs: `ResponsesStreamFunctionArgumentsDeltaEvent`).
5. **Stream errors are documented inconsistently.** The guarded guide says errors arrive as events "prefixed with `error.`". The direct guide says "Errors are not delivered as SSE events", only as HTTP status. The cookbook treats any `error*` event, or any unknown event, as terminal. Docs: [Responses](https://docs.gloo.com/api-guides/responses), [Direct Responses](https://docs.gloo.com/api-guides/direct-responses). Sample: cookbook `main.py`. The [Errors page](https://docs.gloo.com/api-reference/general/errors) documents stream errors only for Chat Completions (`finish_reason: "error"` plus an `error` object). The Responses stream-error shape is **not documented**.
6. **Extra response fields:** `ignored_params` (`{params, model, reason}`, listing sampling params the model ignored). An `incomplete` response carries `incomplete_details` and a non-standard `suggestion`. Grounded responses add `sources_returned`. Docs: `ResponsesIgnoredParams` in [openapi.json](https://docs.gloo.com/openapi/openapi.json), [Provider Failover](https://docs.gloo.com/api-guides/provider-failover), [Grounded Responses](https://docs.gloo.com/api-guides/grounded-responses).
7. **Extra request fields:** `frequency_penalty`, `presence_penalty`, `stop`, `response_format`, `image_generation`, and `tradition` (guarded only). Docs: [V2ResponsesRequest](https://docs.gloo.com/openapi/openapi.json).

### 3. Capabilities

- **Tools / function calling: yes**, on Responses and Chat Completions, using the nested tool schema described above. `tool_choice` accepts a string such as `"required"`. The object form for forcing one specific function is not documented. Docs: [Tool Use](https://docs.gloo.com/api-guides/tool-use). 139 of the 149 generation models report `supports_tools: true` (verified: catalog).
- **Streaming: yes.** Every generation model reports `supports_streaming: true` (verified: catalog). Usage is always included in `response.completed` without asking for it. Docs: [Responses streaming](https://docs.gloo.com/api-guides/responses).
- **Streamed tool calls: yes, but thinly documented.** There is a `response.function_call_arguments.delta` event, and `output_item.added/done` carries `function_call` items (`call_id`, `name`, `arguments`). Docs: [Direct Responses streaming](https://docs.gloo.com/api-guides/direct-responses), [openapi.json](https://docs.gloo.com/openapi/openapi.json). No full streamed tool-call example exists in the docs or the cookbook. Whether `output_item.added` already carries `name` and `call_id` is open.
- **Structured output: yes on guarded; unclear on direct.**
  - Guarded accepts OpenAI's `text.format` (`{"type":"json_schema","name","schema","strict"}`, `json_object` or `text`), plus a `response_format` that "takes precedence" when both are sent. Docs: `text` description in `V2ResponsesRequest`, [openapi.json](https://docs.gloo.com/openapi/openapi.json).
  - The direct schema (`V3Request`) lists only `response_format` ("Structured-output / JSON schema controls") and has no `text` field. Docs: same spec.
  - The shape of `response_format` on Responses is not documented (flat or Chat-style `json_schema` wrapper).
- **Image input: yes.** Use `{"type":"input_image","image_url": <https URL or data: URI>}` inside a user content array. Docs: [Responses multimodal](https://docs.gloo.com/api-guides/responses). 105 generation models list `image` in `input_modalities` (verified: catalog).
- **PDF / file input: not supported according to the docs.** No `input_file` is documented anywhere. No generation model lists `file` in `input_modalities`; only two Gemini embedding models do (verified: catalog). Treat it as unsupported until shown otherwise (inferred).
- **Reasoning:** there is a `reasoning` request object ("e.g. effort"). Reasoning output items carry `summary` only: no `encrypted_content`, and no `include` parameter. Docs: `ResponsesReasoning` and `V2ResponsesRequest`, [openapi.json](https://docs.gloo.com/openapi/openapi.json). Whether reasoning items can be sent back in `input` is not documented.
- **Image generation:** this is out of scope for nucleus, but `image_generation_call` items can appear in `output[]`, so the parser must skip them or keep them as raw data. Docs: [Responses](https://docs.gloo.com/api-guides/responses).

### 4. Models and naming

- **IDs are `gloo-<vendor>-<model>`**, for example `gloo-anthropic-claude-sonnet-4.6`, `gloo-openai-gpt-5-mini`, `gloo-google-gemini-3.1-pro` or `gloo-xai-grok-4.7`. They contain dots and hyphens but never slashes. Each entry also has `provider_id` (`anthropic/claude-sonnet-4.6`), but requests use the `gloo-` ID. Verified: catalog. Docs: [Get Models V2](https://docs.gloo.com/api-reference/get-models/v2).
- The full nucleus model string would be `gloo/gloo-anthropic-claude-sonnet-4.6`. That is redundant but unambiguous (inferred).
- **Catalog endpoint: `GET https://platform.ai.gloo.com/platform/v2/models`, no auth.** It returns 179 models: 149 generation and 30 embedding (verified). Each entry has:
  - context and token limits
  - `input_modalities` and `output_modalities`
  - `supports_tools`, `supports_streaming`, `supports_vision`, `supports_reasoning`
  - `dimensions` (embedding models only)
  - `is_deprecated` and `replacement_model`
  - list **pricing** per 1K and 1M tokens for input, output, `cache_read` and `cache_write` (`5m`/`1h`)
- The catalog includes Anthropic, OpenAI, Google, xAI, Cohere, Mistral, Qwen, DeepSeek, Kimi and many others (verified).
- Models are retired with a `replacement_model`, and 14 are currently marked deprecated (verified). Docs: [Model Lifecycle](https://docs.gloo.com/api-guides/model-lifecycle).

### 5. Embeddings

- **`POST https://platform.ai.gloo.com/ai/v2/direct/embeddings`**, in OpenAI Embeddings shape. Docs: [Embeddings](https://docs.gloo.com/api-guides/embeddings), [Direct Embeddings reference](https://docs.gloo.com/api-reference/embeddings/direct).
  - Request: `model`, `input` (one string, or 1 to 2048 non-empty strings; token-ID arrays are not supported), `encoding_format` (`float`/`base64`), optional `dimensions`.
  - Response: `{object:"list", data:[{index, embedding}], model, usage:{prompt_tokens,total_tokens}}`. `usage` may be missing.
- **30 models, native dimensions from 384 to 4096.** Verified: catalog. Examples:
  - `gloo-openai-text-embedding-3-small` (1536) and `-3-large` (3072)
  - `gloo-baai-bge-m3` (1024)
  - `gloo-voyageai-voyage-4` (1024)
  - `gloo-qwen-qwen3-embedding-8b` (4096)
  - `gloo-sentence-transformers-all-minilm-l6-v2` (384)
- Some models accept `dimensions`, but which values each accepts is not in the catalog; a bad value returns `400`. Gloo always sends the request `encoding_format` to the provider; if the provider ignores it, Gloo returns `503`. There is no failover; a provider failure returns `503 PROVIDER_UNAVAILABLE` with `retryable: true`. Docs: [Embeddings](https://docs.gloo.com/api-guides/embeddings).
- Embeddings are billed per input token with no platform fee. Docs: [Embeddings pricing](https://docs.gloo.com/api-guides/embeddings).

### 6. Batch: none

There is no batch endpoint in either OpenAPI spec (active or full), in the docs index or in the changelog. The only "batch" features are Data Engine file uploads and the 2048-string embeddings input. Docs: [openapi.full.json](https://docs.gloo.com/openapi/openapi.full.json) path list, [llms.txt](https://docs.gloo.com/llms.txt), [changelog](https://docs.gloo.com/changelog).

### 7. Limits, errors, usage, pricing

- **There is no universal rate-limit contract.** "Do not assume that a `429` response includes `X-RateLimit-*`". Docs: [Rate Limits](https://docs.gloo.com/api-reference/general/limits). The kinds of 429:
  - `429 SPENDING_LIMIT_EXCEEDED` (code 1005): `retryable: false`, no `Retry-After`.
  - Endpoint-specific quotas: `429`, possibly with `Retry-After` (in seconds).
  - Upstream rate limits after a stream starts: an SSE error `2004 RATE_LIMIT`.
- Credit-funded usage is capped at **$100 per week** unless support raises it. Docs: [Quickstart](https://docs.gloo.com/getting-started/quickstart-developers).
- **Error object** on inference endpoints, documented for Completions: `{"error": {message, type, code (numeric), name, category, description, fault: client|provider|internal, retryable, trace_id}}`. It is often sent alongside a legacy `detail` field. Docs: [Errors, AI Error Object](https://docs.gloo.com/api-reference/general/errors).
- The body format varies by layer:
  - The gateway returns `{"error": "Missing Authorization", "code": ...}`.
  - FastAPI returns `422` with a `detail[]` array.
  - Some endpoints return a bare `{"detail": "..."}`.
  - Docs: same page.
- **A 403 is not always an auth error.** On guarded endpoints, a guardrail hard block returns `403` with `detail.type == "content_policy_violation"` and `retryable: false`. Docs: [tutorial, Guardrail Behavior](https://docs.gloo.com/tutorials/responses), [Completions V2 reference](https://docs.gloo.com/api-reference/completions/v2).
- **402 `INSUFFICIENT_CREDIT`** (code 1006) is not retryable. Docs: [Errors, 402](https://docs.gloo.com/api-reference/general/errors).
- **Provider failover is built in.** Gloo retries on another provider serving the same model, including before a stream starts, and for some models in the middle of a stream. Docs: [Provider Failover](https://docs.gloo.com/api-guides/provider-failover). Client retries are still recommended when `retryable: true`.
- **Usage fields:**
  - Responses: `usage.{input_tokens, output_tokens, total_tokens, input_tokens_details.cached_tokens, output_tokens_details.reasoning_tokens}`, always present on streams. Docs: `ResponsesUsage` in [openapi.json](https://docs.gloo.com/openapi/openapi.json).
  - The pricing formula includes a `cache_write_tokens` term, but that field is not in the Responses usage schema. Docs: [Prompt Caching](https://docs.gloo.com/api-guides/prompt-caching).
- **Pricing:** per-token list rates come from the public catalog. Guarded endpoints add a **6.5% platform fee**; direct and embeddings add none. Payment is prepaid credits (minimum $10). Docs: [Billing & Plans](https://docs.gloo.com/studio/billing), [Responses pricing](https://docs.gloo.com/api-guides/responses). An older blog post says 5.5%; the current docs win ([blog](https://gloo.com/resources/blog/introducing-gloo-ai-studio)).

### 8. Gloo-specific features a provider abstraction must allow for

- `tradition` (`evangelical|catholic|mainline|not_faith_specific`): a request body field, guarded only. Docs: [Responses](https://docs.gloo.com/api-guides/responses).
- **`X-Cache-TTL: 5m|1h`: a request header**, not a body field. It turns on Anthropic prompt caching; Gloo then adds the `cache_control` marker to the system prompt itself. Docs: [Prompt Caching](https://docs.gloo.com/api-guides/prompt-caching), plus the `X-Cache-TTL` header parameter on both Responses paths in [openapi.json](https://docs.gloo.com/openapi/openapi.json). OpenAI-family models use `prompt_cache_key`.
- Grounded Responses: `rag_publisher`, `sources_limit` (1–10) and `certainty_threshold` in the body, on a different path. The response carries `sources_returned`, or an `X-Sources-Returned` header when streaming. Docs: [Grounded Responses](https://docs.gloo.com/api-guides/grounded-responses), [changelog, July 2026](https://docs.gloo.com/changelog).
- Choice of endpoint family (guarded / direct / grounded); see question 9.
- Response extras (`ignored_params`, `suggestion`, `sources_returned`) only need to reach the raw response.
- Chat Completions V2 only: `auto_routing` and `model_family`, which mean no `model` field. Nucleus defers Chat Completions, so these would wait (inferred).

### 9. Which endpoint family should `gloo/` use? (inferred, needs a decision)

Guarded is what Gloo recommends and is the reason to pick Gloo at all: values alignment, moderation, `tradition`. It has three costs:
- It is not byte-faithful: Gloo adds prompt augmentation, and moderation can replace the model's answer with a static message.
- It charges 6.5% more.
- It returns `403` on content blocks.

Direct matches the nucleus promise that only the model string changes behavior. The request body is identical, so the choice is just a URL. Docs: [Endpoint Types](https://docs.gloo.com/api-guides/endpoint-types).

### 10. Reasoning replay (inferred)

The design says provider-specific reasoning items are "replayed when they match and dropped otherwise". Gloo reasoning items are summary-only, with no `encrypted_content`, so there is nothing opaque to replay. The safe default is to drop Gloo reasoning items from replayed `input`, unless a live call shows they are accepted and useful.

## Impact on the design

1. **Close the open decision "Gloo auth": static API key**, sent as a Bearer token from `GLOO_API_KEY`. Client credentials are deprecated and end by December 2026, before 0.2 would see real use.
   - Remove "OAuth client credentials" from the Phase 3 auth-layer checkbox and from the Architecture `Auth:` line. It becomes `Auth: API key | Grok account login [experimental]`.
   - Keep the auth type `#[non_exhaustive]` so OAuth can come back for Azure, Vertex or Bedrock.
   - With nothing left to refresh except Grok login, the "token caching and refresh shared" note shrinks to Grok only.
2. **Keep Gloo on the Responses protocol, but add a per-provider dialect** to the private protocol layer. The Architecture line becomes `Responses API → OpenAI, Grok (xAI), Gloo AI (dialect: nested tools, flattened stream)`. The dialect covers:
   - **Tool serialization:** flat (OpenAI, xAI) or nested Chat-style (Gloo).
   - **Stream parsing:** always dispatch on the SSE `event:` name. Accept both nested and flattened payloads. Build output from deltas and `output_item.done`, and never require `response.completed.output`. Handle `function_call_arguments.delta` keyed only by `output_index`. Treat `error*` events as terminal. Ignore unknown `response.*` events but log them with `tracing`.
   - **Structured output field:** `text.format` on guarded. `response_format` on direct is pending the open question below.
   - **No `previous_response_id`, `store` or `include`.** Drop reasoning items on replay to Gloo.
   - The Phase 0.5 xAI/OpenAI spike should say whether this dialect hook also covers xAI's differences.
3. **Mock fixtures for Gloo must cover both stream shapes,** nested as in the guides and flattened as in the OpenAPI spec and tutorial note, until a live capture settles which one is real. Add this to the Phase 3 shared test suite item.
4. **Add an endpoint option to the Gloo provider:** `Guarded` (default, matching Gloo's guidance) or `Direct`, set by a builder option or base-URL override. Grounded waits; for now it is reachable with a base-URL override plus `provider_options`. This needs a decision in the Phase 2 design note. The alternative is defaulting to `Direct` for the "only the model string changes" promise.
5. **Design rule 8 (escape hatch) must cover HTTP headers as well as body fields.** Gloo's prompt caching uses `X-Cache-TTL`. Proposed: `provider_options` plus `provider_headers`, or one options type with both. `tradition` and the grounded fields go through `provider_options`.
6. **Error kinds** in `nucleus::Error` need:
   - A content-policy / refused kind that a guarded `403 content_policy_violation` maps to, separate from auth.
   - Non-retryable billing errors: `402 INSUFFICIENT_CREDIT` and `429 SPENDING_LIMIT_EXCEEDED`. "429 means retry" must not be hard-coded.
   - Use Gloo's `error.retryable` and `fault` when present. Honour `Retry-After` when present.
   - Keep `trace_id` in the error chain.
7. **Batch:** Gloo has no batch API. In the Phase 3 batch item, "possibly Gloo" becomes "Gloo has none": it returns a clear unsupported error, or runs limited concurrent requests if explicitly asked.
8. **Model strings:** `gloo/<gloo model id>`, sent verbatim (for example `gloo/gloo-anthropic-claude-sonnet-4.6`). IDs never contain `/`, so they need nothing special in the grammar. Whether to also accept `gloo/anthropic-claude-sonnet-4.6` and add the prefix is a small sugar question for the model-string grammar in `core-api.md`.
9. **Usage cost (Phase 2 question):** Gloo's public catalog has machine-readable list prices, including cache read and write, plus capability flags. That supports "built-in price table that users can override", and Gloo could fill both `Usage` cost and `capabilities()`. The guarded 6.5% fee must be applied on top. Any runtime fetch has to stay optional so tests stay offline.
10. **Phase 3 image/PDF item:** Gloo supports images (URL or data URI) but not PDF. Report this through `capabilities()`.
11. **Phase 4:** the Gloo embedder is OpenAI-shaped on `/ai/v2/direct/embeddings`, so it can share the OpenAI embedder code with a different base URL. Chunk inputs to fit `max_input_tokens`, which can be as low as 512, and send at most 2048 strings per call. Always send `encoding_format: "float"`.

## Open questions

These need a live call (an API key and a few cents) unless noted.
- **Real stream shape** on guarded and direct Responses: flattened or nested? Does `response.created` appear on direct? Is there a `response.function_call_arguments.done`? Does `output_item.added` for a `function_call` already carry `call_id` and `name`?
- **Responses stream-error shape:** event name (`error`? `error.*`?) and payload. The direct docs say errors never arrive in-stream; check that.
- Is the **flat Responses tool schema** accepted for OpenAI-family models, and what is the object form of `tool_choice` for forcing one function (Chat-style or Responses-style)?
- **Structured output:** the shape of `response_format` on Responses, and whether direct honours `text.format` (its schema omits it, but it allows extra properties).
- Can `reasoning` items be sent back in `input`, and are `store` / `include` / `encrypted_content` accepted or silently dropped?
- Are `system` / `developer` role items in `input` accepted the same as `instructions`?
- **PDF input:** confirm that `input_file` is rejected, rather than just undocumented.
- Do `Retry-After` and any rate-limit headers appear on real inference `429`s?
- How much does guarded prompt augmentation ("ministry-aware prompting") change output for non-faith workloads? This informs the guarded vs direct default.
- Is `cache_write_tokens` reported anywhere in Responses `usage` for Anthropic models when `X-Cache-TTL` is set?
- The exact sunset date for OAuth client credentials ("by December 2026"). Gloo support could answer this without a live call.
