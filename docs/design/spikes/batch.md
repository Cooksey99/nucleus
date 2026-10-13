# Spike: Batch APIs

Date: 2026-10-11. Versions and doc pages checked:

- OpenAI: Batch guide <https://developers.openai.com/api/docs/guides/batch>, Batch API reference <https://developers.openai.com/api/reference/resources/batches/methods/create>, webhooks guide <https://developers.openai.com/api/docs/guides/webhooks>, Files reference <https://developers.openai.com/api/reference/resources/files/methods/create>, rate-limits guide <https://developers.openai.com/api/docs/guides/rate-limits>. `openai-openapi` `openapi.json` at commit `e95c0fe` (<https://github.com/openai/openai-openapi/blob/e95c0fe615f45a19e8878af923a729948445a6bd/openapi.json>). `openai-python` at `b7a1990` (`src/openai/types/batch_create_params.py`, `batch.py`, `webhooks/batch_*_webhook_event.py`).
- Anthropic: Batch processing guide <https://platform.claude.com/docs/en/build-with-claude/batch-processing>, rate limits <https://platform.claude.com/docs/en/api/rate-limits#message-batches-api>, structured outputs <https://platform.claude.com/docs/en/build-with-claude/structured-outputs#feature-compatibility>. `anthropic-sdk-python` at `b4b7916` (`src/anthropic/types/messages/*.py`).
- xAI: Batch API guide <https://docs.x.ai/developers/advanced-api-usage/batch-api>, REST reference <https://docs.x.ai/developers/rest-api-reference/inference/batches>, pricing <https://docs.x.ai/developers/pricing#batch-api-pricing>, model pages under <https://docs.x.ai/developers/models>. `xai-sdk-python` at `1d9e1df` (`src/xai_sdk/proto/v6/batch_pb2.pyi`).
- Gloo AI: docs index <https://docs.gloo.com/llms.txt>, full docs dump <https://docs.gloo.com/llms-full.txt>, rate limits <https://docs.gloo.com/api-reference/general/limits.md>.
- mistral.rs: crates.io `mistralrs` 0.8.1 (latest on crates.io, published 2026-04-02) source, plus docs on `master` at `3f2515e` (`docs/src/content/docs/guides/perf/throughput-tuning.mdx`, `guides/perf/paged-attention.mdx`, `examples/rust/advanced/batching.md`).

No API keys and no paid calls were used. All findings come from docs, specs and SDK source. Nothing was verified by running code. Each claim is tagged **[docs]** (read in docs, spec or SDK source) or **[inferred]**.

## Questions

1. OpenAI Batch API: batchable endpoints (is `/v1/responses` supported?), JSONL format and `custom_id`, file upload flow, completion window, limits, discount, statuses, cancellation, expiry, result and error files, ordering, retention.
2. Anthropic Message Batches: request format, limits, processing window, statuses, cancellation, results and retention, discount; do prompt caching, tools, thinking and structured output work inside batches?
3. xAI Batch API: does it exist, what shape does it have, per-model discounts, unsupported models, limits.
4. Gloo AI: is there a batch API?
5. mistral.rs: what do the docs say about running concurrent local requests?
6. Design synthesis: `BatchHandle` contents, a common status and progress model, polling or webhooks, partial results, cancellation, per-request errors, request-ID constraints, providers without batch support, and what doesn't map cleanly.

Found along the way:

7. The current default Grok models don't support batch at all, and the xAI discount is 20%, not 50%. This weakens the "cost savings" case for `grok/`.
8. xAI's REST reference and its guide disagree on whether batch requests can use the Responses shape. This matters because Nucleus plans to speak only the Responses protocol to xAI.

## Findings

### 1. OpenAI Batch API

**Endpoints.** `/v1/responses` is supported. The full list is `/v1/responses`, `/v1/chat/completions`, `/v1/embeddings`, `/v1/completions`, `/v1/moderations`, `/v1/images/generations`, `/v1/images/edits` and `/v1/videos`. **[docs]** [SDK `batch_create_params.py`](https://github.com/openai/openai-python/blob/b7a1990353baec896ee5e04b40b9f2dc2925a4f5/src/openai/types/batch_create_params.py), [API reference](https://developers.openai.com/api/reference/resources/batches/methods/create). The guide's list doesn't include `/v1/videos` yet. **[docs]** [guide](https://developers.openai.com/api/docs/guides/batch). One batch has one `endpoint`. **[docs]** same reference.

**Input format.** The input is a JSONL file with one request per line: `{"custom_id", "method", "url", "body"}`. `body` takes the same parameters as the target endpoint. Each `custom_id` must be unique, and each input file can target only one model. **[docs]** [guide](https://developers.openai.com/api/docs/guides/batch). No length or charset limit for `custom_id` is documented anywhere: the guide, the API reference and the OpenAPI spec all say nothing, and the spec has no schema for the line format. **[docs, absence]** [reference](https://developers.openai.com/api/reference/resources/batches), [spec](https://github.com/openai/openai-openapi/blob/e95c0fe615f45a19e8878af923a729948445a6bd/openapi.json). The guide has no example line for `/v1/responses`. That a Responses body (including `text.format` for structured output) is accepted as-is follows from "the body matches the endpoint". **[inferred]** [guide](https://developers.openai.com/api/docs/guides/batch). For moderations, the batch worker rejects `stream=true`. Nothing is documented about `stream`, `background`, `store` or `previous_response_id` for Responses bodies in a batch. **[docs]** same guide.

**Flow.**
1. Upload the file with `POST /v1/files`, `purpose=batch`.
2. Create the batch with `POST /v1/batches {input_file_id, endpoint, completion_window: "24h", metadata?, output_expires_after?}`.
3. Poll with `GET /v1/batches/{id}`.
4. Download `output_file_id` and `error_file_id` through the Files content endpoint.

**[docs]** [guide](https://developers.openai.com/api/docs/guides/batch), [reference](https://developers.openai.com/api/reference/resources/batches/methods/create). `metadata` allows up to 16 pairs, with keys up to 64 characters and values up to 512. **[docs]** same reference.

**Completion window.** Only `"24h"` is accepted. **[docs]** [SDK](https://github.com/openai/openai-python/blob/b7a1990353baec896ee5e04b40b9f2dc2925a4f5/src/openai/types/batch_create_params.py).

**Limits.** A batch holds at most 50,000 requests and 200 MB. Embeddings batches are also capped at 50,000 inputs. At most 2,000 batches can be created per hour. There is a per-model cap on enqueued prompt tokens: tokens of pending batches count against it and are released when the batch completes, and the values appear only in the org's console. There is no output-token limit. Batch rate limits are a separate pool from synchronous limits. **[docs]** [guide](https://developers.openai.com/api/docs/guides/batch), [rate limits](https://developers.openai.com/api/docs/guides/rate-limits). The docs don't say what happens when a batch exceeds the enqueued-token cap (rejected at validation, or a `failed` batch). **[docs, absence]** [rate limits](https://developers.openai.com/api/docs/guides/rate-limits). The Files API allows 512 MB per file and 2.5 TB per project. **[docs]** [Files](https://developers.openai.com/api/reference/resources/files/methods/create).

**Discount.** Batch costs 50% less than the synchronous APIs. **[docs]** [guide](https://developers.openai.com/api/docs/guides/batch).

**Statuses.** `validating`, `failed`, `in_progress`, `finalizing`, `completed`, `expired`, `cancelling`, `cancelled`. **[docs]** [spec `Batch.status`](https://github.com/openai/openai-openapi/blob/e95c0fe615f45a19e8878af923a729948445a6bd/openapi.json). The batch object has a timestamp for each transition, `request_counts {total, completed, failed}`, `errors.data[] {code, line, message, param}` for batch-level (validation) errors, and `usage`, which is filled only for batches created after 2025-09-07. **[docs]** [reference](https://developers.openai.com/api/reference/resources/batches/methods/create). `request_counts` has no separate expired or cancelled count. **[docs]** same.

**Cancellation.** `POST /v1/batches/{id}/cancel` moves the batch to `cancelling` for up to 10 minutes, then to `cancelled`. **[docs]** [guide](https://developers.openai.com/api/docs/guides/batch).

**Expiry.** A batch that hasn't finished in 24 hours becomes `expired`. Its unfinished requests are cancelled, its completed responses stay available and are billed, and the expired requests go to the error file with code `batch_expired`. **[docs]** [guide](https://developers.openai.com/api/docs/guides/batch).

**Results.** Each output line is `{id, custom_id, response: {status_code, request_id, body}, error}`. Failed requests go to the error file. Line order "may not match" input order, so callers must match on `custom_id`. **[docs]** [guide](https://developers.openai.com/api/docs/guides/batch). The docs don't say whether a request rejected with a 4xx appears in the output file (with `status_code` ≥ 400) or in the error file, or what code cancelled requests get. Treat any line with `status_code != 200` or a non-null `error` as failed. **[inferred]**

**Retention.** By default the output file is deleted 30 days after the batch completes. `output_expires_after {anchor: "created_at", seconds: 3600..2592000}` sets the expiry of the output and error files, anchored at the file's creation time, not the batch's. **[docs]** [guide](https://developers.openai.com/api/docs/guides/batch), [spec `BatchFileExpirationAfter`](https://github.com/openai/openai-openapi/blob/e95c0fe615f45a19e8878af923a729948445a6bd/openapi.json). The uploaded input file can have its own `expires_after`, but the reference shows it only in a curl example, so its default and allowed range are unclear. **[docs]** [Files](https://developers.openai.com/api/reference/resources/files/methods/create).

**Webhooks.** OpenAI sends `batch.completed`, `batch.failed`, `batch.expired` and `batch.cancelled` events. The payload is only `{data: {id}}`. Endpoints are configured per project in the dashboard. Delivery retries for up to 72 hours, duplicates are possible, and `webhook-id` is the idempotency key. **[docs]** [spec `WebhookBatch*`](https://github.com/openai/openai-openapi/blob/e95c0fe615f45a19e8878af923a729948445a6bd/openapi.json), [webhooks guide](https://developers.openai.com/api/docs/guides/webhooks).

### 2. Anthropic Message Batches

**Request format.** `POST /v1/messages/batches {requests: [{custom_id, params}]}`. `params` is a non-streaming Messages create body, and the request is plain JSON with no file upload. **[docs]** [SDK `batch_create_params.py`](https://github.com/anthropics/anthropic-sdk-python/blob/b4b7916deaf4e5570cd627b1ae7ee4394a4de39f/src/anthropic/types/messages/batch_create_params.py). `custom_id` must match `^[a-zA-Z0-9_-]{1,64}$` and be unique within the batch. **[docs]** [guide](https://platform.claude.com/docs/en/build-with-claude/batch-processing). Requests in one batch may mix types, and each one names its own model. **[docs]** same guide. `params` are validated asynchronously: validation errors come back as per-request `errored` results after the batch ends, not at creation. **[docs]** same guide.

**Limits.** A batch holds at most 100,000 requests or 256 MB, whichever comes first. Exceeding the size limit gives `413 request_too_large`. **[docs]** [guide](https://platform.claude.com/docs/en/build-with-claude/batch-processing). Batch rate limits are separate from Messages limits and shared across models. Batch HTTP endpoints allow 1,000, 2,000 or 4,000 RPM (Start, Build, Scale). The queue holds at most 200k, 300k or 500k batch requests in processing, and a batch has at most 100k requests. **[docs]** [rate limits](https://platform.claude.com/docs/en/api/rate-limits#message-batches-api). Each request needs `max_tokens ≥ 1`. `stream: true` and `speed` (fast mode) are rejected. **[docs]** [guide](https://platform.claude.com/docs/en/build-with-claude/batch-processing).

**Processing window.** Most batches finish in under an hour. Results are available once all requests are done, or after 24 hours. A batch expires if it isn't done within 24 hours, and expiry is more likely under load. **[docs]** [guide](https://platform.claude.com/docs/en/build-with-claude/batch-processing).

**Statuses.** `processing_status` is one of `in_progress`, `canceling` or `ended`. `request_counts` has `{processing, succeeded, errored, canceled, expired}`. Timestamp fields are `created_at`, `expires_at`, `ended_at`, `cancel_initiated_at` and `archived_at`, and `results_url` is null until the batch has ended. **[docs]** [SDK `message_batch.py`](https://github.com/anthropics/anthropic-sdk-python/blob/b4b7916deaf4e5570cd627b1ae7ee4394a4de39f/src/anthropic/types/messages/message_batch.py), [`message_batch_request_counts.py`](https://github.com/anthropics/anthropic-sdk-python/blob/b4b7916deaf4e5570cd627b1ae7ee4394a4de39f/src/anthropic/types/messages/message_batch_request_counts.py).

**Cancellation.** `POST /v1/messages/batches/{id}/cancel` sets `canceling`. The batch then reaches `ended` and may contain partial results. **[docs]** [guide](https://platform.claude.com/docs/en/build-with-claude/batch-processing). A submitted batch can't be modified. **[docs]** same guide (FAQ).

**Results.** The results are JSONL at `results_url`, which should be streamed. Each line is `{custom_id, result: {type: succeeded|errored|canceled|expired, ...}}`. A `succeeded` result carries a full `message`. An `errored` result carries an error response such as `invalid_request_error` or a server error. Errored, canceled and expired requests aren't billed. Order isn't guaranteed. **[docs]** [guide](https://platform.claude.com/docs/en/build-with-claude/batch-processing), [SDK `message_batch_errored_result.py`](https://github.com/anthropics/anthropic-sdk-python/blob/b4b7916deaf4e5570cd627b1ae7ee4394a4de39f/src/anthropic/types/messages/message_batch_errored_result.py).

**Retention.** Results are available for 29 days after the batch's creation, not after it ends. `DELETE /v1/messages/batches/{id}` deletes an ended batch, and an in-progress batch must be cancelled first. Batches are scoped to a Workspace. **[docs]** [guide](https://platform.claude.com/docs/en/build-with-claude/batch-processing).

**Discount.** Batch costs 50% of standard prices on all usage, including input, output and special tokens. **[docs]** [guide](https://platform.claude.com/docs/en/build-with-claude/batch-processing).

**Features inside a batch:**
- *Prompt caching:* supported, and the discounts stack. Cache hits are best-effort, typically 30–98%. The docs recommend identical `cache_control` blocks in every request and suggest the 1-hour TTL because batches can take longer than 5 minutes. `max_tokens: 0` cache pre-warming isn't allowed. **[docs]** [guide](https://platform.claude.com/docs/en/build-with-claude/batch-processing).
- *Tools:* client tools and all server tools are supported, and the server-side loop runs more iterations before `pause_turn`. **[docs]** same guide. Client tool calls come back as `tool_use` for the caller to handle, which would need a new request. That fits the roadmap's decision to keep tool loops out of batch. **[inferred]**
- *Thinking:* extended thinking is supported. **[docs]** same guide.
- *Structured output:* `output_config.format` "works with batch processing". **[docs]** [structured outputs](https://platform.claude.com/docs/en/build-with-claude/structured-outputs#feature-compatibility).
- *Batch-only extra:* the beta header `output-300k-2026-03-24` raises `max_tokens` to 300k on recent models in batches only. **[docs]** [guide](https://platform.claude.com/docs/en/build-with-claude/batch-processing).

**Webhooks.** None documented. The guide describes polling only. **[docs, absence]** [guide](https://platform.claude.com/docs/en/build-with-claude/batch-processing).

### 3. xAI Batch API

**Existence and shape.** xAI has its own batch API. It is not OpenAI's file-first API, but it also accepts an OpenAI-style JSONL file. **[docs]** [guide](https://docs.x.ai/developers/advanced-api-usage/batch-api).
- Native flow:
  1. `POST /v1/batches {name}` creates an empty, named batch.
  2. `POST /v1/batches/{id}/requests {batch_requests: [{batch_request_id, batch_request: {...}}]}` adds requests, and can be called many times.
  3. `GET /v1/batches/{id}` returns state counters.
  4. `GET /v1/batches/{id}/results` returns results, paginated with `limit` ≤ 1000 and `pagination_token`.
  5. `GET /v1/batches/{id}/requests` returns per-request metadata.
  6. `POST /v1/batches/{id}:cancel` cancels the batch.

  **[docs]** [REST reference](https://docs.x.ai/developers/rest-api-reference/inference/batches).
- File flow: `POST /v1/files`, then `POST /v1/batches {input_file_id}`. Lines are `{custom_id, method, url, body}`, and `custom_id` maps to `batch_request_id`. An invalid line cancels the whole batch. **[docs]** [guide](https://docs.x.ai/developers/advanced-api-usage/batch-api).
- Request kinds: the guide says the `batch_request` object is keyed `responses`, `chat_get_completion`, `image_generation`, `image_edit`, `video_generation` or `video_extension`, and lists `/v1/responses` and `/v1/chat/completions` among the batchable endpoints. **[docs]** [guide](https://docs.x.ai/developers/advanced-api-usage/batch-api). The REST reference shows only `chat_get_completion`, which returns a `chat.completion`-shaped body with xAI extensions. **[docs]** [REST reference](https://docs.x.ai/developers/rest-api-reference/inference/batches). The gRPC proto has only `completion_request` (native chat), `image_request`, `video_request` and `video_extension_request`, with no Responses variant. **[docs]** [SDK proto `batch_pb2.pyi`](https://github.com/xai-org/xai-sdk-python/blob/1d9e1dffc9a0521ede0e6bc7f2b906177b940d34/src/xai_sdk/proto/v6/batch_pb2.pyi). So Responses-in-batch is documented only in the guide. **Open**: does the native endpoint accept `responses`, and in what result shape?
- Request IDs: `batch_request_id` is optional (the server generates a UUID if it's missing) and must be unique within the batch. No length or charset limit is documented. **[docs]** [REST reference](https://docs.x.ai/developers/rest-api-reference/inference/batches).

**State.** The batch object has `batch_id`, `name`, `create_time`, `expire_time`, `cancel_time`, `cancel_by_xai_message`, `state {num_requests, num_pending, num_success, num_error, num_cancelled}` and `cost_breakdown`. There is no batch-level status enum: the batch is done when `num_pending == 0`. Per-request states are `pending`, `succeeded`, `failed`, `cancelled` and `unknown`. **[docs]** [SDK proto](https://github.com/xai-org/xai-sdk-python/blob/1d9e1dffc9a0521ede0e6bc7f2b906177b940d34/src/xai_sdk/proto/v6/batch_pb2.pyi), [guide](https://docs.x.ai/developers/advanced-api-usage/batch-api). There's no expired count. **[docs]** same proto.

**Results.** Results can be fetched at any time, including before the batch finishes (partial results). Each result is `{batch_request_id, batch_result: {response | error}}`. Cost is reported in `cost_usd_ticks` (1e-10 USD). **[docs]** [guide](https://docs.x.ai/developers/advanced-api-usage/batch-api), [SDK proto](https://github.com/xai-org/xai-sdk-python/blob/1d9e1dffc9a0521ede0e6bc7f2b906177b940d34/src/xai_sdk/proto/v6/batch_pb2.pyi).

**Window and retention.** Completion within 24 hours is best effort, not guaranteed. Batches are kept until `expire_time`, after which results can't be fetched. No retention period is stated. **[docs]** [guide](https://docs.x.ai/developers/advanced-api-usage/batch-api).

**Cancellation.** Pending requests stop, completed results stay, and the batch accepts no new requests. **[docs]** [guide](https://docs.x.ai/developers/advanced-api-usage/batch-api).

**Limits.**
- Unlimited batches per team, created at most 2 per second.
- 25 MB per individual request.
- 1,000 add-requests calls per 30 seconds per team, shared across batches.
- Batches with more than 100,000 requests may be throttled.
- JSONL files are limited to 200 MB and 50,000 requests.
- Batch requests don't count against per-minute rate limits.

**[docs]** [guide](https://docs.x.ai/developers/advanced-api-usage/batch-api). (A web-search snippet quoted "100 calls every 30 seconds" from an older copy. The current guide says 1,000.)

**Tools.** Server-side tools run during processing. Client function tools return `tool_calls`, and continuing the conversation needs a new batch. **[docs]** [guide](https://docs.x.ai/developers/advanced-api-usage/batch-api). The guide doesn't mention structured output in batches. **[docs, absence]**

**Discounts and unsupported models.** "The size of the discount varies by model." Today only `grok-4.3`, `grok-4.20-0309-reasoning`, `grok-4.20-0309-non-reasoning` and `grok-4.20-multi-agent-0309` get 20% off. The discount applies to all token types and to text models only, and image and video run at standard price. **[docs]** [pricing](https://docs.x.ai/developers/pricing#batch-api-pricing), [grok-4.3 page](https://docs.x.ai/developers/models/grok-4.3).
- Not supported: `grok-4.7`, `grok-4.6`, `grok-4.5` and `grok-build-0.1` (model pages say "Batch API: Not supported"). **[docs]** [grok-4.7](https://docs.x.ai/developers/models/grok-4.7), [grok-4.6](https://docs.x.ai/developers/models/grok-4.6), [grok-4.5](https://docs.x.ai/developers/models/grok-4.5), [grok-build-0.1](https://docs.x.ai/developers/models/grok-build-0.1).
- Requests to unsupported models are rejected with "not supported for batch processing". **[docs]** [guide](https://docs.x.ai/developers/advanced-api-usage/batch-api).
- The models list says only "Not every model accepts Batch API requests. See Details on each model page." There is no machine-readable flag. **[docs]** [models](https://docs.x.ai/developers/models).

**Grok account proxy.** Nothing documented. Batch through the experimental account login is almost certainly unsupported. **[inferred]**

### 4. Gloo AI

Gloo has no batch API. The docs index (167 entries) has no batch, async-job or queue page, and the full docs dump mentions "batch" only for embeddings and content ingestion. **[docs, absence]** [llms.txt](https://docs.gloo.com/llms.txt), [llms-full.txt](https://docs.gloo.com/llms-full.txt). The only "batch" in inference is embeddings, which take up to 2,048 strings in one synchronous call. **[docs]** [Direct Embeddings](https://docs.gloo.com/api-reference/embeddings/direct.md).

Relevant to a concurrency fallback: Gloo has no universal request quota and no standard rate-limit headers.
- A spending-limit 429 (`1005 SPENDING_LIMIT_EXCEEDED`, `retryable: false`) must not be retried.
- Endpoint quotas may send `Retry-After`.
- Upstream rate limits can arrive mid-stream as SSE error `2004`.

**[docs]** [rate limits](https://docs.gloo.com/api-reference/general/limits.md).

### 5. mistral.rs (docs level)

- The documented pattern is to send many requests concurrently to one loaded `Model` (`model.send_chat_request(..)` futures joined with `join_all`). The engine batches them. **[docs]** [batching example](https://github.com/EricLBuehler/mistral.rs/blob/3f2515e9b5adc2ac44949c128c50a13b15721294/docs/src/content/docs/examples/rust/advanced/batching.md).
- `with_max_num_seqs(n)` caps the number of sequences running at once (Rust default 32), and extra requests queue. **[docs]** [throughput tuning](https://github.com/EricLBuehler/mistral.rs/blob/3f2515e9b5adc2ac44949c128c50a13b15721294/docs/src/content/docs/guides/perf/throughput-tuning.mdx). The method exists in crates.io 0.8.1 (`src/builder_macros.rs:167`, `src/gguf.rs:194`). **[docs]** crate source.
- With paged attention you get continuous batching against a shared block pool, with prefix sharing. Without it, the scheduler is first-come-first-served and batches only sequences of equal length.
  - Paged attention defaults to `auto`: on for CUDA, off for Metal and CPU.
  - So on a Mac, concurrent requests mostly queue.

  **[docs]** [throughput tuning](https://github.com/EricLBuehler/mistral.rs/blob/3f2515e9b5adc2ac44949c128c50a13b15721294/docs/src/content/docs/guides/perf/throughput-tuning.mdx), [paged attention](https://github.com/EricLBuehler/mistral.rs/blob/3f2515e9b5adc2ac44949c128c50a13b15721294/docs/src/content/docs/guides/perf/paged-attention.mdx).
- Prefix caching (`with_prefix_cache_n`, default 16) skips prefill for shared prompt prefixes, which helps batches that share a system prompt. **[docs]** same throughput page.
- `mistralrs` has no persistent job queue. A local batch can't survive a process restart. **[inferred]**
- Throughput numbers are for the separate hands-on spike.

### 6. Design synthesis

**Request IDs.** Validate every request ID against `^[A-Za-z0-9_-]{1,64}$` when it is constructed (`RequestId::new` returns `Result`). This is Anthropic's rule [docs](https://platform.claude.com/docs/en/build-with-claude/batch-processing), and OpenAI and xAI document only uniqueness, so it is the only rule that is portable everywhere. Check uniqueness in `batch()` before any network call. Pass IDs through unchanged as `custom_id` / `batch_request_id`, so no translation table is needed and the IDs show up as-is in provider consoles. If the caller gives no IDs, generate `"0"`, `"1"`, … from the input index, which keeps the common `Vec` case trivial. **[inferred]**

**`BatchHandle` (serde, versioned, no secrets)**, proposed shape:

```rust
#[derive(Serialize, Deserialize)]
#[non_exhaustive]
pub struct BatchHandle {
    pub version: u32,                 // format version, starts at 1
    pub provider: ProviderKind,       // openai | anthropic | grok | local | concurrent(…)
    pub model: String,                // full model string, e.g. "anthropic/claude-…"
    pub base_url: Option<String>,     // to detect resuming against the wrong endpoint
    pub parts: Vec<BatchPart>,        // one per provider batch; 0.2 creates exactly one
    pub request_ids: Vec<RequestId>,  // submission order; lets us report missing results
    pub kinds: RequestKinds,          // which IDs are `extract` (+ schema hash), the rest one-shot
    pub created_at: OffsetDateTime,
    pub expires_at: Option<OffsetDateTime>,         // processing deadline (24h)
    pub results_available_until: Option<OffsetDateTime>, // OpenAI 30d default/configured, Anthropic created+29d, xAI expire_time
}
#[non_exhaustive]
pub struct BatchPart {
    pub batch_id: String,             // batch_…, msgbatch_…, xAI batch_id
    pub input_file_id: Option<String>,// OpenAI (and xAI file mode): to delete after completion
    pub endpoint: Option<String>,     // OpenAI: "/v1/responses"
}
```

- Credentials are never serialized. Resuming takes the handle plus the same provider config: `nucleus::batch_resume(handle)` or `Batch::from_handle(handle, provider)`. **[inferred]**
- `output_file_id` and `error_file_id` aren't stored, because each refresh returns them. **[docs]** [reference](https://developers.openai.com/api/reference/resources/batches/methods/create)
- Worst case `request_ids` is 100k × 64 B ≈ 6.5 MB of JSON. That's acceptable, but document it. An alternative is to store only the count and rely on provider counts. **[inferred]**

**Common status and progress model:**

```rust
#[non_exhaustive] pub enum BatchStatus { Submitting, InProgress, Cancelling, Ended(EndReason) }
#[non_exhaustive] pub enum EndReason { Completed, Cancelled, Expired, Failed /* whole batch rejected */ }
pub struct BatchProgress { pub total: u64, pub succeeded: u64, pub failed: u64,
                           pub cancelled: Option<u64>, pub expired: Option<u64>, pub pending: u64 }
```

| Nucleus | OpenAI [docs](https://github.com/openai/openai-openapi/blob/e95c0fe615f45a19e8878af923a729948445a6bd/openapi.json) | Anthropic [docs](https://github.com/anthropics/anthropic-sdk-python/blob/b4b7916deaf4e5570cd627b1ae7ee4394a4de39f/src/anthropic/types/messages/message_batch.py) | xAI [docs](https://github.com/xai-org/xai-sdk-python/blob/1d9e1dffc9a0521ede0e6bc7f2b906177b940d34/src/xai_sdk/proto/v6/batch_pb2.pyi) |
|---|---|---|---|
| InProgress | `validating`, `in_progress`, `finalizing` | `in_progress` | `num_pending > 0`, no `cancel_time` |
| Cancelling | `cancelling` | `canceling` | `cancel_time` set and `num_pending > 0` **[inferred]** |
| Ended(Completed) | `completed` | `ended`, no cancel, `expired == 0` | `num_pending == 0` |
| Ended(Cancelled) | `cancelled` | `ended` with `cancel_initiated_at` | `cancel_time` set and `num_pending == 0` |
| Ended(Expired) | `expired` | `ended` with `expired > 0` **[inferred]** | `expire_time` passed **[inferred]** |
| Ended(Failed) | `failed` (+ `errors.data[]`) | n/a: validation is per request | n/a |
| progress | `total/completed/failed` only | all five counts | `num_*` (no expired) |

**Per-request results.** `results()` returns a stream of `(RequestId, BatchItem)`, where:

```rust
#[non_exhaustive] pub enum BatchItem { Succeeded(Response), Failed(nucleus::Error), Cancelled, Expired, Missing }
```

- `Response` is the same type a one-shot call returns, with `Usage`.
- `extract` results are parsed through `Response::parse::<T>()` or a `batch_extract::<T>` convenience, so the handle stays non-generic.
- `Failed` reuses `nucleus::Error` kinds:
  - Anthropic `invalid_request_error` maps to InvalidRequest.
  - A server error maps to Provider (retryable).
  - An OpenAI line with `status_code ≥ 400` maps by status.
  - An xAI `error` (a gRPC-style status) maps by code.
- `Missing` covers IDs that come back in no result at all (for example, an OpenAI batch that failed outright). **[inferred]**

**Polling vs webhooks.** Poll. The library holds no server code (design rule 7). Anthropic and xAI document no webhooks, and OpenAI's webhooks need a public endpoint configured in the dashboard. [OpenAI webhooks](https://developers.openai.com/api/docs/guides/webhooks), [Anthropic](https://platform.claude.com/docs/en/build-with-claude/batch-processing). Expose:
- `handle.refresh()`, which does one status call. Apps that receive OpenAI webhooks call this when `batch.completed` arrives.
- `handle.wait(opts)`, which polls with backoff (for example 30 s rising to 5 min; Anthropic says most batches finish within 1 hour) and honours `retry-after`.

**[inferred]**

**Partial results.** Only xAI serves results before the batch ends. Anthropic's `results_url` is null until `ended`, and OpenAI's output file is produced at the end. **[docs]** sources above. For 0.2, `results()` returns `Error::NotReady` until `Ended(_)`, and when the batch has ended it returns whatever exists: partial results for cancelled or expired batches, with the remaining IDs as `Cancelled`, `Expired` or `Missing`. Early partial results on xAI can be added later without breaking anything. **[inferred]**

**Cancellation.** `handle.cancel()` maps to all three provider cancel endpoints, and then the status moves to `Cancelling`. OpenAI can take up to 10 minutes. **[docs]** [OpenAI](https://developers.openai.com/api/docs/guides/batch), [Anthropic](https://platform.claude.com/docs/en/build-with-claude/batch-processing), [xAI](https://docs.x.ai/developers/rest-api-reference/inference/batches). For local or concurrent runs, cancel drops the queued futures. **[inferred]**

**Limits and splitting.** Use one model per batch. The `batch(model, requests)` signature already says this, and OpenAI requires it. **[docs]** [OpenAI guide](https://developers.openai.com/api/docs/guides/batch). Check limits before submitting:
- OpenAI: 50k requests and 200 MB of JSONL.
- Anthropic: 100k requests and 256 MB.
- xAI: 25 MB per request, with adds chunked under the call-rate limit.

If a check fails, return a clear `InvalidRequest`. `parts: Vec<BatchPart>` leaves room for auto-splitting later. **[inferred]**

**Cleanup.** After OpenAI results are fetched, optionally delete the input file and output files, or set `expires_after` when uploading, because files count against the 2.5 TB project quota. Anthropic offers `DELETE /v1/messages/batches/{id}`. Expose this as `handle.delete()` rather than doing it implicitly. **[inferred]** from [Files](https://developers.openai.com/api/reference/resources/files/methods/create) and [Anthropic](https://platform.claude.com/docs/en/build-with-claude/batch-processing).

**Providers without batch support (Gloo, unsupported Grok models, custom providers).**
- Default: return `Error::Unsupported { feature: Batch, provider, model }` before any network call. Base it on `capabilities()` where the answer is known statically, such as a built-in table of the xAI models that support batch, and fall back to mapping the provider's "not supported for batch processing" rejection. **[inferred]**
- Opt in with `BatchOptions::concurrent_fallback(max_in_flight)`. This runs the requests as ordinary calls with bounded concurrency and the reliability layer's retry logic, and returns the same `results()` stream. The resulting handle is marked non-resumable, and there's no discount.
  - On Gloo, a `SPENDING_LIMIT_EXCEEDED` result must stop the whole run rather than be retried per request. **[docs]** [Gloo limits](https://docs.gloo.com/api-reference/general/limits.md)

**Local.** Local batches always run concurrently on the shared loaded model, bounded by `max_num_seqs`. **[docs]** [mistral.rs](https://github.com/EricLBuehler/mistral.rs/blob/3f2515e9b5adc2ac44949c128c50a13b15721294/docs/src/content/docs/guides/perf/throughput-tuning.mdx). They use the same handle type with `provider: Local`. They can't be resumed after a restart: resuming an unfinished local handle returns `Error::BatchNotResumable`. **[inferred]**

**What doesn't map cleanly:**
1. **Submission model.** OpenAI uploads a file and then creates the batch. Anthropic takes one JSON POST. xAI creates an empty batch and then appends requests over several calls. An xAI submission can be interrupted half-way, so the handle must be persisted only after all adds succeed, or carry a `Submitting` state with the count added so far. **[docs]** sources above. **[inferred]** consequence.
2. **Validation timing.** OpenAI can fail the whole batch (`failed` with line errors). Anthropic accepts everything and reports invalid params per request after the batch ends. An invalid xAI JSONL line cancels the whole batch. Nucleus should validate request shapes locally where it can. **[docs]**
3. **Progress fields.** OpenAI has no cancelled or expired counts, and xAI has no expired count. Hence the `Option` fields above. **[docs]**
4. **Expiry visibility.** OpenAI and Anthropic report expired requests explicitly. xAI has only `expire_time`, and its per-request states have no `expired`. **[docs]**
5. **Protocol on xAI.** Nucleus speaks Responses to xAI, but xAI's batch reference and SDK show only the chat-completions shape. If `responses` doesn't work in batches, `grok/` batch needs a small chat-completions request and response mapper just for batch. **[docs]** + **open**.
6. **Discount and capability per model.** The discount is 50/50 for OpenAI and Anthropic but 20% or unsupported for xAI, depending on the model. Cost calculation in `Usage` must know when a request ran in a batch. **[docs]** [xAI pricing](https://docs.x.ai/developers/pricing#batch-api-pricing)
7. **Retention anchor.** OpenAI's is the output file's creation plus 30 days (configurable). Anthropic's is the batch's creation plus 29 days. xAI's is `expire_time`, which is unspecified. **[docs]**
8. **Prompt caching.** Anthropic caching in a batch is best-effort and the docs suggest a 1-hour TTL. Nucleus's "automatic prompt caching" may want a batch-specific TTL option, since 1-hour cache writes cost more. **[docs]** [Anthropic](https://platform.claude.com/docs/en/build-with-claude/batch-processing)
9. **Anthropic batch-only options** such as the 300k output beta. Reach these through `provider_options`, not first-class API. **[docs]**

## Impact on the design

Proposed ROADMAP.md changes, to be made in the PR that merges this note:

- **Architecture, "Batch" line:** add "Gloo has no batch API; xAI batch only for some models (20% discount)".
- **Phase 3, "Batch requests":**
  - Change "backed by … xAI's Batch API" to "xAI's Batch API for the models that support it (today grok-4.3 and grok-4.20-*; not grok-4.5/4.6/4.7 or grok-build), with a clear `Unsupported` error otherwise".
  - Add: request IDs must match `^[A-Za-z0-9_-]{1,64}$` and be unique, and are auto-numbered if omitted.
  - Add: one model per batch, and provider limits (OpenAI 50k/200 MB, Anthropic 100k/256 MB, xAI 25 MB per request) are checked before submitting.
  - Add: polling only (`refresh`, `wait`), with no webhook receiver in the library.
  - Add: results are available after the batch ends, as `Succeeded | Failed | Cancelled | Expired | Missing` per ID.
  - Add: local and concurrent-fallback handles aren't resumable after a restart.
  - Change "providers without batch support (possibly Gloo)" to "(Gloo: confirmed no batch API)".
  - Add an item: settle the xAI batch request shape (Responses vs chat completions). A chat-completions mapper for batch may be needed.
- **Phase 3 "Done when":** add a test that resumes a serialized `BatchHandle` against the mocked OpenAI and Anthropic batch APIs.
- **Phase 2 design gate (`core-api.md`):**
  - Include `BatchHandle`, `BatchStatus`, `BatchProgress`, `BatchItem` and `RequestId` as sketched above.
  - Batch operations become optional `Provider` methods with an `Unsupported` default (or a separate `BatchProvider` trait), so custom providers keep compiling.
  - `capabilities()` reports batch support per model.
- **Phase 2 open question on `Usage` cost:** cost must account for batch discounts (OpenAI 50%, Anthropic 50%, xAI 20% on some models).
- **Decisions, "Batch requests are in 0.2, for cost savings":** keep it. The savings are real for OpenAI and Anthropic, smaller for xAI, and absent for Gloo.

## Open questions

Only a live call can answer the first four:

1. Does xAI's native batch API accept `batch_request.responses` (Responses API bodies), and what result shape comes back? The REST reference and the gRPC proto show only chat completions.
2. Does OpenAI's batch accept a Responses body with `text.format` (structured output), `reasoning`, `store: false` and `include: ["reasoning.encrypted_content"]` exactly as the synchronous endpoint does? `background: true` and `stream: true` should be rejected or ignored, but which one happens?
3. Where do OpenAI requests rejected with a 4xx go (the output file with `status_code` ≥ 400, or the error file), and what code do cancelled requests carry?
4. What does xAI do when a batch passes `expire_time` with requests still pending (are they reported as failed or cancelled), and what is the default `expire_time`?

Answerable from docs or design:

5. What happens to an OpenAI batch that exceeds the per-model enqueued-token cap (rejected at create, or `failed` after validation)? This affects whether we can pre-check.
6. Does structured output work in xAI batches? It's undocumented, though server and client tools are documented.
7. What are the default and range of OpenAI `purpose=batch` input file expiry (`expires_after`)? Should Nucleus set it on upload?
8. Does batch work through the Grok account proxy (experimental)? Assume no, and return `Unsupported`.
9. The xAI add-requests rate limit has been quoted as 100 and 1,000 calls per 30 s in different copies of the docs. Confirm before choosing chunk sizes.
10. For the local throughput figures (concurrent requests on Metal without paged attention), see the hands-on mistral.rs spike.
