# Design: Phase 2 core API

Date: 2026-10-19. Status: proposal for review, not final. Settles the open questions in [ROADMAP.md](../../ROADMAP.md) Phase 2 (lines 179-210, questions at 181-188) and records the full public signatures the implementation is written against. Points where a choice was made between named options are repeated under "Open questions" with a recommendation.

Sources read for this note:

- `ROADMAP.md` — the Target API snippet (14-35), Design rules (55-68), Decisions (119-145), the Phase 2 section (179-210), Known issues (289-310).
- The six Phase 0.5 spike notes in `docs/design/spikes/`: [mistralrs.md](spikes/mistralrs.md), [gloo.md](spikes/gloo.md), [responses-api.md](spikes/responses-api.md), [anthropic-messages.md](spikes/anthropic-messages.md), [rmcp.md](spikes/rmcp.md), [batch.md](spikes/batch.md).
- The existing code this replaces: `src/chat/manager.rs` (`ChatManager`, the agent-loop reference), `src/chat/commands.rs` (slash commands), `src/provider/{mod,types,grok,mistralrs}.rs`, `src/tool/{plugin,registry}.rs`, `src/tools/commands.rs` (`ExecPlugin`), `src/patterns.rs`, `src/rag/`.

Where a spike settles a question, this note follows it and cites the note; it does not re-argue. Where the spike left a question open and Phase 2 must still pick a default, the choice is marked **(Open N)** and collected at the end.

---

## 1. Scope

Phase 2 is the core the rest of 0.2 is built on: the data model, the agent loop, the traits, the error type and the streaming surface. It does not implement the providers (Phase 3), RAG (Phase 4), MCP (Phase 5) or the examples (Phase 6); it only puts their seams in place, each left as `todo!()` in the implementation PR and mocked in tests (Phase 2 "Done when", ROADMAP.md:210).

Everything here is checked against the Design rules (ROADMAP.md:55-68):

1. Every public data type is `Serialize + Deserialize` (rule 1).
2. Agents can be defined as data via `AgentSpec` (rule 2).
3. Owned data and `Arc<dyn …>` trait objects; streams yield owned `Event`s, no borrowed callbacks (rule 3).
4. Public enums and option structs are `#[non_exhaustive]` (rule 4).
5. Extension points are traits: `Provider`, `Tool`, `Embedder`, `VectorStore`, `ConversationStore`, hooks (rule 5).
6. Protocol details stay private: no `rmcp` or mistral.rs types in the public API (rule 6; rmcp.md §2 "wrap completely").
7. No server code in the library (rule 7).
8. Escape hatches: `provider_options(json!(..))` and `provider_headers(..)` (rule 8; widened to headers per gloo.md §Impact 5).
9. Feature flags only add things (rule 9).
10. History is append-only (rule 10; anthropic-messages.md §Impact 2).

Code lints carried from the quality bar (ROADMAP.md:81): `#![forbid(unsafe_code)]`, `#![deny(missing_docs)]`, no `unwrap`/`expect`/`panic!` in library code, rustdoc warnings are errors. Every fallible public method returns `Result<_, nucleus::Error>`; no `anyhow` in the public API.

## 2. Crate and module layout

One `nucleus` crate with the feature flags already in `Cargo.toml:46-63` (`hosted` default; `local`, `rag`, `lancedb`, `mcp`, `experimental`, `metal`, `cuda` add-only). `nucleus-cli` stays a separate workspace member (ROADMAP.md:122).

```
src/
  lib.rs            // re-exports, prelude, nucleus::complete
  error.rs          // nucleus::Error, ErrorKind data, RequestId
  message.rs        // Message, Part, ProviderItem, Role, ToolCallId
  conversation.rs   // Conversation, ConversationId
  event.rs          // Event, EventStream, StopReason, Usage, Cost
  options.rs        // RunOptions, Effort, Thinking, ThinkingDisplay, CacheTtl
  agent.rs          // Agent, AgentBuilder, AgentSpec, hooks, Turn
  extract.rs        // extract::<T>, schema transform + local validation
  store.rs          // ConversationStore trait
  provider/         // Provider trait, Capabilities, ChatRequest, MockProvider
    mod.rs
    mock.rs
    responses/      // Phase 3, private
    messages/       // Phase 3, private
    local/          // Phase 3, private
  tool/             // Tool trait, tool_fn, ToolContext, ToolOutput, Approval
  tools/            // built-in files() and exec(), on the new Tool trait
  batch.rs          // BatchHandle, BatchStatus, BatchProgress, BatchItem, RequestId
```

`ChatManager` and `ChatManagerBuilder` are removed; `Agent` + `Conversation` replace them (ROADMAP.md:126). What is worth keeping from `src/chat/manager.rs` is the per-turn rollback on failure (`run_turn` → `run_turn_inner`, manager.rs:493-513) and the "run one turn against a caller-supplied history" shape (`query_with`, manager.rs:448-464); both appear here as `Turn` and the append-only commit rule. What is not kept: no step limit, tool errors ending the turn, and the system prompt stored as a history message (ROADMAP.md:309).

## 3. Model strings

**Recommendation (Open 1):** split on the **first** `/` only.

```
model-string := provider "/" model-id
provider     := "openai" | "anthropic" | "grok" | "gloo" | "local"
model-id     := everything after the first "/" , verbatim
```

The provider prefix is the first path segment; the remainder is opaque and passed through unchanged. This handles every case the roadmap names (ROADMAP.md:181):

- Hosted, no slashes in the id: `openai/gpt-5.1`, `anthropic/claude-opus-5-5`, `grok/grok-4.3`, `gloo/gloo-anthropic-claude-sonnet-4.6`. Gloo ids contain dots and hyphens but **no slashes** (gloo.md §4), so they need nothing special.
- HF repo: `local/Qwen/Qwen3-8B` — `model-id` = `Qwen/Qwen3-8B`.
- HF GGUF: `local/Qwen/Qwen3-8B-GGUF:Qwen3-8B.Q4_K_M.gguf` — `model-id` = `Qwen/…:Qwen3-8B.Q4_K_M.gguf`. The `repo:file.gguf` form is exactly what `GgufModelBuilder::new(repo, vec![file])` takes (mistralrs.md finding 9).
- Filesystem path: the tail's **first character** decides. `/` or `.` begins a path, anything else is a HF repo id.
  - absolute: `local//Users/me/models/Qwen3-8B.gguf` (the first slash separates the provider, the second begins the absolute path);
  - relative: `local/./models/Qwen3-8B.gguf` or `local/../shared/model.gguf`.

All three local forms load in the spike (mistralrs.md finding 9), so the grammar only has to pick them apart, not invent anything. `ModelString` is a public parsed type so apps and `AgentSpec` can store structured data:

```rust
/// A parsed model string. Serde-serializable so it survives app config.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct ModelString {
    /// Which provider serves this model.
    pub provider: ProviderKind,
    /// The provider's own model id, path or `repo:file` spec, verbatim.
    pub id: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum ProviderKind {
    OpenAI,
    Anthropic,
    Grok,
    Gloo,
    Local,
    /// A provider the caller installed with `AgentBuilder::provider`.
    Custom,
}

impl ModelString {
    /// Parse `provider/id`. Splits on the first `/`; the tail is opaque.
    pub fn parse(s: &str) -> Result<Self, Error>;
    pub fn as_str(&self) -> String;
}

impl ProviderKind {
    /// The prefix that selects this provider in a model string.
    pub fn prefix(self) -> &'static str;
}
```

Rationale: the "split on the first slash, keep the tail opaque" rule is the only one that needs no per-provider knowledge and no escaping, and it matches what `grok.rs` and `mistralrs.rs` already assume about their ids. The only cost is the double slash for an absolute local path. **(Open 1)** weighs a `file://` alternative; the recommendation is to ship the double slash and add sugar later if it grates.

## 4. Messages and conversations

### 4.1 `Message` and `Part`

Assistant messages are an **ordered list of parts** (ROADMAP.md:190; anthropic-messages.md §Impact 1), because Anthropic interleaves thinking → tool_use → thinking → tool_use and the order must replay exactly. Tool-call input is stored as **raw JSON** so the bytes replay identically (ROADMAP.md:191; anthropic-messages.md Q3), which is why `input` is `Box<serde_json::value::RawValue>` and not `Value`.

```rust
/// One message in a conversation.
///
/// `Role` is only `User` or `Assistant`: the system prompt belongs to the
/// `Agent` and is never stored here (ROADMAP.md:195). Tool results travel in a
/// `User` message, as they do on the wire.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct Message {
    pub role: Role,
    /// Ordered parts. Order is significant and is replayed as-is.
    pub parts: Vec<Part>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum Role {
    User,
    Assistant,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub enum Part {
    /// Visible text.
    Text { text: String },

    /// Reasoning / thinking, for display and (where the provider replays it)
    /// for storage inside a `Provider` part alongside it.
    Reasoning { text: String, summary: bool },

    /// A tool call the model asked for. `input` is raw JSON so it replays
    /// byte for byte (ROADMAP.md:191).
    ToolCall {
        id: ToolCallId,
        name: String,
        input: Box<serde_json::value::RawValue>,
    },

    /// The result sent back for a tool call.
    ToolResult {
        call_id: ToolCallId,
        output: ToolOutput,
        is_error: bool,
    },

    /// Retrieved knowledge attached to a user turn, kept separate from what
    /// the user typed so apps can render or hide citations (ROADMAP.md:194).
    Knowledge {
        text: String,
        sources: Vec<KnowledgeSource>,
    },

    /// Opaque provider-specific output, replayed unchanged to the same
    /// dialect (ROADMAP.md:192). See `ProviderItem`.
    Provider(ProviderItem),
}

impl Message {
    pub fn user(text: impl Into<String>) -> Self;
    pub fn assistant_text(text: impl Into<String>) -> Self;
    pub fn parts(role: Role, parts: Vec<Part>) -> Self;
    /// Concatenated `Text` parts, the common case.
    pub fn text(&self) -> String;
    pub fn tool_calls(&self) -> impl Iterator<Item = (&ToolCallId, &str, &str)>;
    /// The `Provider` parts in order.
    pub fn provider_items(&self) -> impl Iterator<Item = &ProviderItem>;
}
```

```rust
/// A provider's own tool-call id, passed through unchanged. The Responses
/// protocol calls this the `call_id` (responses-api.md Q2).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[non_exhaustive]
pub struct ToolCallId(pub String);

/// A tool's result. `content` is what the model reads; `structured` is the
/// JSON form when the tool produced one (rmcp.md §6 structured_content).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct ToolOutput {
    pub content: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub structured: Option<serde_json::Value>,
}
```

```rust
/// A source citation for a `Part::Knowledge`. Line ranges are 1-based and
/// inclusive; a whole-file citation has `start_line == 1` and `end_line == 0`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct KnowledgeSource {
    pub file: String,
    pub start_line: u32,
    pub end_line: u32,
}
```

### 4.2 `ProviderItem` — opaque, tagged, prefix-bound

This is the type that makes replay safe. It follows the corrected rule from anthropic-messages.md Q3 and the Responses rule from responses-api.md Q3: tag the item with the **endpoint dialect**, the **model**, and a **fingerprint of the request prefix** (rendered system + tools) that produced it. The Anthropic protocol keeps a thinking block only while the fingerprint matches; on a mismatch the block and every later thinking block are dropped (the same thing the beta `drop_block` option does), except inside the in-flight tool loop. The Responses protocol drops reasoning items when the model family changes, and any reasoning item without `encrypted_content` always.

```rust
/// A provider-specific output item stored verbatim and replayed unchanged.
///
/// Never contains a tool call or tool result: those are ordinary `Part`s and
/// are never dropped (ROADMAP.md:193). Examples: an Anthropic `thinking` or
/// `redacted_thinking` block, an OpenAI/xAI `reasoning` item with
/// `encrypted_content`, the assistant `phase`, a `compaction` item.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct ProviderItem {
    /// Endpoint dialect, e.g. "anthropic", "openai", "xai", "grok-account",
    /// "gloo". The proxy is its own dialect so its ciphertext cannot leak to
    /// api.x.ai (responses-api.md Q8).
    pub dialect: String,
    /// The model that produced the item.
    pub model: String,
    /// A short kind tag for diagnostics ("thinking", "reasoning", "phase", …).
    pub kind: String,
    /// Hash of the rendered system prompt + tool set at the time of capture.
    /// Empty for dialects that do not bind to a prefix.
    pub prefix_fingerprint: String,
    /// The item exactly as received (a `response.output_item.done` item, or an
    /// Anthropic content block).
    pub raw: Box<serde_json::value::RawValue>,
}
```

The replay decision lives in the private protocol layer, not in the public type: the public layer only guarantees the data is kept and ordered. A mock test asserts the stripped request body after `with_system` (anthropic-messages.md §Impact 3).

### 4.3 `Conversation`

```rust
/// Plain, serializable conversation state with a stable id. No provider
/// handles, no agent state.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct Conversation {
    /// Stable for the life of the conversation. Used as the prompt-cache key
    /// and, on Grok, for the per-conversation routing headers
    /// (responses-api.md Q9.11; ROADMAP.md:196).
    pub id: ConversationId,
    /// Append-only message history. See "History is append-only" below.
    pub messages: Vec<Message>,
    /// Creation time, RFC 3339, for stores and diagnostics.
    pub created_at: String,
}

/// A conversation's stable identifier. Printable, serde, and cheap to clone.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[non_exhaustive]
pub struct ConversationId(/* opaque, e.g. a UUIDv7 string */ String);

impl ConversationId {
    pub fn new() -> Self;
    pub fn parse(s: &str) -> Result<Self, Error>;
    pub fn as_str(&self) -> &str;
}

impl Conversation {
    pub fn new() -> Self;
    pub fn with_id(id: ConversationId) -> Self;
    /// Append a message. The only mutator used by the agent loop.
    pub fn push(&mut self, message: Message);
    pub fn messages(&self) -> &[Message];
    /// Drop all turns, keep the id. This is what an app's `/reset` calls.
    pub fn clear(&mut self);
    /// Load from / save to a store, when the agent has one configured.
    pub async fn load(store: &dyn ConversationStore, id: &ConversationId) -> Result<Option<Self>, Error>;
    pub async fn save(&self, store: &dyn ConversationStore) -> Result<(), Error>;
}
```

The system prompt is **never** a message here; it is rendered into each request by the agent (ROADMAP.md:195). A conversation always ends with a user message when sent, because Anthropic rejects assistant prefill on 4.6+ (anthropic-messages.md Q1/Q7).

## 5. Events, streaming and usage

### 5.1 `Event` and `EventStream`

Streaming is `Stream<Item = Event>`, owned, no borrowed callbacks (rule 3; ROADMAP.md:206). Events must **not** assume argument deltas: hosted responses stream tool-call arguments as `input_json_delta` / `function_call_arguments.delta`, while mistral.rs delivers the whole call in one chunk (mistralrs.md finding 3). So a complete `ToolCall` event is always emitted, and deltas are optional.

```rust
/// A live-streamed item. Errors travel as `Event::Error`, so the stream item
/// is plain `Event` rather than `Result`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub enum Event {
    /// Visible text. For live display only; the authoritative text is rebuilt
    /// from parts when the turn ends.
    TextDelta { text: String },
    /// Reasoning/thinking text or summary.
    ReasoningDelta { text: String, summary: bool },
    /// A tool-call argument fragment. Hosted providers only; may never appear.
    ToolCallDelta { id: ToolCallId, arguments_delta: String },
    /// A complete tool call, once its arguments are whole. Always emitted,
    /// whichever way the provider delivered it.
    ToolCall { id: ToolCallId, name: String, input: Box<serde_json::value::RawValue> },
    /// A provider-specific event worth surfacing; unknown events pass through
    /// here and are logged with `tracing` (responses-api.md Q1).
    Provider { dialect: String, kind: String, raw: serde_json::Value },
    /// Token usage. Present once, near the end, when the provider reports it.
    Usage(Usage),
    /// The model stopped, with the reason. Terminal for the turn.
    Stop(StopReason),
    /// The provider errored. Terminal for the stream.
    Error(Error),
}

/// The owned event stream. No lifetime parameter.
pub type EventStream = std::pin::Pin<Box<dyn futures::Stream<Item = Event> + Send>>;
```

The library **never prints**: no `println!`, no `eprint!` (this is what `grok.rs` got wrong, ROADMAP.md:295). Text reaches the user through `TextDelta` or the returned `Turn`; diagnostics go to `tracing` only (ROADMAP.md:83).

`StopReason` follows anthropic-messages.md Q5 and responses-api.md Q5 (Responses has no finish reason, so it is derived):

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub enum StopReason {
    EndTurn,
    ToolUse,
    /// The output hit the token limit. Never execute a trailing partial tool
    /// call (anthropic-messages.md Q5).
    MaxTokens,
    StopSequence { sequence: String },
    /// The output filled the context window; a hint to compact.
    ContextWindowExceeded,
    /// The model declined. The partial turn is not committed.
    Refusal { category: Option<String>, explanation: Option<String> },
    /// A server-side tool loop hit its iteration limit; resending continues.
    PauseTurn,
    /// Unrecognised reason, kept verbatim.
    Other(String),
}
```

### 5.2 `Usage` and cost

(Answers ROADMAP.md:182.) `Usage` is attached to every result (ROADMAP.md:207). Field semantics follow responses-api.md Q5: `output_tokens` **includes** reasoning tokens and `input_tokens` **includes** cache-read tokens on the Responses protocol; cached counts are clamped so a provider that reports `cached > input` never yields a negative (responses-api.md §Impact 4).

```rust
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[non_exhaustive]
pub struct Usage {
    pub input_tokens: u64,
    pub output_tokens: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_read_tokens: Option<u64>,
    /// 5-minute-TTL cache writes.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_write_tokens: Option<u64>,
    /// 1-hour-TTL cache writes, kept separate because they cost more
    /// (anthropic-messages.md Q4).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_write_1h_tokens: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_tokens: Option<u64>,
    /// The total cost, when it could be determined. See "Cost calculation".
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cost: Option<Cost>,
    /// Whether this usage came from a batch request (affects the discount).
    #[serde(default)]
    pub batch: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct Cost {
    pub usd: f64,
    pub source: CostSource,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum CostSource {
    /// The provider reported the cost (xAI `cost_in_usd_ticks`).
    Provider,
    /// Computed from the built-in or user-supplied price table.
    PriceTable,
}
```

**Cost calculation (recommendation for ROADMAP.md:182):**

1. Prefer the **provider-reported cost** when present. xAI reports `cost_in_usd_ticks` (1 USD = 1e10 ticks; responses-api.md Q5). Convert and set `CostSource::Provider`.
2. Otherwise compute from a **built-in, overridable price table**. The table is per `provider/model` and gives an input rate, an output rate, a cache-read multiplier and cache-write multipliers. Gloo's public catalog (`GET /platform/v2/models`, no auth) supplies per-model prices including cache read/write and can seed the table, but any fetch stays optional so tests stay offline (gloo.md §Impact 9).
3. Apply the **cache multipliers**: read 0.1× base input, but 0.05× on Opus 5.5 / Sonnet 5.5 and 0.025× on Fable 5.1; write 1.25× (5m) or 2× (1h) (anthropic-messages.md Q4).
4. Apply the **batch discount** when `Usage.batch` is set: 50% OpenAI, 50% Anthropic, 20% on the xAI models that support batch (batch.md §6.6; ROADMAP.md:131).
5. On Gloo **guarded** endpoints, add the **6.5% platform fee** on top (gloo.md §7).
6. If neither the provider nor the table knows the price, leave `cost = None`. Never guess.

The user overrides the table with `PriceTable`:

```rust
/// Per-model prices in USD per 1M tokens. Apps override the built-in table.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[non_exhaustive]
pub struct PriceTable {
    entries: std::collections::BTreeMap<String, ModelPrice>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[non_exhaustive]
pub struct ModelPrice {
    pub input_per_million: f64,
    pub output_per_million: f64,
    /// Multiplier applied to `input_per_million` for a cache read.
    pub cache_read_multiplier: f64,
    /// Multiplier for a 5-minute cache write.
    pub cache_write_multiplier: f64,
    /// Multiplier for a 1-hour cache write.
    pub cache_write_1h_multiplier: f64,
}

impl PriceTable {
    pub fn builtin() -> Self;
    pub fn with(self, model: &str, price: ModelPrice) -> Self;
    pub fn get(&self, model: &str) -> Option<ModelPrice>;
}
```

The agent holds the table; `AgentBuilder::price_table(..)` replaces it. Cost is computed once, after the terminal event, so a mid-stream error leaves earlier partial usage uncosted (correct: nothing was billed for a stream that never finished).

## 6. The error type

(Answers ROADMAP.md:188.) **One** `nucleus::Error`, `#[non_exhaustive]`, serde-serializable, with kinds that cover anthropic-messages.md Q7, gloo.md §Impact 6 and responses-api.md §Impact 3.

```rust
/// The single public error type. `#[non_exhaustive]`.
#[derive(Debug, Clone, Serialize, Deserialize, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    /// The model declined (HTTP 200 refusal, or a guarded content block).
    /// The partial turn is not committed. `complete`/`extract` return this so
    /// `?` works (anthropic-messages.md Q5).
    #[error("refused")]
    Refused(Refusal),

    /// Content policy, kept separate from auth because a guarded Gloo 403
    /// `content_policy_violation` and an Anthropic refusal are not auth errors
    /// (gloo.md §Impact 6).
    #[error("content policy")]
    ContentPolicy(Refusal),

    /// Retryable rate limit. `retry_after` from `retry-after-ms`,
    /// `retry-after` (seconds or HTTP date) or a parsed "try again in Ns".
    #[error("rate limited")]
    RateLimited(Retryable),

    /// The provider is overloaded (Anthropic 529, OpenAI
    /// `server_is_overloaded`). Retryable.
    #[error("provider overloaded")]
    Overloaded(Retryable),

    /// Timed out (HTTP 504, or the request/idle timeout). Retryable.
    #[error("timed out")]
    Timeout(Retryable),

    /// Output was cut off (`max_tokens` or `model_context_window_exceeded`).
    /// `partial` carries what was produced.
    #[error("truncated")]
    Truncated(Truncation),

    /// Authentication or permission failure (401, 403, Gloo gateway 401/403).
    /// Not retryable, not a content policy.
    #[error("authentication failed")]
    Auth(ProviderError),

    /// A non-retryable billing or spend-cap error: Anthropic 402
    /// `billing_error` and a 429 with `enforced_spend_limit_reached`, OpenAI
    /// `insufficient_quota` / `*_spend_limit_exceeded`, Gloo 402
    /// `INSUFFICIENT_CREDIT` / 429 `SPENDING_LIMIT_EXCEEDED`. A 429 must not
    /// be hard-coded retryable (gloo.md §Impact 6). Not retryable.
    #[error("billing error")]
    Billing(ProviderError),

    /// A malformed request the provider rejected (400, 404, 413, 422). Not
    /// retryable.
    #[error("invalid request")]
    InvalidRequest(ProviderError),

    /// Any other provider-side failure, often retryable (500, 409, a
    /// `response.failed`). `provider.retryable` says which.
    #[error("provider error")]
    Provider(ProviderError),

    /// A tool failed. Goes back to the model as an error result, not raised
    /// out of the loop, unless it reaches the top level.
    #[error("tool '{name}' failed")]
    Tool { name: String, #[source] source: Box<Error> },

    /// The provider or model cannot do what was asked (batch on Gloo, a
    /// model that cannot turn thinking off, an unsupported capability).
    #[error("unsupported: {feature}")]
    Unsupported { feature: String, provider: String, model: Option<String> },

    /// Batch results are not available yet (batch.md §6 "partial results").
    #[error("batch results not ready")]
    NotReady,
    /// A local or concurrent batch cannot be resumed (batch.md §6).
    #[error("batch not resumable")]
    BatchNotResumable,

    /// A conversation-store failure.
    #[error("conversation store error")]
    Store(String),
    /// Serialization or schema-conversion failure (a `Box<RawValue>` that
    /// would not parse, an `extract` schema that cannot be transformed).
    #[error("serialization error")]
    Serialization(String),
    /// I/O against a file, socket or child process. The message carries the
    /// context; no `std::io::Error` is exposed so the type stays serde.
    #[error("io error")]
    Io(String),
}

/// A provider-originated error. **Every** provider error carries the request
/// id, so a failed call can be quoted to the provider's support.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct ProviderError {
    /// The provider's request id (`req_…` on Anthropic, `x-request-id`
    /// elsewhere). `None` only when the provider sent none, e.g. some
    /// in-stream errors.
    pub request_id: Option<RequestId>,
    /// HTTP status, when the error came from a response.
    pub status: Option<u16>,
    /// The provider's error code, e.g. `insufficient_quota`,
    /// `enforced_spend_limit_reached`, `1005`.
    pub code: Option<String>,
    pub message: String,
    /// Whether the reliability layer may retry. Set from `retryable`,
    /// `x-should-retry`, the code and the status (responses-api.md §Impact 3).
    pub retryable: bool,
}

/// A provider request id. A newtype so it is never confused with other ids.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[non_exhaustive]
pub struct RequestId(pub String);

#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct Retryable {
    pub error: ProviderError,
    /// Seconds to wait before retrying, when the provider said.
    pub retry_after: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct Refusal {
    pub category: Option<String>,
    pub explanation: Option<String>,
    pub error: ProviderError,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct Truncation {
    pub reason: StopReason,
    /// Text produced before truncation, when the caller asked for it.
    pub partial: String,
    pub error: ProviderError,
}

pub type Result<T, E = Error> = std::result::Result<T, E>;
```

Every variant that comes from the wire embeds a `ProviderError` with a `request_id`. `Error::source()` is implemented so the chain survives (via `thiserror`), and the whole enum is `Serialize`, so it can cross an HTTP boundary when nucleus becomes a service.

Retry classification (Phase 3 implements it, Phase 2 fixes the shape): honour `x-should-retry`, `retry-after-ms`, `retry-after` (seconds or HTTP date), and retry only before the first output event (anthropic-messages.md Q7; responses-api.md §Impact 3). A mid-stream SSE `error` is retried only if nothing was delivered.

## 7. Traits

### 7.1 `Tool` and `tool_fn`

`Tool` is the extension point (rule 5). It replaces `Plugin` (`src/tool/plugin.rs`) and keeps the same idea — name, description, JSON schema, async execute — with the fixes the Known issues name: the schema is the **whole** JSON Schema (not just `properties`, mistralrs.md finding 2), and the interface is the model-facing tool, not the LLM tool spec (ROADMAP.md:170).

```rust
#[async_trait::async_trait]
pub trait Tool: Send + Sync {
    /// The model-facing name. `[A-Za-z0-9_-]`, at most 64 characters
    /// (rmcp.md §6).
    fn name(&self) -> &str;
    fn description(&self) -> &str;
    /// The full JSON Schema for `call`'s arguments.
    fn parameters(&self) -> serde_json::Value;
    /// Untrusted hints (read-only, destructive). Reaches the approval hook as
    /// context and is never used to skip approval (rmcp.md §6).
    fn annotations(&self) -> ToolAnnotations { ToolAnnotations::default() }
    /// Run the tool. `arguments` is the raw JSON the model produced;
    /// `context` carries cancellation and progress.
    async fn call(&self, arguments: Box<serde_json::value::RawValue>, context: ToolContext)
        -> Result<ToolOutput, Error>;
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[non_exhaustive]
pub struct ToolAnnotations {
    pub title: Option<String>,
    pub read_only_hint: bool,
    pub destructive_hint: bool,
    pub idempotent_hint: bool,
    pub open_world_hint: bool,
}

/// Run-scoped context passed to `Tool::call`. Owned; no borrows.
pub struct ToolContext {
    call_id: ToolCallId,
    conversation_id: ConversationId,
    /* cancellation token, progress sink, tracing span — private */
}
impl ToolContext {
    pub fn call_id(&self) -> &ToolCallId;
    pub fn conversation_id(&self) -> &ConversationId;
    /// Whether the caller has cancelled the run.
    pub fn is_cancelled(&self) -> bool;
}
```

`tool_fn` adapts an ordinary function with typed arguments. The argument type derives `schemars::JsonSchema`, whose schema becomes the tool's parameters, and the raw JSON is deserialized into it before the call — so a mistyped `serde_json::json!` cannot reach a typed handler.

```rust
/// Wrap a plain function as a `Tool`. The schema comes from `schemars`.
///
/// ```no_run
/// # use std::sync::Arc;
/// # use nucleus::{tool_fn, ToolOutput, Error};
/// #[derive(serde::Deserialize, schemars::JsonSchema)]
/// struct Weather { city: String }
///
/// let weather = tool_fn("get_weather", "Look up the weather", |a: Weather| async move {
///     Ok(ToolOutput { content: format!("Sunny in {}", a.city), structured: None })
/// });
/// ```
pub fn tool_fn<A, F, Fut>(name: &str, description: &str, f: F) -> Arc<dyn Tool>
where
    A: serde::de::DeserializeOwned + schemars::JsonSchema + Send + 'static,
    F: Fn(A, ToolContext) -> Fut + Send + Sync + 'static,
    Fut: std::future::Future<Output = Result<ToolOutput, Error>> + Send;

/// Sync convenience wrapper, for tools that do no async work.
pub fn tool_fn_sync<A, F>(name: &str, description: &str, f: F) -> Arc<dyn Tool>
where
    A: serde::de::DeserializeOwned + schemars::JsonSchema + Send + 'static,
    F: Fn(A) -> Result<ToolOutput, Error> + Send + Sync + 'static;
```

The built-in `files()` and `exec()` tools are rebuilt on this trait (ROADMAP.md:200). `exec()` fixes the `ExecPlugin` bug (`Command::new(command)` with dropped args, ROADMAP.md:302) by running through a shell when the command is a single string; both go through the approval hook.

### 7.2 `Provider`

The `Provider` trait (rule 5) replaces `src/provider/types.rs::Provider`. It returns a stream rather than taking a callback (rule 3; fixes the callback signature at types.rs:55), and it reports capabilities so the loop can decide what to send.

```rust
#[async_trait::async_trait]
pub trait Provider: Send + Sync {
    /// Which provider this is, for diagnostics and error reporting.
    fn kind(&self) -> ProviderKind;

    /// Stream a chat request. Never blocks on a callback; the caller drains
    /// the stream. The library never prints.
    async fn chat_stream(&self, request: ChatRequest) -> Result<EventStream, Error>;

    /// Embed texts. Implementations that cannot embed return
    /// `Error::Unsupported`. The `Embedder` trait arrives in Phase 4; this
    /// stays for parity with the hosted APIs.
    async fn embed(&self, request: EmbedRequest) -> Result<EmbedResponse, Error> {
        Err(Error::Unsupported { feature: "embeddings".into(), provider: self.kind().prefix().into(), model: None })
    }

    /// What this provider and model can do. Consulted before sending, so an
    /// unsupported request fails locally with a clear message rather than as
    /// a provider 400.
    fn capabilities(&self, model: &str) -> Capabilities;

    // --- Batch: optional methods with an `Unsupported` default (Open 6). ---

    async fn batch_submit(&self, request: BatchRequest) -> Result<BatchHandle, Error> {
        Err(self.batch_unsupported())
    }
    async fn batch_status(&self, handle: &BatchHandle) -> Result<BatchProgress, Error> {
        Err(self.batch_unsupported())
    }
    async fn batch_results(&self, handle: &BatchHandle) -> Result<Vec<(RequestId, BatchItem)>, Error> {
        Err(self.batch_unsupported())
    }
    async fn batch_cancel(&self, handle: &BatchHandle) -> Result<(), Error> {
        Err(self.batch_unsupported())
    }
    // Private helper, not part of the public surface.
    // fn batch_unsupported(&self) -> Error;
}
```

`ChatRequest` is the public, provider-neutral request the agent builds. It carries owned data only; the wire encoding is the provider's private business.

```rust
/// One provider request. Built by the agent, owned, serde.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct ChatRequest {
    pub model: ModelString,
    /// The messages, excluding the system prompt.
    pub messages: Vec<Message>,
    /// Rendered system prompt, owned by the agent and never in `messages`.
    pub system: Option<String>,
    /// Tool definitions. Raw JSON so the provider encodes them its way.
    pub tools: Vec<ToolSpec>,
    pub tool_choice: ToolChoice,
    pub options: RunOptions,
    /// The conversation id, for prompt-cache keys and routing headers.
    pub conversation_id: ConversationId,
    /// Extra body fields, passed through unchanged (rule 8).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider_options: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct ToolSpec {
    pub name: String,
    pub description: String,
    /// The full JSON Schema (schemars output, transformed per provider).
    pub parameters: serde_json::Value,
    /// Whether strict mode was requested. MCP tools are never strict
    /// (rmcp.md §6).
    pub strict: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub enum ToolChoice {
    Auto,
    None,
    Required,
    Tool { name: String },
}
```

Capabilities (roadmap batch question, ROADMAP.md:187; gloo.md §Impact 10; responses-api.md §7):

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct Capabilities {
    pub context_window: Option<u32>,
    pub max_output_tokens: Option<u32>,
    pub streaming: bool,
    pub tools: bool,
    pub structured_output: bool,
    pub thinking: ThinkingSupport,
    /// Effort levels the model accepts.
    pub effort_levels: Vec<Effort>,
    pub images: bool,
    pub pdf: bool,
    /// Batch support for this model, including the discount (Open 6).
    pub batch: BatchSupport,
    /// Minimum cacheable prefix in tokens, when the provider caches.
    pub cache_min_tokens: Option<u32>,
}

/// How `Thinking::Off` maps on this model (anthropic-messages.md Q3).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum ThinkingSupport {
    /// `Off` maps to a per-model value and always succeeds.
    Mappable,
    /// `Off` is an error before sending (Opus 5.5, Fable 5.1).
    AlwaysOn,
    /// The provider has no thinking to configure.
    NotApplicable,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum BatchSupport {
    Unsupported,
    /// Supported, with the batch discount as a fraction (0.5 = 50% off).
    Supported { discount: f32 },
    /// Not known without a live call; treat as unsupported.
    Unknown,
}
```

### 7.3 `ConversationStore`

(Answers part of ROADMAP.md:184.) A minimal async trait; implementations come later (file, SQLite — ROADMAP.md:318). It stores whole `Conversation`s, so it needs no knowledge of the message model beyond serde.

```rust
/// Persistence for conversations. Implementations arrive after Phase 2.
#[async_trait::async_trait]
pub trait ConversationStore: Send + Sync {
    /// The stored conversation, or `None` if the id is unknown.
    async fn load(&self, id: &ConversationId) -> Result<Option<Conversation>, Error>;
    /// Create or replace. Overwriting an id is the expected update path.
    async fn save(&self, conversation: &Conversation) -> Result<(), Error>;
    /// All known ids, for a "resume a conversation" UI.
    async fn list(&self) -> Result<Vec<ConversationId>, Error>;
    async fn delete(&self, id: &ConversationId) -> Result<(), Error>;
}
```

`Conversation::load`/`save` are thin wrappers over these, and `Agent::run` writes through the store at the end of a successful turn when one is configured. Batch handles are serializable separately (see §10); the store is only for conversations.

**(Open 5):** whether Phase 2 ships a `MemoryStore` so mock tests can exercise save/resume. Recommendation: yes, a tiny `MemoryStore` behind nothing (always compiled), because the Phase 2 gate requires a save/resume test (ROADMAP.md:210).

## 8. The agent

### 8.1 `Agent` and `AgentBuilder`

`Agent` is settings only — model, system prompt, tools, knowledge, options — cheap to clone and shareable across tasks and users. The one exception is MCP: an agent built with MCP servers holds shared `Arc` handles to the live connections (ROADMAP.md:37, 141; rmcp.md §4).

```rust
/// An agent: settings plus shared provider/tool handles. Cheap to clone.
#[derive(Clone)]
pub struct Agent {
    /* Arc<dyn Provider>, Arc<[Arc<dyn Tool>]>, system, options, hooks,
       optional RAG engine, optional MCP handles, optional conversation store,
       price table — all private */
}

impl Agent {
    pub fn builder() -> AgentBuilder;

    // --- Settings (clone-and-change; each returns a new Agent) ---

    /// The current model string.
    pub fn model(&self) -> &ModelString;
    /// Switch model, keeping everything else. A handoff or a local/hosted
    /// swap is exactly this (ROADMAP.md:31).
    pub fn with_model(&self, model: impl Into<String>) -> Result<Agent, Error>;
    /// Replace the system prompt. Restarts the prompt cache and, on Anthropic
    /// current models, invalidates earlier thinking (see §9).
    pub fn with_system(&self, system: impl Into<String>) -> Agent;
    /// Replace the tool set. Same cache/thinking note as `with_system`.
    pub fn with_tools(&self, tools: Vec<Arc<dyn Tool>>) -> Agent;
    /// Current options.
    pub fn options(&self) -> &RunOptions;

    // --- Running ---

    /// Run one turn against `conversation`, appending and committing on
    /// success. Returns the assistant turn (text, tool calls, usage).
    pub async fn run(&self, conversation: &mut Conversation, input: impl Into<Message>)
        -> Result<Turn, Error>;

    /// Streaming variant. The returned stream yields `Event`s; the
    /// conversation is committed when the stream reaches `Event::Stop`.
    pub fn run_stream<'a>(&'a self, conversation: &'a mut Conversation, input: impl Into<Message>)
        -> Result<EventStream, Error>;

    /// One-shot, no conversation. Uses native JSON-schema output, validates
    /// locally, returns the parsed value (ROADMAP.md:208).
    pub async fn extract<T>(&self, input: impl Into<Message>) -> Result<T, Error>
    where T: serde::de::DeserializeOwned + schemars::JsonSchema + Send + 'static;

    /// Wrap this agent as a tool another agent can call (ROADMAP.md:39).
    pub fn as_tool(self: &Arc<Self>, name: impl Into<String>, description: impl Into<String>)
        -> Arc<dyn Tool>;
}

/// The committed result of one turn.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct Turn {
    pub message: Message,
    pub usage: Option<Usage>,
    pub stop: StopReason,
    /// How many agent-loop steps this turn took (1 = no tool calls).
    pub steps: usize,
}
```

```rust
/// Builds an `Agent`. Every setter takes `self` and returns `Self`.
pub struct AgentBuilder { /* private */ }

impl AgentBuilder {
    pub fn new() -> Self;

    // Provider and model
    pub fn model(self, model: impl Into<String>) -> Self;
    /// Install a provider for a custom model string. Without this, the model
    /// prefix selects the built-in provider.
    pub fn provider(self, provider: Arc<dyn Provider>) -> Self;

    // Prompt and tools
    pub fn system(self, system: impl Into<String>) -> Self;
    pub fn tool(self, tool: Arc<dyn Tool>) -> Self;
    pub fn tools(self, tools: impl IntoIterator<Item = Arc<dyn Tool>>) -> Self;

    // Options
    pub fn options(self, options: RunOptions) -> Self;
    pub fn price_table(self, table: PriceTable) -> Self;

    // Knowledge (feature = "rag")
    #[cfg(feature = "rag")]
    pub fn knowledge(self, path: impl Into<std::path::PathBuf>) -> Self;
    #[cfg(feature = "rag")]
    pub fn knowledge_mode(self, mode: KnowledgeMode) -> Self;

    // MCP (feature = "mcp")
    #[cfg(feature = "mcp")]
    pub fn mcp(self, server: McpServerSpec) -> Self;

    // Persistence
    pub fn conversation_store(self, store: Arc<dyn ConversationStore>) -> Self;

    // Observability hooks
    pub fn on_request(self, hook: impl Fn(&ChatRequest) + Send + Sync + 'static) -> Self;
    pub fn on_response(self, hook: impl Fn(&Turn) + Send + Sync + 'static) -> Self;
    pub fn on_tool_result(self, hook: impl Fn(&ToolCallId, &Result<ToolOutput, Error>) + Send + Sync + 'static) -> Self;

    // Approval gate (ROADMAP.md:124, 198)
    pub fn on_tool_call(self, hook: impl Fn(&ToolCall) -> Approval + Send + Sync + 'static) -> Self;

    // Build (async: connects MCP servers)
    pub async fn build(self) -> Result<Agent, Error>;

    /// Build from data. The named tools are resolved by `tools`.
    pub async fn from_spec(spec: AgentSpec, tools: &dyn ToolRegistry) -> Result<Agent, Error>;
}
```

**(Open 8)** resolves the name clash on `on_tool_call`: the roadmap uses it for both the approval gate (Decisions 124) and the observability hooks (Phase 2 line 207). Recommendation: `on_tool_call` is the **approval gate** returning `Approval`; the observability hook is `on_tool_result`, which sees the call id and the outcome.

```rust
/// The approval hook's answer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum Approval {
    Approve,
    /// Refuse; a denial goes back to the model as a tool result with
    /// `is_error: true` (ROADMAP.md:198).
    Deny,
}
```

`ToolCall` here is the pre-execution view passed to the hook:

```rust
/// The tool-call information a hook sees.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct ToolCall {
    pub id: ToolCallId,
    pub name: String,
    /// Raw arguments JSON, as the model produced it.
    pub input: Box<serde_json::value::RawValue>,
    /// Untrusted annotations from the tool, if any (rmcp.md §6).
    pub annotations: Option<ToolAnnotations>,
}
```

A refusal or a tool error is **not** an `Err` out of `run`: it is appended as a `Part::ToolResult { is_error: true }` and the loop continues (ROADMAP.md:203). `Error::Tool` only surfaces when a tool error must reach the caller.

### 8.2 `AgentSpec`

(Answers part of ROADMAP.md:184.) `AgentSpec` is the serde form of an `Agent` (rule 2), loadable from a config file. It holds only data, so it cannot carry `Arc<dyn Tool>`; named tools are resolved by a registry at build time (rmcp.md §4).

```rust
/// A serializable agent description. Load it from config, then build it.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct AgentSpec {
    pub model: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub system: Option<String>,
    /// Tool names, resolved by the `ToolRegistry` passed to `build`.
    #[serde(default)]
    pub tools: Vec<String>,
    /// Knowledge paths (feature = "rag").
    #[serde(default)]
    pub knowledge: Vec<std::path::PathBuf>,
    #[serde(default)]
    pub knowledge_mode: KnowledgeMode,
    /// MCP servers (feature = "mcp"). Only specs; connected at build.
    #[serde(default)]
    pub mcp: Vec<McpServerSpec>,
    #[serde(default)]
    pub options: RunOptions,
    /// The tool-loop step limit. `None` uses the `RunOptions` default.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_steps: Option<usize>,
}

/// Resolves tool names to tools, for `AgentSpec::build`.
pub trait ToolRegistry: Send + Sync {
    fn get(&self, name: &str) -> Option<Arc<dyn Tool>>;
    fn names(&self) -> Vec<String>;
}
```

```rust
/// One MCP server, as data. Serde so `AgentSpec` can hold it. Connected at
/// `build().await`; no rmcp types appear here (rmcp.md §4).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub enum McpServerSpec {
    /// A named stdio server: `npx`, `uvx`, a binary.
    Stdio {
        name: String,
        command: String,
        #[serde(default)]
        args: Vec<String>,
        #[serde(default)]
        env: std::collections::BTreeMap<String, String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        cwd: Option<std::path::PathBuf>,
        /// Clear the inherited environment before adding `env`, so the
        /// server does not see the app's API keys (rmcp.md §5).
        #[serde(default)]
        env_clear: bool,
    },
    /// A streamable-HTTP server with a static bearer token or custom headers.
    Http {
        name: String,
        url: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        bearer: Option<String>,
        #[serde(default)]
        headers: std::collections::BTreeMap<String, String>,
    },
}

impl McpServerSpec {
    pub fn name(&self) -> &str;
}
```

`KnowledgeMode` matches the Decision at ROADMAP.md:143:

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[non_exhaustive]
pub enum KnowledgeMode {
    /// Search with the user message, append results as a `Part::Knowledge` in
    /// the same user turn, persist it. The default (ROADMAP.md:143).
    #[default]
    Inject,
    /// Expose a `search_knowledge` tool the model calls itself.
    Tool,
    /// Both.
    Both,
}
```

## 9. History is append-only, and how `with_system` / `with_tools` / handoff stay valid

(Answers ROADMAP.md:185.) Nothing that has been sent to a provider is edited or removed later (rule 10; anthropic-messages.md §Impact 2). On current Claude models, a thinking block's signature binds the top-level `system`, the `tools`, and every earlier message; if any changes, that block **and every later thinking block** become invalid, which is a 400 for accounts created on or after 2026-08-31 (anthropic-messages.md Q3).

Three options were on the table (ROADMAP.md:185):

- **(a) A mid-conversation `role: "system"` message.** Opus 5.5 / Sonnet 5.5 / Haiku 5.5 / Fable 5.1 accept a `{"role":"system"}` entry inside `messages`, and unlike editing the top-level `system` it keeps the cache and the thinking blocks valid (anthropic-messages.md Cross-check). But it is only accepted on those models, and moving the system prompt into `messages` changes where it lives relative to history.
- **(b) Client-side removal of thinking blocks whose prefix no longer matches.** The protocol layer tags each `ProviderItem` with `prefix_fingerprint` (§4.2) and drops the mismatched block and every later one. This is portable across all current models and needs no beta.
- **(c) The API's `drop_block` option** (beta `thinking-binding-controls-2026-08-01`). It drops the first mismatched block and all later ones server-side, and reports `input_transformations` (anthropic-messages.md Q3, Cross-check).

**Recommendation (Open 2):** ship **(b)** as the default in the Anthropic protocol layer, and keep **(c)** reachable through `provider_options`, and treat **(a)** as the future-preferred optimisation once every target model accepts mid-conversation system messages. Reasons: (b) needs no beta flag and no model allow-list, it is the exact behaviour the docs recommend ("strip blocks yourself only to save input tokens"), and it makes the two first-class features that change the prefix — `with_system`/`with_tools` and a handoff — never 400. Inside an in-flight tool loop, no block is ever dropped (the API requires the latest assistant turn's thinking blocks to be intact, anthropic-messages.md Q3).

Consequences to document in user-facing docs:

- `with_system`, `with_tools` and handoff to an agent with a different system prompt or tools **restart the prompt cache** and **invalidate earlier thinking** (the library strips it so the request stays valid) (ROADMAP.md:185; anthropic-messages.md Q3 cross-check).
- Changing the model does **not** invalidate Anthropic thinking: blocks are replayed to any Claude model, and the API drops the ones it cannot read, unbilled (anthropic-messages.md Q3). The model-only half of the old roadmap rule is corrected here; the provider half stands (thinking items go only to the dialect that made them).
- RAG is safe because injected knowledge is a `Part::Knowledge` **inside the persisted user turn** (`KnowledgeMode::Inject`, §8.2): it is appended once and never spliced into or out of a past turn (anthropic-messages.md §Impact 2).

A mock test asserts the stripped request body after `with_system` (anthropic-messages.md §Impact 3).

## 10. `RunOptions` and batch

### 10.1 `RunOptions`

(Answers ROADMAP.md:186.) `effort` is first-class and held constant per agent, because changing it starts the prompt cache over (anthropic-messages.md Q3/Q4). `thinking` is `Auto` / `Adaptive` / `Off`, mapped per model. `temperature` and `top_p` are `Option` and omitted unless set, since current Claude and OpenAI reasoning models reject them (anthropic-messages.md Q7; responses-api.md §7).

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct RunOptions {
    /// Reasoning effort. `None` uses the model's default. First-class, held
    /// constant per agent.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub effort: Option<Effort>,

    /// Thinking mode. Mapped per model; see `ThinkingSupport`.
    #[serde(default)]
    pub thinking: Thinking,

    /// How much thinking text comes back.
    #[serde(default)]
    pub thinking_display: ThinkingDisplay,

    /// Sent only when set. Omitted by default (anthropic-messages.md Q7).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_output_tokens: Option<u32>,
    #[serde(default)]
    pub stop_sequences: Vec<String>,

    /// Tool-loop step limit. `None` uses the library default.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_steps: Option<usize>,

    /// Per-request timeout. Covers stream-start and the gap between events
    /// (the idle timeout), not the whole request (anthropic-messages.md Q9).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub timeout: Option<std::time::Duration>,

    /// Prompt-cache TTL. 5 minutes is right for an agent loop; 1 hour for a
    /// human-paced chat (anthropic-messages.md Q4).
    #[serde(default)]
    pub cache_ttl: CacheTtl,

    /// Extra body fields, passed through unchanged (rule 8).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider_options: Option<serde_json::Value>,
    /// Extra HTTP headers, passed through unchanged (gloo.md §Impact 5).
    #[serde(default)]
    pub provider_headers: std::collections::BTreeMap<String, String>,
}

impl Default for RunOptions { /* all omitted / defaults */ }

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum Effort {
    Low,
    Medium,
    High,
    XHigh,
    Max,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[non_exhaustive]
pub enum Thinking {
    /// Omit the field; the model's own default applies (adaptive on current
    /// Claude models).
    #[default]
    Auto,
    /// Ask for adaptive thinking explicitly.
    Adaptive,
    /// Turn thinking off, mapped per model. On a model that cannot turn it
    /// off (Opus 5.5, Fable 5.1) this is `Error::InvalidRequest` **before**
    /// sending (anthropic-messages.md Q3).
    Off,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[non_exhaustive]
pub enum ThinkingDisplay {
    /// The default on current Claude models; thinking text is not streamed.
    #[default]
    Omitted,
    Summarized,
    /// Beta on Claude; progress notes between tool calls.
    Updates,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[non_exhaustive]
pub enum CacheTtl {
    #[default]
    FiveMinutes,
    OneHour,
}
```

Local models map their own way: `thinking = Auto` on `local/` should mean off, because Qwen3 thinking multiplies tokens on small models (mistralrs.md §Impact 2), overridable per agent. The mapping from `Thinking`/`Effort` to a provider's wire value is private, driven by `Capabilities::thinking` and `Capabilities::effort_levels`.

### 10.2 Batch

(Answers ROADMAP.md:187.) **Recommendation (Open 6):** batch is a set of **optional `Provider` methods with an `Unsupported` default**, not a separate trait. Rationale: a custom provider that implements only `chat_stream` keeps compiling (the Phase 3 requirement that one example runs against all providers, ROADMAP.md:257); a separate `BatchProvider` trait would force callers to hold `Arc<dyn BatchProvider>` and downcast from `Arc<dyn Provider>`, which the owned/`Arc<dyn …>` rule does not want. Support is reported **per model** in `Capabilities::batch` (batch.md §Impact). The free function and types follow batch.md §6:

```rust
/// Submit a batch. One model per batch, IDs validated before any network call.
pub async fn batch(model: impl AsRef<str>, requests: Vec<BatchRequest>) -> Result<BatchHandle, Error>;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct BatchHandle {
    pub version: u32,
    pub provider: ProviderKind,
    pub model: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub base_url: Option<String>,
    pub parts: Vec<BatchPart>,
    pub request_ids: Vec<RequestId>,
    pub created_at: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expires_at: Option<String>,
}
// BatchPart, BatchStatus, EndReason, BatchProgress, BatchItem, BatchOptions
// are as sketched in docs/design/spikes/batch.md §6: no credentials in the
// handle; results as Succeeded | Failed | Cancelled | Expired | Missing.
```

Providers without batch (Gloo, unsupported Grok models, custom) return `Error::Unsupported { feature: "batch", … }` before any network call, unless `BatchOptions::concurrent_fallback(max_in_flight)` is set, which runs the requests as ordinary bounded-concurrency calls (batch.md §6).

## 11. One-shot calls and typed output

### 11.1 `nucleus::complete`

```rust
/// Level 1: a single hosted or local call, no conversation, no tools.
///
/// ```no_run
/// # async fn example() -> Result<(), nucleus::Error> {
/// let text = nucleus::complete("openai/gpt-5.1", "Summarize: …").await?;
/// # Ok(()) }
/// ```
pub async fn complete(model: impl AsRef<str>, prompt: impl Into<String>) -> Result<String, Error>;
```

It reads the provider's API key from the environment (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `XAI_API_KEY`, `GLOO_API_KEY`), sends one user message with no system prompt and the cheapest sensible options, and returns the concatenated text. It returns `Error::Refused` on a refusal so `?` works (anthropic-messages.md Q5).

### 11.2 `agent.extract::<T>()`

Uses each provider's native JSON-schema output — never a forced tool call, which is a 400 on Opus 5.5 / Sonnet 5.5 / Fable 5.1 (ROADMAP.md:208; anthropic-messages.md Q6; responses-api.md Q4). The path is: `schemars::schema_for!(T)` → a per-provider schema transform (OpenAI strict: every property required, optionals nullable, `additionalProperties: false`; Anthropic: a port of the Python SDK's `transform_schema`; xAI: mostly as-is; local: `Constraint::JsonSchema` plus `"x-guidance": {"whitespace_flexible": false}` and a `max_len` cap) → send as the format field with no tools → check `stop_reason` before parsing (refusal → `Error::Refused`, max_tokens → `Error::Truncated`) → parse and **validate locally against the original schema**. A validation failure is `Error::Serialization` carrying the raw text. Extract schemas must not contain a `reasoning` field, which triggers `reasoning_extraction` refusals (anthropic-messages.md Q5/Q6).

```rust
/// Convert a schemars schema for one provider dialect. Private; exposed for
/// tests.
pub(crate) fn transform_schema(for_dialect: &str, schema: serde_json::Value) -> Result<serde_json::Value, Error>;
```

## 12. The public `MockProvider`

A public deterministic provider so clients and nucleus can test the loop without keys or network (ROADMAP.md:209). It both replays scripted `Event` sequences and can be built from fixture SSE bytes for parser tests.

```rust
/// A provider that returns scripted responses. Public so clients can test.
pub struct MockProvider { /* private */ }

impl MockProvider {
    pub fn new() -> Self;
    /// Enqueue one canned response: a list of events ending in `Stop`.
    pub fn push(self, events: Vec<Event>) -> Self;
    /// Enqueue a text response ending in `EndTurn`.
    pub fn push_text(self, text: impl Into<String>) -> Self;
    /// Enqueue a tool call and then, after the tool runs, `text`.
    pub fn push_tool_call(self, name: &str, input: serde_json::Value) -> Self;
    /// Fail the next request.
    pub fn push_error(self, error: Error) -> Self;
    /// Every request the mock received, for assertions.
    pub fn requests(&self) -> Vec<ChatRequest>;
}

#[async_trait::async_trait]
impl Provider for MockProvider { /* … */ }
```

`MockProvider` is also the backbone of the shared provider test suite (Phase 3, ROADMAP.md:253) and of the agent-loop tests (ROADMAP.md:210).

## 13. Slash commands

(Answers ROADMAP.md:183.) **Recommendation (Open 3):** the slash commands (`/help`, `/reset`, `/exit`, `/quit` in `src/chat/commands.rs`) **leave the library** and belong to apps. Rationale: they are a terminal/UI affordance; their output is human text, and the library never prints (ROADMAP.md:83). `/reset` and `/exit` are not library concepts — the library primitive is `Conversation::clear()` (§4.3), which the app calls when it sees `/reset`. Keeping the parser in the library would also mean shipping user-facing strings the library must never emit. `nucleus-cli` owns its command set. The `CommandEffect`/`action`/`parse` code (commands.rs:1-122) is deleted in Phase 2; apps keep their own version.

## 14. Feature flags

Only add things (rule 9; Cargo.toml:46-63). The public API above compiles under `hosted` alone.

| Feature | Adds | Public surface guarded by it |
|---|---|---|
| `hosted` (default) | reqwest, futures | the hosted providers, `nucleus::complete` for hosted models |
| `local` | mistral.rs | the local provider; `ModelString` still parses `local/` no matter what, so a `local/` model returns `Error::Unsupported` without the feature |
| `rag` | local embeddings + a store | `AgentBuilder::knowledge`, `KnowledgeMode`, `KnowledgeSource` in `Part::Knowledge` |
| `lancedb` | LanceDB | (implied by `rag` until Phase 4) |
| `mcp` | rmcp (wrapped) | `McpServerSpec`, `AgentBuilder::mcp` |
| `experimental` | — | Grok account login |
| `metal` / `cuda` | mistral.rs backends | the `Local::builder()` device/quantisation options |

Rule 6 holds: `rmcp`, mistral.rs and reqwest types never appear in a public signature. MCP tools are `Arc<dyn Tool>`; MCP errors map to `nucleus::Error` kinds (rmcp.md §4); MCP server stderr goes to `tracing` (rmcp.md §5).

## 15. Phase 6 examples compiled against `todo!()` stubs

These programs are written in Phase 2 against the signatures above, with the bodies of the not-yet-implemented parts as `todo!()`, so the public API is checked before the internals exist (ROADMAP.md:76, 284). They live in `examples/` and are gated by `[[example]] required-features`, as the current `Cargo.toml:75-101` does.

| Example | `required-features` | Uses | Phase 2 stub |
|---|---|---|---|
| `one_shot` | `hosted` | `nucleus::complete` | provider `chat_stream` is `todo!()`; the signature must compile |
| `batch` | `hosted` | `nucleus::batch`, `BatchHandle`, `refresh`/`wait`, resume from a saved handle | `batch_submit` default returns `Unsupported`; the example compiles against the types |
| `agent` | `hosted` | `Agent::builder().system().tools().build()`, `run` | the loop is `todo!()`; builder and signatures compile |
| `local_vs_hosted` | `local` | the same example body with only the model string changed (`openai/…` vs `local/…`) | both providers are `todo!()` stubs behind the same trait |
| `custom_tool` | none | `tool_fn` with typed arguments, approval hook | `Tool::call` is exercised only in tests |
| `typed_extract` | `hosted` | `agent.extract::<T>()`, `#[derive(JsonSchema)]` | schema transform is `todo!()` |
| `rag` | `rag` | `.knowledge("./docs")`, `KnowledgeMode::Inject`, a `Part::Knowledge` citation | indexing is `todo!()` (Phase 4) |
| `mcp` | `mcp` | `.mcp(McpServerSpec::stdio("fs", "npx", …))` | connection is `todo!()` (Phase 5) |
| `streaming` | `hosted` | `run_stream`, draining `Stream<Item = Event>`, no printing | provider stream is `todo!()` |
| `swap_and_handoff` | `hosted` | `with_model`, running a second agent on the same `Conversation`, `as_tool` | the loop is `todo!()` |

Each example must build with `cargo build --examples --all-features` (and with only its required features), which is part of the Phase 2 gate. The `Done when` for Phase 2 (ROADMAP.md:210) is met by mock-provider tests covering the loop, swaps, handoff, agent-as-tool and save/resume; these examples are the compile-time half of "examples first" (ROADMAP.md:76).

## 16. What this note does not cover

- Provider implementations, auth, retries, timeouts and fallback chains (Phase 3).
- `Embedder` and `VectorStore` and the retrieval benchmark (Phase 4).
- The MCP client beyond `McpServerSpec` and the wrapped-`Tool` seam (Phase 5).
- `nucleus-cli`'s `login`/`logout` (ROADMAP.md:123).

## Open questions

Each is a choice made above; the recommendation is the default unless review says otherwise.

1. **Model-string grammar for a local absolute path.** Ship `local//abs/path` (split on the first `/`, the tail's leading `/` marks a path), or add a `file://` scheme so it reads `local/file:///abs/path`? **Recommendation: ship `local//abs/path`.** It needs no extra syntax, is unambiguous, and the grammar stays "split on the first slash". Revisit if it grates. (ROADMAP.md:181)

2. **How `with_system` / `with_tools` / handoff stay valid on Anthropic.** (a) mid-conversation `role:"system"` message, (b) client-side removal of mismatched thinking blocks, (c) the beta `drop_block`. **Recommendation: (b) now, (a) as a later optimisation, (c) reachable via `provider_options`.** (b) is portable and needs no beta; (a) only works on four current models; (c) is beta. (ROADMAP.md:185; anthropic-messages.md Q3, Cross-check)

3. **Slash commands.** Keep `/help`, `/reset`, `/exit` in the library or move them to apps. **Recommendation: move to apps.** They are UI text; the library never prints. `Conversation::clear()` is the primitive. (ROADMAP.md:183)

4. **`Usage` cost.** Provider-reported cost first, then a built-in overridable price table, with cache-read/write tokens, batch discounts and Gloo's 6.5% guarded fee applied on top; `cost = None` when unknown. **Recommendation: as §5.2.** The only sub-choice is `f64` for `Cost.usd`; a decimal type would avoid float drift, but `f64` matches every provider's reporting and is what the catalog gives. (ROADMAP.md:182)

5. **`ConversationStore` implementations in Phase 2.** Ship a `MemoryStore` now, or only the trait. **Recommendation: ship `MemoryStore`** so the required save/resume test can run without a file or SQLite backend. (ROADMAP.md:184, 318)

6. **Batch shape.** Optional `Provider` methods with an `Unsupported` default, or a separate `BatchProvider` trait. **Recommendation: optional methods.** Custom providers keep compiling, no downcasting, and `capabilities()` reports support per model. (ROADMAP.md:187; batch.md §Impact)

7. **`AgentSpec` tool resolution.** `AgentSpec` holds tool **names** (rule 2), so building needs a resolver. Pass a `&dyn ToolRegistry` to `build`, or let the app register tools on the builder after `from_spec`. **Recommendation: `AgentBuilder::from_spec(spec, &dyn ToolRegistry)`.** Keeps the spec pure data and the build explicit. (ROADMAP.md:184; rmcp.md §4)

8. **Hook naming.** The roadmap uses `on_tool_call` for both the approval gate and the observability hook. **Recommendation: `on_tool_call` is the approval gate returning `Approval`; the observer is `on_tool_result`.** Matches Decisions 124 and Phase 2 line 198, which both spell the approval gate as `.on_tool_call(|call| Approve | Deny)`. (ROADMAP.md:124, 198, 207)

9. **`Error` and `Event` serde depth.** `Error` is serde-serializable and so is every payload it carries; the stream is `Stream<Item = Event>` with `Event::Error(Error)`, not `Stream<Item = Result<Event, Error>>`. **Recommendation: as written**, because the roadmap fixes the stream item as `Event` (ROADMAP.md:206) and rule 1 wants serde data. If review prefers the idiomatic `Result` item, that is the single line to change and it ripples through `Agent::run_stream` only.

10. **`Thinking::Auto` default on `local/`.** Follow the model default (Qwen3 thinks, expensively) or default local to off. **Recommendation: default `local/` to off via the per-agent mapping**, because thinking multiplies tokens on small models and the spike recommends it (mistralrs.md §Impact 2). (ROADMAP.md:186)
