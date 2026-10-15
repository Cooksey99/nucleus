# Nucleus 0.2 Roadmap

## Goal

A lightweight Rust library that is extremely easy to use and allows deeper customization when needed.
It should be a better choice than calling OpenAI, Anthropic or OpenRouter directly when adding LLMs to software.

- Two inference types behind one API: **Local** (always mistral.rs) and **Hosted** (OpenAI, Anthropic, Grok, Gloo AI).
- Switching between local and hosted changes only the model string.
- RAG is one line to set up.
- MCP servers plug in as tools.
- Breaking changes are fine; this is a rewrite toward 0.2.

## Target API

```rust
// Level 1: one-shot call
let text = nucleus::complete("openai/gpt-…", "Summarize: …").await?;

// Level 2: an agent with tools, knowledge and MCP
let agent = Agent::builder()
    .model("anthropic/claude-opus-5-5")        // or "local/Qwen/Qwen3-8B", "grok/…", "gloo/…"
    .system("You are a senior Rust reviewer.")
    .tools(tools::files())
    .knowledge("./docs")
    .mcp(McpServer::stdio("fs", "npx", ["-y", "@modelcontextprotocol/server-filesystem", "."]))
    .build().await?;

let mut convo = Conversation::new();
agent.run(&mut convo, "Review src/lib.rs").await?;
agent.with_model("local/Qwen/Qwen3-8B").run(&mut convo, "Summarize that").await?;   // swap, same chat
let invoice: Invoice = agent.extract(&pdf_text).await?;                             // typed output

// Level 3: implement Provider, Tool, Embedder, VectorStore or ConversationStore yourself
```

- `Agent` is settings only (model, system prompt, tools, knowledge, options). It holds no conversation state, is cheap to clone, and can be shared across tasks and users. The one exception is MCP: an agent built with MCP servers holds shared handles to the live connections (see Decisions).
- `Conversation` is plain data with a stable ID. It is serializable, so it can be saved and resumed.
- Multiple agents need no orchestrator: `agent.as_tool(..)` handles delegation, and running another agent on the same conversation is a handoff. Add an `Orchestrator` type only when a real need appears, such as a shared budget or tracing across agents.
- MCP servers are named (`"fs"` above) so their tools can be prefixed (`fs__read_file`). `McpServer` is a serde spec, so `AgentSpec` can hold it.

## Architecture

```
Provider (trait): chat_stream(request) -> Stream<Event>, embed(texts), capabilities()
├── Local: mistral.rs (GGUF path, HF repo, HF GGUF)                 [feature: local]
└── Hosted: protocol + dialect + auth + base URL                    [feature: hosted, default]
    ├── Responses API     → OpenAI, Grok (xAI), Gloo AI   (one lenient parser; a private per-provider dialect)
    └── Messages API      → Anthropic
Auth: API key | Grok account login [experimental]  (auth type stays #[non_exhaustive] for OAuth on Azure, Vertex, Bedrock)
Batch: submit → handle (serializable) → results by request ID
       OpenAI, Anthropic, xAI (some models); local runs concurrently; Gloo has no batch API
```

## Design rules that keep later work possible

These cost little now and keep the door open to an HTTP service, other-language SDKs, persistence and client production needs, without building any of them yet.

1. **Every public data type is serde-serializable:** `Message`, `Event`, `ToolCall`, `Usage`, `Conversation`, `RunOptions`.
2. **Agents can be defined as data.** An `AgentSpec` (model string, system prompt, tool names, knowledge, MCP servers) can be loaded from config, not only built in code.
3. **The public API uses owned data and trait objects** (`Arc<dyn Provider>`, `Arc<dyn Tool>`). Streams return owned `Event`s, not borrowed-lifetime callbacks.
4. **Public enums and option structs are `#[non_exhaustive]`**, so fields and variants can be added without breaking users.
5. **Extension points are traits:** `Provider`, `Tool`, `Embedder`, `VectorStore`, `ConversationStore`, and hooks.
6. **Protocol details stay private.** New providers (Bedrock, Vertex, Azure, …) can be added without API changes. No `rmcp` or mistral.rs types appear in the public API.
7. **The library has no server code.** A future service lives in `nucleus-cli` or its own crate.
8. **There is always an escape hatch:** `provider_options(json!(..))` sends extra body fields, `provider_headers(..)` sends extra HTTP headers, and raw provider responses are always accessible.
9. **Feature flags only add things.** The default build stays small.
10. **History is append-only.** Nothing that has been sent to a provider is edited or removed later. Prompt caching depends on it everywhere, and on current Claude models an edited history invalidates earlier thinking (a 400 for accounts created on or after 2026-08-31). See `docs/design/spikes/anthropic-messages.md`.

## Quality bar

Quality comes before speed. Each phase must meet everything below that has started by then (see "When gates start").

**Process**
- Design first: each phase starts with a short design note (in `docs/design/`) that is reviewed before implementation.
- Examples first: the public API is checked by writing the example programs against stub signatures before the internals exist.
- One PR per logical change. Docs, tests and the CHANGELOG are updated in the same PR, never "later".
- Every bug fix comes with a test that fails without the fix.

**Code**
- Lints: no `unwrap`/`expect`/`panic!` in library code (`clippy::unwrap_used`, `expect_used`, `panic`), `#![forbid(unsafe_code)]`, `#![deny(missing_docs)]`, selected `clippy::pedantic` lints, rustdoc warnings treated as errors.
- One public `nucleus::Error` type, `#[non_exhaustive]`, with error kinds (retryable, auth, invalid request, provider, tool, …) and source chains. No `anyhow` in the public API.
- The library never writes to stdout or stderr; it logs only through `tracing`. That includes dependencies: no global loggers (for example mistral.rs `with_logging`), and MCP server stderr goes to `tracing`.

**Testing**
- Unit tests, plus mock-provider tests for the agent loop.
- No paid API calls in tests or CI, and no API keys needed. Hosted providers are tested against a local mock server, using fixtures built from each provider's documented request, response and stream examples and from the test data in their official SDKs.
- Local (mistral.rs) tests run a tiny GGUF model, which costs nothing.
- Property tests (`proptest`) for parsers and text processing: the SSE stream parser, chunking, model-string parsing.

**CI gates**
- build, test, doctest, clippy, rustfmt
- `cargo-deny` (licenses and security advisories)
- a minimum supported Rust version check
- a feature-combination check (`cargo hack --each-feature`)
- a snapshot of the public API (`cargo public-api`) so every API change is deliberate and visible in review
- after 0.2: `cargo-semver-checks`

**When gates start**
- Phase 0: build, test and doctest for the existing code. Clippy runs but warnings don't fail the build yet.
- End of Phase 1: clippy `-D warnings`, rustfmt, cargo-deny, minimum Rust version, feature-combination check, plus the library lints (no unwrap/expect/panic, `forbid(unsafe_code)`).
- Phase 2 onward: `#![deny(missing_docs)]`, the public-API snapshot, and the `nucleus::Error` rule for all new code.

**Footprint**
- Track the dependency count and compile time of the default build, so an increase is a deliberate decision.
- Baseline before Phase 1: 936 packages in `Cargo.lock`, 749 in the default build. mistral.rs 0.8.1 alone is 665 crates (681 with `metal`), so it must stay behind `local`.

**Dogfooding**
- Before releasing 0.2, build one realistic client-style integration with Nucleus and fix whatever is awkward.

## Working agreements

- The 0.2 rewrite lives on the `overhaul` branch. Work on a branch per phase or logical change, with one PR into `overhaul` for each. `overhaul` merges into `main` when 0.2 is released. Never commit directly to `main` or `overhaul`.
- Tick checkboxes in this file in the same PR that completes them. When a spike or design note changes a decision, update this file in that PR.
- Design notes live in `docs/design/`. Spike findings live in `docs/design/spikes/`.
- Spike code lives in a throwaway `spikes/` directory and is never merged. Only the findings notes are merged.

## Decisions

Confirmed:
- One `nucleus` crate with feature flags, plus `nucleus-cli`.
- `nucleus-cli` keeps only `login`/`logout` (Grok account login, behind `experimental`) for 0.2. Other commands wait for a real design.
- Remove the plugin `Permission` system. Replace it with an optional approval hook: `.on_tool_call(|call| Approve | Deny)`.
- Remove: CoreML (archived on a branch), the IPC `Server`, the CLI stubs (`serve`, `start`, `stop`, `status`, `list`), Qdrant, `nucleus-dev`, the Ollama provider, `detection.rs`, `qdrant_helper`, `ModelRegistry`, and unused config fields.
- `Agent` + `Conversation` replaces `ChatManager`.
- `local` (mistral.rs) is an opt-in feature. The default build is hosted only.
- RAG embeddings run locally on mistral.rs by default (`rag` turns on `local`). Other embedders (OpenAI, Gloo, or your own `Embedder`) can be chosen instead.
- An `experimental` feature holds unofficial or unstable functionality and is excluded from semver guarantees. Grok account login lives there.
- OpenRouter and generic Chat Completions are deferred (see Later).
- Batch requests are in 0.2, for cost savings. The savings are 50% on OpenAI and Anthropic, 20% on some xAI models, and none on Gloo.
- Built-in tools (`tools` module): `files()` (read, write, list directory, search) and `exec()` (shell command). Both go through the approval hook when one is set.
- Gloo auth is a static API key (`GLOO_API_KEY`, sent as a bearer token). Gloo deprecated OAuth client credentials in August 2026 and ends them by December 2026. (Phase 0.5 spike)
- OpenAI, xAI and Gloo share one Responses protocol with a lenient stream parser and a small private per-provider dialect (tool definition shape, stream payload shape, structured-output field, error body, supported request fields). (Phase 0.5 spikes)
- MCP client: `rmcp` 3.x with default features off, stdio and streamable HTTP only (rmcp dropped legacy SSE). MCP OAuth waits until after 0.2; 0.2 supports bearer tokens and custom headers. (Phase 0.5 spike)
- HTTP mocking: `mockito`, the only candidate that delivers a body in delayed chunks, behind a nucleus test helper, plus a small hand-rolled tokio server for connection faults. (Phase 0.5 spike)
- Minimum supported Rust version is at least 1.88, which both `rmcp` and mistral.rs 0.8.1 require.
- Move to reqwest 0.13 so `mcp` doesn't build a second reqwest.
- mistral.rs: use the latest GitHub release, v0.9.4, as a git dependency. It has the tool-calling fixes that crates.io 0.8.1 lacks. crates.io doesn't accept git dependencies, so publishing nucleus waits until upstream publishes 0.9.x or later (tracked in Phase 6).
- Gloo defaults to the `Direct` endpoint (prompt sent unchanged, no fee), so it behaves like the other providers. `Guarded` is a provider option.
- MCP servers on `Agent`: an agent built with MCP servers holds shared `Arc` handles to the live connections. Clones share them, and dropping the last clone shuts the servers down. `AgentSpec` holds only the `McpServer` specs and connects at `build().await`.

- Retrieved knowledge is added to the user's turn by default (`KnowledgeMode::Inject`) and saved in history, which keeps history append-only. It works the same on every model, including small local ones, and needs no extra round trip. `KnowledgeMode::Tool` (a `search_knowledge` tool the model calls itself, better for multi-step agents on strong models) and `KnowledgeMode::Both` are one setting away. The Phase 4 benchmark confirms the default before release.

Open: none. New questions go to the Phase 2 design note.

## Phases

Every phase ends with CI green and meets the quality bar. Phases 0 → 0.5 → 1 → 2 run in order. Phases 3, 4 and 5 can run in any order or in parallel after Phase 2.

### Phase 0: Guardrails
- [x] CI runs `cargo build --all-targets`, `cargo test --workspace` and `cargo test --doc` on Linux and macOS, and `cargo clippy` on Linux (it reports warnings without failing until the end of Phase 1)
- [x] Fix or delete broken tests and examples: the registry test, the nucleus-std examples, and `register()` calls that are never awaited. Delete anything Phase 1 removes anyway rather than fixing it.
- **Done when:** CI is green on `overhaul`.

### Phase 0.5: Spikes (settle the unknowns)
Short experiments in scratch code (see Working agreements). Each produces a findings note in `docs/design/spikes/` and may change the design.
No API keys and no paid calls. Hosted-provider spikes work from API references, OpenAPI specs, official SDK source and SDK test fixtures. A question that only a live call could answer is written down as open in the findings note and does not block the phase.
- [x] mistral.rs (latest crates.io release): tool calling, streamed tool calls, embedding models, JSON-schema constrained output, Metal, sharing a loaded model, compile time and dependency count ([note](docs/design/spikes/mistralrs.md))
- [x] Gloo AI: auth method, and Responses API support for tools, streaming and structured output ([note](docs/design/spikes/gloo.md))
- [x] xAI and OpenAI Responses API: function-call streaming and reasoning items with `store: false`; whether the Grok account proxy behaves the same ([note](docs/design/spikes/responses-api.md); the proxy question stays open)
- [x] Anthropic Messages API: stream event shapes, thinking-block replay, prompt caching; choose the HTTP mock tool (`mockito`) ([note](docs/design/spikes/anthropic-messages.md))
- [x] `rmcp`: maturity, transports, API stability ([note](docs/design/spikes/rmcp.md))
- [x] Batch APIs: OpenAI and Anthropic shapes and limits; xAI per-model discounts and unsupported models; whether Gloo offers batch at all; what mistral.rs offers for concurrent local requests ([note](docs/design/spikes/batch.md))
- **Done when:** each spike has findings, and the decisions and design below are updated to match.

### Phase 1: Delete and consolidate
- [x] Remove everything in the "Remove:" decision above, plus leftovers from the earlier project (Go entries in `.gitignore`, `llm-workspace` socket)
- [x] Merge the crates into `nucleus`: `provider`, `chat`, `tool`, `rag`, `tools`. `agent` replaces `chat` in Phase 2. `mcp` arrives in Phase 5.
- [x] Remove `Permission`. Plugins are stored as `Arc<dyn Plugin>` with no `Mutex`, and registration is synchronous. Renaming `Plugin` to `Tool` waits, because `provider::Tool` is already the LLM tool spec.
- [ ] Feature flags: `hosted` (default), `local`, `rag`, `lancedb`, `mcp`, `experimental`, `metal`, `cuda`
- [ ] Set the minimum supported Rust version (1.88 or later) and the edition (2021 or 2024)
- [ ] Move to reqwest 0.13
- [ ] Move mistral.rs from the current git commit to the v0.9.4 tag, and re-check the spike findings measured on 0.8.1 (tool-call history, one call per turn, JSON output that never stops, the Metal prefix-cache and short-embedding bugs); record what changed in `docs/design/spikes/mistralrs.md`
- Existing behavior (`ChatManager`, Grok, mistral.rs behind `local`, RAG behind `rag`) keeps compiling and passing tests. The rewrites happen in later phases.
- **Done when:** the default build depends on little more than reqwest, serde and tokio, and the Phase 1 CI gates pass.

### Phase 2: Core
- [ ] **Design gate:** `docs/design/core-api.md` with the full public signatures (`Agent`, `Conversation`, `Message`, `Event`, `Tool`, `Provider`, `Error`, `RunOptions`, `Batch`, hooks), plus the Phase 6 example programs compiling against `todo!()` stubs. Reviewed and approved before implementation starts. The note must also settle these questions:
  - The model-string grammar, including `local/` paths that contain slashes, absolute paths, and `repo:file.gguf`. Gloo model IDs (`gloo-anthropic-claude-sonnet-4.6`) contain no slashes.
  - How `Usage` cost is calculated. Inputs: cost reported by the provider where available (xAI reports it exactly), a built-in price table users can override, separate cache-read and cache-write tokens, batch discounts, and Gloo's 6.5% guarded fee
  - Whether slash commands (the current `/help`, `/reset`, `/exit` in `chat/commands.rs`) stay in the library or belong to apps
  - The shape of `ConversationStore` and `AgentSpec`
  - How `with_system`, `with_tools` and handoff keep history append-only on Anthropic: a mid-conversation `role: "system"` message, client-side removal of thinking blocks whose prefix no longer matches, or the API's `drop_block` option (beta). Changing system or tools also restarts the prompt cache, and the docs must say so.
  - `RunOptions`: `effort` as a first-class option; thinking as `Auto` / `Adaptive` / `Off`, mapped per model (Off is an error before sending on models that can't turn it off); thinking display; `temperature` and `top_p` optional and left out unless set, since current Claude and OpenAI reasoning models reject them
  - Batch as optional `Provider` methods with an `Unsupported` default, or a separate trait; batch support reported per model in `capabilities()`
  - Error kinds: `Refused`, `RateLimited { retry_after }`, `Overloaded`, `Timeout`, `Truncated`, content-policy blocks (separate from auth), and non-retryable billing or spend-cap errors (some 429s and 402s). Every provider error carries the request ID.
- [ ] Message model:
  - assistant messages are an ordered list of parts (text, reasoning, tool call, opaque provider item)
  - tool calls have IDs, and tool-call input is stored as raw JSON so it replays byte for byte
  - provider-specific parts (Anthropic thinking blocks, Responses reasoning items with `encrypted_content`) are tagged with the endpoint dialect and model, and replayed unchanged to the same dialect. Anthropic thinking goes back to any Claude model, because the API ignores blocks a model can't read. Responses reasoning is dropped when the model family changes. Everything provider-specific is dropped for other providers, and reasoning without `encrypted_content` is always dropped.
  - tool calls and tool results are never dropped
  - user messages can carry retrieved knowledge as its own part (text plus source file and line range), separate from what the user typed, so apps can show citations or hide it
- [ ] The system prompt belongs to the agent and is added to each request, never saved in history
- [ ] `Agent`, `Conversation` (with a stable ID, used for prompt-cache keys), `RunOptions`, `with_model` / `with_system` / `with_tools`, `as_tool`
- [ ] `Tool` trait plus `tool_fn` with typed arguments (schemars)
- [ ] Approval hook: `.on_tool_call(|call| Approve | Deny)`; a denial goes back to the model as a tool result
- [ ] `AgentSpec` (serde) that builds an `Agent`; the `ConversationStore` trait (implementations come later)
- [ ] Built-in tools rebuilt on the new `Tool` trait (see Known issues for `ExecPlugin`)
- [ ] Tool loop:
  - maximum step count
  - tool errors and denials sent back to the model as error results
  - independent calls run in parallel, and all results of a step go back in one message
  - when the step limit stops a run with tool calls still pending, they get error results, so the next request is valid
- [ ] `Stream<Item = Event>` streaming. The library never prints; it uses only `tracing`. Tool calls may arrive whole (mistral.rs) or as argument deltas (hosted), so events must not assume deltas.
- [ ] `Usage` (tokens, cache-read and cache-write tokens, cost) on every result; hooks for `on_request`, `on_response`, `on_tool_call`
- [ ] `nucleus::complete()` one-shot and `agent.extract::<T>()` typed output, using each provider's native JSON-schema output (forced tool choice is rejected by current Claude models), with schemars output converted per provider and validated locally
- [ ] Public `MockProvider` for tests (ours and clients')
- **Done when:** the loop, swaps, handoff, agent-as-tool and save/resume are covered by mock-provider tests.

### Phase 3: Providers
- [ ] Auth layer: API key, and Grok device-code login (moved from `grok.rs`, behind `experimental`)
- [ ] Responses protocol (built from the existing Grok code), configured for OpenAI, Grok and Gloo:
  - always send `store: false` and `include: ["reasoning.encrypted_content"]` where supported; the system prompt goes in `instructions`
  - one parser that follows the SSE spec, decodes UTF-8 across chunk boundaries, dispatches on the `event:` name, and accepts both nested and flattened payloads and both in-stream error shapes
  - build the result from `response.output_item.done`, ordered by `output_index`; never rely on `response.completed.output`
  - a stream that ends without a terminal event is an error; keep the incomplete reason and error codes
  - unknown event types are passed through as provider events and logged with `tracing`
  - a private dialect per provider: flat or nested tool definitions, structured-output field, error body shape, reasoning effort values, which request fields are allowed
  - Gloo gets an endpoint option: `Direct` by default, `Guarded` on request
  - the Grok account proxy is its own dialect (stricter: `instructions` required, ciphertext required on replay, extra headers)
- [ ] Messages protocol for Anthropic, over raw HTTP:
  - streaming with text, thinking, signature and `input_json_delta`; tolerate padding after the JSON; `ping` counts as liveness for the idle timeout
  - `tool_use` / `tool_result`
  - adaptive thinking and effort, mapped per model
  - prompt caching by default: a breakpoint on the last system block plus top-level automatic caching, 5-minute TTL with a `cache_ttl` option, and deterministic tool serialization
  - every stop reason surfaced: `refusal` (with its category; the partial turn isn't saved), `max_tokens`, `pause_turn`, `model_context_window_exceeded`, …
- [ ] Local provider (mistral.rs):
  - use the request's tools (the whole JSON Schema for parameters), sampling settings and history
  - always stream internally and drain the stream in our own task, so a slow consumer can't stall other requests on the model; this also gives timeouts and cancellation
  - send the full sampling settings on every request (mistral.rs defaults to `top_k = 1`); make Qwen3 thinking an explicit option
  - strip tool-call markup from the text; make sure earlier tool calls survive in the chat template (0.8.1 silently dropped them; check on 0.9.4)
  - load the model once and share it (`Arc<Model>`); a chat and an embedding model can load together
  - prefix cache and paged attention off by default (NaN logits and a hang at 16 concurrent requests on Metal)
  - `Local::builder()` for quantization, device and context size
- [ ] Reliability:
  - typed retryable errors, decided by error code as well as HTTP status
  - retries honour `retry-after-ms`, `retry-after` and `x-should-retry`, and happen only before the first output event
  - timeouts and cancellation
  - fallback chains, which also trigger on refusals
- [ ] Image and PDF input where the provider supports it, reported through `capabilities()` (Gloo takes images but not PDF)
- [ ] Batch requests:
  - `nucleus::batch(model, requests)` returns a serializable, versioned `BatchHandle` (no credentials in it), so a job can be resumed after a restart
  - request IDs match `^[A-Za-z0-9_-]{1,64}$`, are unique, and are numbered automatically if omitted
  - one model per batch; provider limits (OpenAI 50k requests / 200 MB, Anthropic 100k / 256 MB) are checked before submitting
  - polling only (`refresh`, `wait`); results once the batch has ended, as `Succeeded`, `Failed`, `Cancelled`, `Expired` or `Missing` per request ID
  - backed by OpenAI's Batch API, Anthropic's Message Batches, and xAI's Batch API for the models that support it (grok-4.3 and grok-4.20-* today)
  - settle whether xAI batch accepts the Responses shape or needs Chat Completions
  - local runs the requests concurrently through mistral.rs and can't be resumed after a restart
  - providers or models without batch support (Gloo, newer Grok models) give a clear `Unsupported` error, or run as limited concurrent normal requests if explicitly asked
  - covers one-shot and `extract` requests; multi-step tool loops are out of scope for batch
- [ ] Shared test suite for every provider:
  - fixture replays against a local `mockito` server in CI, each replayed whole and with arbitrary chunk boundaries (proptest)
  - fixtures vendored from the Anthropic SDKs (MIT) and the Vercel AI SDK (Apache-2.0), with a NOTICE; the xAI function-call stream is hand-built from the docs and marked synthetic; Gloo covers both stream shapes
  - a tiny GGUF model for local
- **Done when:** one example runs unchanged against `local/` and the mocked `openai/`, `anthropic/`, `grok/` and `gloo/` endpoints, changing only the model string, and a batch example runs against the mocked OpenAI and Anthropic batch APIs and local, including resuming from a saved `BatchHandle`.

### Phase 4: RAG
- [ ] An `Embedder` trait separate from chat; local mistral.rs embeddings by default (Qwen3-Embedding-0.6B, run on CPU even when Metal is on, because Metal gets very short inputs wrong); OpenAI and Gloo embeddings optional (Gloo reuses the OpenAI embedder with another base URL)
- [ ] A built-in vector store that saves to a file and compares every vector (the default); LanceDB behind a feature
- [ ] `.knowledge(path)`: index on first use, then re-index only files whose content hash changed
- [ ] File filtering matches whole path components, honours `.gitignore`, and skips binaries
- [ ] Chunk on paragraphs and lines with a size limit; record source file and line range for citations
- [ ] `KnowledgeMode::Inject` (default): search with the user's message and add the results to that turn as a knowledge part. Only results above a relevance threshold are added, so a message like "thanks" adds nothing. Excerpts already in the conversation aren't added again. Results are capped by count and tokens.
- [ ] `KnowledgeMode::Tool` and `KnowledgeMode::Both`: a `search_knowledge` tool that returns the same excerpts with citations
- [ ] A small retrieval benchmark run on local models only (no paid calls), comparing the modes, which confirms the default before release
- **Done when:** RAG works in three lines with zero setup and is tested.

### Phase 5: MCP
- [ ] `rmcp` client (stdio and streamable HTTP) behind the `mcp` feature, fully wrapped: no rmcp types in the public API
- [ ] MCP tools become `Tool`s, prefixed by server name (`server__tool`), with names cleaned to `[A-Za-z0-9_-]`, capped at 64 characters and mapped back; `$schema` stripped and never `strict`; the approval hook applies to them, and tool annotations reach it only as untrusted context
- [ ] Server lifecycle:
  - start each stdio server in its own process group, so shutdown also stops `npx` and `uvx` children
  - list tools, and list again on `tools/list_changed`
  - a default per-call timeout that cancels the call on the server
  - reconnect a crashed stdio server ourselves, with backoff and a cap
  - async `shutdown()` with a bounded wait, plus shutdown on drop
  - server stderr goes to `tracing`; the environment can be cleared or limited (servers otherwise see API keys)
- **Done when:** an example uses a reference MCP server, with an integration test against a small Rust test server (so CI needs no node).

### Phase 6: Docs and examples
- [ ] README built around levels 1–3
- [ ] Examples built in CI: one-shot, batch, agent, local vs. hosted, custom tool, typed extract, RAG, MCP, streaming, swap and handoff
- [ ] Cut the mdbook down to match the code; its code blocks become compiled doctests
- [ ] The mistral.rs dependency is a crates.io release (needs upstream to publish 0.9.x or later; until then nucleus can't be published)
- [ ] Changelog, release 0.2

## Known issues in code being kept or reworked

Found in an audit on 2026-10-09 and in the Phase 0.5 spikes. Most of the codebase is being deleted or rewritten. These are the issues in code that survives into the new design or serves as its starting point:

- `nucleus-core/src/provider/grok.rs` (becomes the Responses protocol):
  - tool definitions are sent but function-call events are never parsed (`parse_stream_event`), and `to_grok_input` drops `tool_calls` and tool results
  - reasoning text is printed with `eprint!` instead of being emitted as an event
  - UTF-8 is decoded per network chunk, which corrupted 2 to 13 characters in 8 of 38 replayed streams
  - SSE parsing doesn't follow the spec and silently drops bad JSON
  - a stream with no terminal event counts as success; the incomplete reason and error codes are lost
  - usage is never read, and `include` is never sent
  - the cache key and session ID are per provider instance, not per conversation
- `nucleus-core/src/patterns.rs`: `should_exclude` matches patterns as substrings of the full path, so `out` excludes `layout.rs`, `data` excludes `metadata.rs`, `models` excludes `src/models/`, and glob patterns like `*.egg-info` never match. It affects RAG indexing and the search tool.
- `nucleus-std/src/commands.rs` (`ExecPlugin`): runs `Command::new(command)` with no shell, so "git status" fails. The `args` passed by its test and `run()` helper are silently dropped.
- `nucleus-core/src/provider/mistralrs.rs`:
  - ignores `request.tools` (builds its own from the registry) and `temperature`, and passes only `properties` instead of the whole parameter schema
  - drops `tool_calls` from earlier assistant messages
  - hard-coded 30s per-chunk and 60s stream-start timeouts
  - calls `with_logging`, which installs a global stderr logger
- `nucleus-core/src/rag/`: `RagEngine::new` unwraps `config.rag` (panics when RAG isn't configured); `clear_knowledge_base` never awaits the clear; LanceDB only appends, so re-indexing duplicates chunks; the default `chunk_size` is set to `embedding_dim`.
- `nucleus-core/src/chat/manager.rs` (the reference for the new agent loop): no step limit, tool errors end the turn, and the system prompt is stored as a history message. It saves retrieved RAG text into history as plain text; the new design keeps it in history too, but as a separate knowledge part. Worth keeping: the history rollback on failure (`run_turn`) and `query_with`.
- All three YAML configs (`config.yaml`, `examples/config.yaml`, `nucleus-core/config.yaml`) fail to parse, and `Config::load_or_default` silently falls back to defaults.

## Later (the path exists, not scheduled)

| Need | Already supported by |
|---|---|
| HTTP service, OpenAI-compatible endpoint | serde types, `AgentSpec`, no server code in the library |
| Python / TypeScript SDKs | owned data, trait objects, serde types |
| Persistence | `ConversationStore` trait (add file and SQLite implementations) |
| OpenTelemetry GenAI traces | `tracing` spans plus hooks |
| OpenRouter, Azure OpenAI, vLLM (Chat Completions protocol) | private protocol layer plus auth layer |
| Bedrock, Vertex | private protocol layer plus auth layer |
| MCP OAuth | `rmcp`'s `auth` feature, behind an `mcp-oauth` feature |
| Orchestrator, shared budgets | `Agent` + `Conversation` + hooks |
| MCP server mode | the `Tool` trait |
| 1.0 stability | `#[non_exhaustive]`, semantic versioning, minimum Rust version, docs.rs |
