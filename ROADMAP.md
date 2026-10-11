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
    .mcp_stdio("npx", ["-y", "@modelcontextprotocol/server-filesystem", "."])
    .build().await?;

let mut convo = Conversation::new();
agent.run(&mut convo, "Review src/lib.rs").await?;
agent.with_model("local/Qwen/Qwen3-8B").run(&mut convo, "Summarize that").await?;   // swap, same chat
let invoice: Invoice = agent.extract(&pdf_text).await?;                             // typed output

// Level 3: implement Provider, Tool, Embedder, VectorStore or ConversationStore yourself
```

- `Agent` is settings only (model, system prompt, tools, knowledge, options). It holds no state, is cheap to clone, and can be shared across tasks and users.
- `Conversation` is plain data. It is serializable, so it can be saved and resumed.
- Multiple agents need no orchestrator: `agent.as_tool(..)` handles delegation, and running another agent on the same conversation is a handoff. Add an `Orchestrator` type only when a real need appears, such as a shared budget or tracing across agents.

## Architecture

```
Provider (trait): chat_stream(request) -> Stream<Event>, embed(texts), capabilities()
├── Local: mistral.rs (GGUF path, HF repo, HF GGUF)                 [feature: local]
└── Hosted: protocol + auth + base URL                              [feature: hosted, default]
    ├── Responses API     → OpenAI, Grok (xAI), Gloo AI
    └── Messages API      → Anthropic
Auth: API key | OAuth client credentials | Grok account login [experimental]  (token caching and refresh shared)
Batch: submit → handle (serializable) → results by request ID; hosted batch APIs, local runs concurrently
```

## Design rules that keep later work possible

These cost little now and keep the door open to an HTTP service, other-language SDKs, persistence and client production needs, without building any of them yet.

1. **Every public data type is serde-serializable:** `Message`, `Event`, `ToolCall`, `Usage`, `Conversation`, `RunOptions`.
2. **Agents can be defined as data.** An `AgentSpec` (model string, system prompt, tool names, knowledge, MCP servers) can be loaded from config, not only built in code.
3. **The public API uses owned data and trait objects** (`Arc<dyn Provider>`, `Arc<dyn Tool>`). Streams return owned `Event`s, not borrowed-lifetime callbacks.
4. **Public enums and option structs are `#[non_exhaustive]`**, so fields and variants can be added without breaking users.
5. **Extension points are traits:** `Provider`, `Tool`, `Embedder`, `VectorStore`, `ConversationStore`, and hooks.
6. **Protocol details stay private.** New providers (Bedrock, Vertex, Azure, …) can be added without API changes.
7. **The library has no server code.** A future service lives in `nucleus-cli` or its own crate.
8. **There is always an escape hatch:** `provider_options(json!(..))` sends extra fields, and raw provider responses are always accessible.
9. **Feature flags only add things.** The default build stays small.

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
- The library never writes to stdout or stderr; it logs only through `tracing`.

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

**Dogfooding**
- Before releasing 0.2, build one realistic client-style integration with Nucleus and fix whatever is awkward.

## Working agreements

- Work on a branch per phase or logical change, with one PR into `main` for each. Never commit directly to `main`.
- Tick checkboxes in this file in the same PR that completes them. When a spike or design note changes a decision, update this file in that PR.
- Design notes live in `docs/design/`. Spike findings live in `docs/design/spikes/`.
- Spike code lives in a throwaway `spikes/` directory on the spike branch and is never merged. Only the findings notes are merged.

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
- Batch requests are in 0.2, for cost savings.
- Built-in tools (`tools` module): `files()` (read, write, list directory, search) and `exec()` (shell command). Both go through the approval hook when one is set.

Open:
- [ ] Move mistral.rs from a git commit to a crates.io release (required before publishing; checked in the Phase 0.5 spike)
- [ ] Gloo auth: static API key or client credentials? (check the Gloo Studio account; the auth layer supports both)

## Phases

Every phase ends with CI green and meets the quality bar. Phases 0 → 0.5 → 1 → 2 run in order. Phases 3, 4 and 5 can run in any order or in parallel after Phase 2.

### Phase 0: Guardrails
- [x] CI runs `cargo build --all-targets`, `cargo test --workspace` and `cargo test --doc` on Linux and macOS, and `cargo clippy` on Linux (it reports warnings without failing until the end of Phase 1)
- [x] Fix or delete broken tests and examples: the registry test, the nucleus-std examples, and `register()` calls that are never awaited. Delete anything Phase 1 removes anyway rather than fixing it.
- **Done when:** CI is green on `main`.

### Phase 0.5: Spikes (settle the unknowns)
Short experiments in scratch code (see Working agreements). Each produces a findings note in `docs/design/spikes/` and may change the design.
No API keys and no paid calls. Hosted-provider spikes work from API references, OpenAPI specs, official SDK source and SDK test fixtures. A question that only a live call could answer is written down as open in the findings note and does not block the phase.
- [ ] mistral.rs (latest crates.io release): tool calling, streamed tool calls, embedding models, JSON-schema constrained output, Metal, sharing a loaded model, compile time and dependency count
- [ ] Gloo AI: auth method, and Responses API support for tools, streaming and structured output
- [ ] xAI and OpenAI Responses API: function-call streaming and reasoning items with `store: false`; whether the Grok account proxy behaves the same
- [ ] Anthropic Messages API: stream event shapes, thinking-block replay, prompt caching; choose the HTTP mock tool (e.g. `wiremock`)
- [ ] `rmcp`: maturity, transports, API stability
- [ ] Batch APIs: OpenAI and Anthropic shapes and limits; xAI per-model discounts and unsupported models; whether Gloo offers batch at all; what mistral.rs offers for concurrent local requests
- **Done when:** each spike has findings, and the decisions and design below are updated to match.

### Phase 1: Delete and consolidate
- [ ] Remove everything in the "Remove:" decision above, plus leftovers from the earlier project (Go entries in `.gitignore`, `llm-workspace` socket)
- [ ] Merge the crates into `nucleus`: `provider`, `agent`, `tool`, `rag`, `mcp`, `tools`
- [ ] Remove `Permission`. Tools become `Arc<dyn Tool>` with no `Mutex`, and registration is synchronous
- [ ] Feature flags: `hosted` (default), `local`, `rag`, `lancedb`, `mcp`, `experimental`, `metal`, `cuda`
- [ ] Choose the minimum supported Rust version and the edition (2021 or 2024)
- Existing behavior (`ChatManager`, Grok, mistral.rs behind `local`, RAG behind `rag`) keeps compiling and passing tests. The rewrites happen in later phases.
- **Done when:** the default build depends on little more than reqwest, serde and tokio, and the Phase 1 CI gates pass.

### Phase 2: Core
- [ ] **Design gate:** `docs/design/core-api.md` with the full public signatures (`Agent`, `Conversation`, `Message`, `Event`, `Tool`, `Provider`, `Error`, `RunOptions`, `Batch`, hooks), plus the Phase 6 example programs compiling against `todo!()` stubs. Reviewed and approved before implementation starts. The note must also settle these questions:
  - The model-string grammar, including `local/` paths that contain slashes, absolute paths, and `repo:file.gguf`
  - How `Usage` cost is calculated: a built-in price table that users can override, or token counts only
  - Whether slash commands (the current `/help`, `/reset`, `/exit` in `chat/commands.rs`) stay in the library or belong to apps
  - The shape of `ConversationStore` and `AgentSpec`
- [ ] Message model: tool calls have IDs; provider-specific data (thinking or reasoning items) is tagged with provider and model, replayed when they match and dropped otherwise
- [ ] The system prompt belongs to the agent and is added to each request, never saved in history
- [ ] `Agent`, `Conversation`, `RunOptions`, `with_model` / `with_system` / `with_tools`, `as_tool`
- [ ] `Tool` trait plus `tool_fn` with typed arguments (schemars)
- [ ] Approval hook: `.on_tool_call(|call| Approve | Deny)`; a denial goes back to the model as a tool result
- [ ] `AgentSpec` (serde) that builds an `Agent`; the `ConversationStore` trait (implementations come later)
- [ ] Built-in tools rebuilt on the new `Tool` trait (see Known issues for `ExecPlugin`)
- [ ] Tool loop: maximum step count, tool errors sent back to the model, independent calls run in parallel
- [ ] `Stream<Item = Event>` streaming. The library never prints; it uses only `tracing`
- [ ] `Usage` (tokens and cost) on every result; hooks for `on_request`, `on_response`, `on_tool_call`
- [ ] `nucleus::complete()` one-shot and `agent.extract::<T>()` typed output
- [ ] Public `MockProvider` for tests (ours and clients')
- **Done when:** the loop, swaps, handoff, agent-as-tool and save/resume are covered by mock-provider tests.

### Phase 3: Providers
- [ ] Auth layer: API key, OAuth client credentials, and Grok device-code login (moved from `grok.rs`, behind `experimental`)
- [ ] Responses protocol (built from the existing Grok code), configured for OpenAI, Grok and Gloo
- [ ] Messages protocol for Anthropic, over raw HTTP:
  - streaming with text, thinking and `input_json_delta`
  - `tool_use` / `tool_result`
  - adaptive thinking and effort
  - automatic prompt caching
  - `refusal` stop reason
- [ ] Local provider (mistral.rs):
  - use the request's tools, temperature and history
  - configurable timeouts
  - load the model once and share it
  - `Local::builder()` for quantization, device and context size
- [ ] Reliability: typed retryable errors, retries that respect `retry-after`, timeouts, cancellation, fallback chains
- [ ] Image and PDF input where the provider supports it
- [ ] Batch requests:
  - `nucleus::batch(model, requests)` returns a serializable `BatchHandle`, so a job can be resumed after a restart; results come back keyed by request ID
  - backed by OpenAI's Batch API, Anthropic's Message Batches and xAI's Batch API
  - local runs the requests concurrently through mistral.rs
  - providers without batch support (possibly Gloo) give a clear error, or run as limited concurrent normal requests if explicitly asked
  - covers one-shot and `extract` requests; multi-step tool loops are out of scope for batch
- [ ] Shared test suite for every provider: fixture replays against a local mock server in CI, and a tiny GGUF model for local
- **Done when:** one example runs unchanged against `local/` and the mocked `openai/`, `anthropic/`, `grok/` and `gloo/` endpoints, changing only the model string, and a batch example runs against the mocked OpenAI and Anthropic batch APIs and local.

### Phase 4: RAG
- [ ] An `Embedder` trait separate from chat; local mistral.rs embeddings by default; OpenAI and Gloo embeddings optional
- [ ] A built-in vector store that saves to a file and compares every vector (the default); LanceDB behind a feature
- [ ] `.knowledge(path)`: index on first use, then re-index only files whose content hash changed
- [ ] File filtering matches whole path components, honours `.gitignore`, and skips binaries
- [ ] Chunk on paragraphs and lines with a size limit; record source file and line range for citations
- [ ] Retrieved text is either added to every request or exposed as a `search_knowledge` tool
- **Done when:** RAG works in three lines with zero setup and is tested.

### Phase 5: MCP
- [ ] `rmcp` client (stdio and streamable HTTP) behind the `mcp` feature
- [ ] MCP tools become `Tool`s, prefixed by server name; the approval hook applies to them
- [ ] Server lifecycle: start, list tools, reconnect, shut down on drop
- **Done when:** an example uses a reference MCP server, with an integration test against a test server.

### Phase 6: Docs and examples
- [ ] README built around levels 1–3
- [ ] Examples built in CI: one-shot, batch, agent, local vs. hosted, custom tool, typed extract, RAG, MCP, streaming, swap and handoff
- [ ] Cut the mdbook down to match the code; its code blocks become compiled doctests
- [ ] Changelog, release 0.2

## Known issues in code being kept or reworked

Found in an audit on 2026-10-09. Most of the codebase is being deleted or rewritten. These are the issues in code that survives into the new design or serves as its starting point:

- `nucleus-core/src/provider/grok.rs` (becomes the Responses protocol): tool definitions are sent but function-call events are never parsed (`parse_stream_event`), and `to_grok_input` drops `tool_calls` and tool results. Reasoning text is printed with `eprint!` instead of being emitted as an event.
- `nucleus-core/src/patterns.rs`: `should_exclude` matches patterns as substrings of the full path, so `out` excludes `layout.rs`, `data` excludes `metadata.rs`, `models` excludes `src/models/`, and glob patterns like `*.egg-info` never match. It affects RAG indexing and the search tool.
- `nucleus-std/src/commands.rs` (`ExecPlugin`): runs `Command::new(command)` with no shell, so "git status" fails. The `args` passed by its test and `run()` helper are silently dropped.
- `nucleus-core/src/provider/mistralrs.rs`: ignores `request.tools` (builds its own from the registry) and `temperature`; drops `tool_calls` from earlier assistant messages; hard-coded 30s per-chunk and 60s stream-start timeouts.
- `nucleus-core/src/rag/`: `RagEngine::new` unwraps `config.rag` (panics when RAG isn't configured); `clear_knowledge_base` never awaits the clear; LanceDB only appends, so re-indexing duplicates chunks; the default `chunk_size` is set to `embedding_dim`.
- `nucleus-core/src/chat/manager.rs` (the reference for the new agent loop): no step limit, tool errors end the turn, retrieved RAG text is saved into history, and the system prompt is stored as a history message. Worth keeping: the history rollback on failure (`run_turn`) and `query_with`.
- All three YAML configs (`config.yaml`, `examples/config.yaml`, `nucleus-core/config.yaml`) fail to parse, and `Config::load_or_default` silently falls back to defaults.
- Only CI is `.github/workflows/metrics.yml`; nothing builds or tests the code.

## Later (the path exists, not scheduled)

| Need | Already supported by |
|---|---|
| HTTP service, OpenAI-compatible endpoint | serde types, `AgentSpec`, no server code in the library |
| Python / TypeScript SDKs | owned data, trait objects, serde types |
| Persistence | `ConversationStore` trait (add file and SQLite implementations) |
| OpenTelemetry GenAI traces | `tracing` spans plus hooks |
| OpenRouter, Azure OpenAI, vLLM (Chat Completions protocol) | private protocol layer plus auth layer |
| Bedrock, Vertex | private protocol layer plus auth layer |
| Orchestrator, shared budgets | `Agent` + `Conversation` + hooks |
| MCP server mode | the `Tool` trait |
| 1.0 stability | `#[non_exhaustive]`, semantic versioning, minimum Rust version, docs.rs |
