# Changelog

All notable changes to this project are documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
0.2 is a rewrite; see [ROADMAP.md](ROADMAP.md) for the plan.

## [Unreleased]

### Added
- CI workflow (`.github/workflows/ci.yml`): build, test and doctest on Linux and macOS.
- Feature flags: `hosted` (default), `std` (default), `local`, `rag`, `lancedb`, `mcp` (empty until Phase 5), `experimental`, `metal`, and `cuda`.
- Phase 1 CI gates: clippy `-D warnings`, rustfmt, cargo-deny, an MSRV check, and `cargo hack --each-feature` (excluding `metal` and `cuda`).
- Phase 2 message model and conversation storage (`src/message.rs`, `src/conversation.rs`, `src/store.rs`): the ordered `Part` list with raw-JSON tool-call input and tagged `ProviderItem`s, a stable `ConversationId` with append-only history and serde save/resume, and an always-compiled in-memory `MemoryStore` for tests.
- `Agent` and `AgentBuilder`, and `Agent::run`: one turn against a `Provider`, including the tool loop, the approval hook, and a save through `ConversationStore` when one is set.
- `examples/agent.rs`: runs that loop against a scripted provider, with no API key. `ToolCallId::new` and `Capabilities::streaming` are the constructors a caller needs to implement `Provider` outside the crate.
- Phase 2 error, usage and options types: the `#[non_exhaustive]` `nucleus::Error` with a `ProviderError`/`RequestId` on every provider failure, `Usage`/`Cost` with cache-read and cache-write tokens and an overridable `PriceTable`/`ModelPrice` for cost from a price table, `StopReason`, and the `RunOptions` set (`Effort`, `Thinking`, `ThinkingDisplay`, `CacheTtl`, `provider_options`/`provider_headers`).

### Changed
- Approve the Phase 2 design note (`docs/design/core-api.md`), settling its open questions (model-string grammar, Anthropic prefix changes, `Usage` cost, `MemoryStore`, batch shape, `AgentSpec` tool resolution, hook naming, error/event serde depth and the `local/` thinking default).
- Merge `nucleus-core`, `nucleus-plugin`, and `nucleus-std` into the `nucleus` crate. Built-in tools stay behind the `std` feature.
- Minimum supported Rust version is 1.94. Edition stays 2021.
- reqwest 0.13, with rustls. Grok account login is behind `experimental`.
- mistral.rs is the v0.9.4 git tag, behind `local`. The default build no longer depends on it or on LanceDB.

### Removed
- Slash commands (`/help`, `/reset`, `/exit`) leave the library and belong to apps; the reset primitive is `Conversation::clear()`.
- The plugin `Permission` system. Registration is synchronous and always succeeds. An approval hook comes later.
- CoreML (archived on `archive/coreml`), the IPC server, Ollama, Qdrant, `ModelRegistry`, `nucleus-dev`, and the CLI stubs other than `login` and `logout`.
- `MistralRsProvider`, the `"mistralrs"` arm of `create_provider`, and `ProviderType`. Local models load through a `local//path.gguf` model string. The legacy config's default provider is now `"grok"`.
- `Provider::embed` and the `EmbedRequest`/`EmbedResponse` types. Embeddings get their own `Embedder` trait in Phase 4 and are not part of any provider.
- Unused config: CoreML feature names, personalization, chat-history and tool-state paths, and the Qdrant gRPC storage mode.

### Fixed
- Remove the CoreML example and switch the `with_provider` doctest off `ProviderType::CoreML`; that variant is macOS-only, so both broke the Linux CI build.
- Build on Rust 1.99: update `ethnum` (pulled in by LanceDB) from 1.5.2 to 1.5.3. 1.5.2 transmutes a standard-library error type that changed size.
- Examples that called the async `PluginRegistry::register` without awaiting it, so no plugin was registered (`read_file_line`, `write_file`, and the `nucleus-std` `read_file` and `search_text` examples, which also failed to compile).
- Registry tests that never registered a plugin: one always failed, the other passed without testing anything.
- Doc examples in `nucleus-core` that no longer compiled against the current `ChatManagerBuilder`, `RagEngine` and `MistralRsProvider` APIs.
