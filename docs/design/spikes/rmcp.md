# Spike: `rmcp` (the official Rust MCP SDK)

Date: 2026-10-11. No API keys were used and no paid calls were made. Only local reference servers were run.

Versions and sources checked:
- `rmcp` 3.5.1 (released 2026-10-05, the latest), read from the crates.io source tarball: `CHANGELOG.md`, `Cargo.toml`, `src/service.rs`, `src/service/client.rs`, `src/handler/client.rs`, `src/transport/child_process.rs`, `src/transport/async_rw.rs`, `src/transport/streamable_http_client.rs`, `src/transport/common/client_side_sse.rs`, `src/transport/auth.rs`, `src/model/tool.rs`, `src/model.rs`, `src/handler/server/tool_name_validation.rs`.
- `process-wrap` 10.0.1 (the process library that rmcp's stdio transport uses).
- GitHub [`modelcontextprotocol/rust-sdk`](https://github.com/modelcontextprotocol/rust-sdk) at `main` on 2026-10-11: `VERSIONING.md`, `ROADMAP.md`, `LICENSE`, the open issue list and contributor stats (via `gh`).
- crates.io API (versions, downloads, reverse dependencies) and docs.rs (`rmcp/3.5.1` builds; `doc_status: true`).
- `@modelcontextprotocol/server-everything` 1.0.0 from npm (built on `@modelcontextprotocol/sdk` ^1.31.0), run with node v26.0.0 and npx.
- `openai/openai-python` `main` (fetched 2026-10-11): `types/shared_params/function_definition.py`, `types/responses/function_tool_param.py`. The Anthropic tool-name rule comes from [anthropic-messages.md](anthropic-messages.md).
- Spike code: `spikes/rmcp/` (a standalone cargo project with an empty `[workspace]` table; rmcp 3.5.1 pinned, Rust 1.94.0, macOS aarch64). Its outputs are `out-*.txt` and `tools-*.json` in the same directory, and they are quoted below as "spike".

How each claim was established: **[verified]** means run in the spike or read directly in the source file named; **[docs]** means read in docs, changelog, issues or SDK docstrings; **[inferred]** means my conclusion from those.

## Questions

From the roadmap (Phase 0.5 "`rmcp`: maturity, transports, API stability" and Phase 5), as assigned:
1. Maturity: latest version, release cadence, breaking-change frequency, maintainers, MSRV, edition, license, downloads, open issue themes.
2. API stability, and whether nucleus should wrap rmcp completely.
3. Transports (stdio, streamable HTTP, legacy SSE): feature flags, dependency footprint, OAuth.
4. Client API: connect, initialize, list tools (pagination, `tools/list_changed`), call tool, structured output, annotations, cancellation, progress, timeouts.
5. Lifecycle: drop, graceful shutdown, reconnecting after a crash or disconnect.
6. Mapping MCP tool schemas onto a nucleus `Tool`.
7. Alternatives to rmcp.

Found along the way:
8. Closing a stdio server started through `npx` kills only `npm exec`, not the node server under it, unless the child gets its own process group.
9. The child server inherits the parent's whole environment, API keys included, and by default its stderr goes to the parent's stderr.
10. rmcp's `reqwest` is 0.13. The current nucleus workspace uses 0.12, so with `mcp` turned on both would compile.
11. The connection is a live resource (a child process or an HTTP session). This conflicts with the roadmap rule that "`Agent` is settings only… holds no state".

## Findings

### 1. Maturity

| Item | Value | Source |
|---|---|---|
| Latest | 3.5.1, 2026-10-05 | crates.io API [verified] |
| Repo / org | `modelcontextprotocol/rust-sdk`, the official SDK under the MCP GitHub org. CODEOWNERS is `@modelcontextprotocol/rust-sdk` | GitHub [verified] |
| Most active contributors | DaleSeo (150 commits), alexhancock (43), 4t145 (42), jokemanfire (42), jamadeo (28), plus dependabot and release bots | GitHub contributors API [verified] |
| Stars / forks / open issues | ~4.0k / 659 / 65 | GitHub API [verified] |
| Downloads | 34.6M total, 18.9M in the last 90 days, 2,189 reverse dependencies | crates.io API [verified] |
| MSRV | `rust-version = "1.88"` (declared since 3.0.0-beta.2) | `Cargo.toml` [verified]; CHANGELOG 3.0.0-beta.2 "declare and check MSRV" [docs] |
| Edition | 2024 | `Cargo.toml` [verified] |
| License | `Apache-2.0` on crates.io since 0.13.0. 0.12.0 and earlier were MIT. The repo LICENSE says MCP is moving from MIT to Apache-2.0 and that some contributions remain MIT until relicensed. Open issue #1335: the crate tarball no longer includes a LICENSE file (a regression since 1.3.0) | crates.io API, repo `LICENSE` [verified]; issue #1335 [docs] |
| Conformance | The SDK says it meets SEP-1730 Tier 1, with 100% server and client conformance on the 2025-11-25 and 2026-07-28 suites | repo `ROADMAP.md` [docs] |
| Release tooling | release-plz with conventional commits and `cargo-semver-checks` | repo `VERSIONING.md` [docs] |

**Cadence and breaking changes** [verified from crates.io dates and CHANGELOG]:
- 2026 alone had three majors: **1.0.0 on 2026-03-03, 2.0.0 on 2026-06-27, 3.0.0 on 2026-07-28**. Before 1.0, the crate went from 0.1 to 0.17 in about 12 months, and every 0.x minor could break.
- Majors follow MCP spec revisions. 2.0 was "align model types with MCP 2025-11-25 spec" and "relax tool result structuredContent type". 3.0 implemented the 2026-07-28 spec, with 11 `[breaking]` entries in 3.0.0-beta.1 alone: MRTR (SEP-2322), cache hints (SEP-2549), server discovery and negotiation (SEP-2575), `Annotations.lastModified` as a string, `structuredContent` type, HTTP standard headers.
- Minors and patches come about weekly: 3.x has had 13 releases in 10 weeks, from 3.0.0 to 3.5.1, with no breaking entries since 3.0.0.
- The types a client touches (`Tool`, `CallToolResult`, `ContentBlock`, `Annotations`) changed in both 2.0 and 3.0. Since 1.0 (#715), most public structs and enums are `#[non_exhaustive]`, which makes additions non-breaking [docs].
- **Expect a new rmcp major with each MCP spec revision, roughly every 4–8 months** [inferred].

**Open issue themes** (65 open; the list was read on 2026-10-11) [docs]:
- Streamable HTTP and OAuth edge cases: #1339, #1330, #1298, #1265, #1203 (redirects are not followed).
- Protocol details of 2026-07-28: #1337, #1321, #1108.
- Two that matter for us:
  - **#1338**: abandoning a `call_tool` future sends no `notifications/cancelled`, because the MRTR helper hides the `RequestHandle`.
  - **#1030**: the stdio line buffer is unbounded, so a malicious server can exhaust our memory.
- Also open: feature-flag cleanup for v3 (#1067) and API naming drift (#1155).
- There are no P0s open. The issues are triaged, labeled and prioritized.

### 2. API stability, and should nucleus wrap it?

- rmcp is well run (semver, semver-checks, Tier 1 conformance), but its public types are the MCP wire schema. Each spec revision becomes a major that changes those types [verified, see §1].
- The spec itself is still moving fast. 2026-07-28 removed `initialize` for new sessions (`ProtocolVersion::LATEST = 2026-07-28`, `LATEST_WITH_INITIALIZE = 2025-11-25`, in `src/model.rs`), added stateless HTTP, and replaced server-to-client requests with MRTR rounds [verified in source; docs in CHANGELOG].
- rmcp also leaks third-party types into its API:
  - `TokioChildProcess::builder` takes `impl Into<process_wrap::tokio::CommandWrap>` (process-wrap 10, which is not re-exported) [verified].
  - The HTTP transport is generic over `reqwest::Client` 0.13 [verified].

**Recommendation: wrap completely. No rmcp types appear in nucleus's public API** [inferred]. Design rule 6 ("protocol details stay private") already implies this, and the churn above makes it necessary: if rmcp types were public, every rmcp major would also be a nucleus major. With a full wrapper, a new rmcp major costs us a private port, which should be small because we use a narrow slice: list tools, call tool, two notifications and shutdown. The nucleus-facing surface should be:
- A serde `McpServerSpec` (name plus `Stdio { command, args, env, cwd }` or `Http { url, headers, bearer }`), so it fits `AgentSpec` (design rule 2).
- MCP tools surfaced as `Arc<dyn Tool>`.
- Errors mapped to `nucleus::Error` kinds.
- An escape hatch that gives raw JSON for tool results (`CallToolResult` serializes to the wire JSON) [inferred].

The spike compiled on the first try against 3.5.1 with the API used below [verified], so the slice we need is straightforward.

### 3. Transports, feature flags, footprint, OAuth

| Transport | rmcp feature | Notes |
|---|---|---|
| stdio child process | `client` + `transport-child-process` (+ `which-command` to resolve `npx.cmd` on Windows) | Built on `tokio::process` and `process-wrap` 10 [verified, `Cargo.toml`] |
| Streamable HTTP client | `client` + `transport-streamable-http-client-reqwest` + a TLS choice (`reqwest` = rustls, `reqwest-native-tls`, or `reqwest-tls-no-provider`) | A custom HTTP client is possible through the `StreamableHttpClient` trait. There is also a Unix-socket variant [verified] |
| Legacy SSE (HTTP+SSE, 2024-11-05) | **none**: removed in 0.11.0, "[breaking] remove SSE transport support (#562)" | No client exists in 3.5.1 [verified: CHANGELOG line for 0.11.0; no SSE transport in `src/transport/`]. If we ever need it, we write it ourselves |
| WebSocket | none (`ws.rs` starts with "Maybe we don't really need a ws implementation?") | [verified] |

**Do not use the default features** (`server`, `macros`, `base64`). They pull in the server side, schemars and the proc macros. Use `default-features = false` [verified, `Cargo.toml`].

**Footprint** [verified, spike `footprint/`]. Counts are unique crates in `cargo tree -e normal --target aarch64-apple-darwin`, added on top of a nucleus-like base of tokio + serde_json + reqwest 0.13 (rustls). The base alone is **90** crates.

| rmcp features (default-features = false) | Total | Added |
|---|---|---|
| `client` | 106 | +16 |
| `client`, `transport-child-process` | 111 | +21 |
| `client`, `transport-streamable-http-client-reqwest` | 107 | +17 |
| stdio + streamable HTTP (`reqwest`) | 112 | **+22** |
| stdio + HTTP + `auth` (OAuth) | 132 | +42 |

- What stdio + HTTP adds: chrono, futures (+executor/io/macro), indexmap, hashbrown, nix, process-wrap, signal-hook-registry, sse-stream, tokio-stream, thiserror, tracing-attributes, num-traits, errno, cfg-if, serde_derive.
- What `auth` adds on top: oauth2 5, sha2, digest, rand 0.8 (an old major), async-trait, serde_path_to_error, zerocopy, and others.
- Compile time: the `rmcp` crate alone takes about 3.5s in a debug build. The whole client+stdio+HTTP tree builds from clean in 11.8s debug on this machine [verified].
- **reqwest:** rmcp needs 0.13.2 and the nucleus workspace pins 0.12, so turning on `mcp` today would compile two reqwests [verified, both `Cargo.toml`s].

**OAuth for HTTP servers** [verified, `src/transport/auth.rs`, 9.7k lines; docs in CHANGELOG]:
- Supported behind the `auth` feature: authorization code + PKCE (S256 required), RFC 9728 protected-resource-metadata discovery, dynamic client registration, pre-registered clients, client credentials (including `private_key_jwt` under `auth-client-credentials-jwt`), refresh, step-up scopes, and pluggable `CredentialStore`/`StateStore`.
- The authorization-code flow needs a browser redirect handled by the app.
- Static tokens work without the `auth` feature: `StreamableHttpClientTransportConfig::auth_header(token)` sends `Authorization: Bearer …`, and `custom_headers` sends anything else [verified].

### 4. Client API

The spike did all of the following against server-everything 1.0.0 over both stdio and streamable HTTP [verified, `out-stdio.txt`, `out-http.txt`]:

- **Connect and initialize:** `handler.serve(transport).await` does the handshake and returns `RunningService<RoleClient, H>`, which derefs to `Peer<RoleClient>`. The negotiated protocol was `2025-11-25`, because the JS server does not speak 2026-07-28. Connecting over stdio took 0.8–2.1s (mostly npx startup); over HTTP it took 18ms. `peer_info()` returns server name, version, capabilities and `instructions`.
- **Client handler:** `()` works for a plain client. A custom `impl ClientHandler` overrides `on_tool_list_changed`, `on_progress`, `on_logging_message`, `on_cancelled` and others. Sampling, elicitation and roots are also handler methods; we leave them at their defaults (method-not-found) [verified, `src/handler/client.rs`].
- **List tools:**
  - `list_all_tools()` loops on `next_cursor` [verified, `src/service/client.rs`]. The server returned one page of 13 tools, so pagination was not exercised live.
  - `list_tools` has a client-side TTL cache (SEP-2549). On error it falls back to the stale cache entry. rmcp clears the cache when `tools/list_changed` arrives [verified in source].
- **`tools/list_changed`:**
  - Over stdio, one notification arrived right after init, because server-everything registers its capability-dependent tools after `initialized` [verified].
  - Over HTTP, none arrived [verified]. My guess is that the server sent it before the client opened its standalone GET SSE stream [inferred].
  - Either way, nucleus must re-list tools when the notification arrives and should not trust a snapshot taken at connect time.
- **Call tool:**
  - `call_tool(CallToolRequestParams::new("echo").with_arguments(map))` worked, as did `get-sum` and `get-structured-content` [verified].
  - **Bad arguments and unknown tools come back as `Ok(CallToolResult { is_error: true, content: [text "MCP error -32602: …"] })`, not as protocol errors**, from this server [verified]. Other servers may return JSON-RPC errors (`ServiceError::McpError`) instead, so we need to handle both [inferred].
- **Structured output:** `CallToolResult.structured_content: Option<serde_json::Value>`, with a JSON text copy in `content`. The tool declares `output_schema` [verified]. Since 3.0, `structured_content` can be any JSON value, including explicit `null` [docs, CHANGELOG 3.0.0-beta.1, 3.5.0].
- **Annotations:** `ToolAnnotations { title, read_only_hint, destructive_hint, idempotent_hint, open_world_hint }`, plus `title`, `icons` and `_meta` on `Tool`. The spec and rmcp's own docs say these are **hints from untrusted servers** [verified, `src/model/tool.rs`; spike shows them per tool].
- **Timeouts:** **There is no default request timeout.** `send_request` uses `PeerRequestOptions::no_options()` [verified, `src/service.rs`]. Per-request timeouts go through `send_request_with_option(ClientRequest::CallToolRequest(..), PeerRequestOptions::with_timeout(d))`, which also offers `reset_timeout_on_progress()` and `with_max_total_timeout()`. On timeout, rmcp sends `notifications/cancelled` and returns `ServiceError::Timeout`. The spike saw a 700ms timeout fire at 701ms, and the connection stayed usable [verified].
- **Cancellation:** `send_cancellable_request` returns a `RequestHandle`, and `handle.cancel(reason)` sends `notifications/cancelled`; the spike got `Ok(())` and could still call tools afterwards [verified]. Dropping a plain `call_tool` future does **not** notify the server (#1338) [docs]. So nucleus should use the `RequestHandle` path for every call. That path skips the MRTR helper, but MRTR only matters for servers that ask the client for input, which we don't support in 0.2 [inferred].
- **Progress:** rmcp **always overwrites `_meta.progressToken`** with its own numeric token (`set_progress_token` runs after the caller's meta is merged) [verified, `src/service.rs` `send_request_with_option_and_subscription`]. Notifications arrive at the global `ClientHandler::on_progress`. To route them to a call, match on `RequestHandle::progress_token`. The spike received 4 of 4 progress notifications for a 4-step operation [verified].
- **Errors:** `ServiceError` (`#[non_exhaustive]`) has the variants `McpError`, `TransportSend`, `TransportClosed`, `UnexpectedResponse`, `Cancelled`, `Timeout`, `SubscriptionLagged` and `InputRequiredRoundsExceeded` [verified]. These map cleanly onto nucleus error kinds:
  - `TransportClosed` and `TransportSend` → retryable/transport.
  - `Timeout` → timeout.
  - `McpError` and `is_error` results → tool error that goes back to the model.

### 5. Lifecycle

Mechanics [verified, source]:
- Dropping `RunningService` fires a `DropGuard` that cancels the service task. The task then drains in-flight responses (for at most 2s on cancel) and calls `transport.close()`.
- For stdio, `close()` is `TokioChildProcess::graceful_shutdown`: drop the child's stdin, wait up to **3s** (the hard-coded `MAX_WAIT_ON_DROP_SECS`), then `kill()` the direct child.
- If the transport itself is dropped without being closed, `ChildWithCleanup::drop` calls `tokio::spawn` to kill the child. Outside a Tokio runtime that call would panic [inferred from `tokio::spawn` semantics; not run].
- `RunningService::close()`, `close_with_timeout(d)` and `cancel()` give explicit async shutdown.

Observed [verified, spike]:

| Scenario | Result |
|---|---|
| Idle server, drop the client (`out-drop.txt`) | `npm exec` and its `node` child are both gone within 1s. The server exits on stdin EOF |
| Idle server, `close()` with a process group (`out-close-pg.txt`) | Returns in 46ms. Both processes are gone |
| **Busy server** (6s tool call in flight), `close()` (`out-close-busy.txt`) | `close()` blocks for **3.0s**, then kills `npm exec`. **The `node` server survives as an orphan** until its work finishes |
| Busy server, `close()` with `process_wrap::tokio::ProcessGroup::leader()` | Blocks 3.0s, then the **whole tree is gone** |
| Drop just before `main` returns (`out-drop-at-exit.txt`) | Both processes are gone. Pipes close at process exit and the server exits on EOF |
| `kill -9` of the server mid-session (`out-kill.txt`) | The next call fails at once (1ms) with `TransportClosed`. `peer().is_transport_closed()` is `true` but `is_closed()` stays `false`. `waiting()` returns `Closed` |
| HTTP server killed mid-session (`out-http-kill.txt`) | The next call fails at once with `TransportSend(… ConnectionRefused)`. POSTs are not retried |
| HTTP server restarted on the same port, same client | The next call **succeeds** (22ms). `reinit_on_expired_session` (default `true`) answers the session 404 by running `initialize` again and retrying once |

Conclusions:
- **Stdio reconnect is ours to write.** rmcp gives a fast `TransportClosed`, and nucleus decides whether to respawn the server, re-list tools and retry, with backoff and a cap.
- **HTTP session re-init is built in.** Our layer needs only connection-level retry with backoff. The SSE reconnect default is `ExponentialBackoff` with no maximum retry count and no maximum delay (`client_side_sse.rs`) [verified], so nucleus should set a cap.
- **Spawn stdio servers in their own process group** (and a Job Object on Windows, untested), or `npx`/`uvx`-launched servers leak on shutdown.

Also found [verified, spike `out-env.txt` and `child_process.rs`]:
- The child inherits our whole environment. The server's `get-env` tool saw 89 variables, including a test `SPIKE_FAKE_API_KEY` we set.
- `TokioChildProcessBuilder` defaults to `stderr = Stdio::inherit()`. That breaks nucleus's rule that the library never writes to stdout or stderr: the server's logs would land on the app's stderr. nucleus should pipe stderr into `tracing`.

### 6. Mapping MCP tools onto a nucleus `Tool`

Observed schemas, saved in `tools-stdio.json` [verified]:
- Every `inputSchema` has `"$schema": "http://json-schema.org/draft-07/schema#"` and `"type": "object"`.
- Keywords seen: `properties`, `required`, `description`, `default`, `enum`, `minimum`, `maximum`, `format`.
- Four tools have an empty `properties`.
- None has `additionalProperties: false`, and several have optional properties. Only the `outputSchema` sets `additionalProperties: false`.
- Real servers (for example pydantic-based ones) also emit `$defs`/`$ref`/`anyOf` [inferred; not seen in this server].

Name rules:

| | Allowed characters | Max length | Source |
|---|---|---|---|
| MCP | `A-Za-z0-9_-.` (SHOULD) | 128 | `tool_name_validation.rs` [verified] |
| OpenAI | `a-zA-Z0-9_-` | **64** | openai-python `function_definition.py` [docs] |
| Anthropic | `^[a-zA-Z0-9_-]{1,128}$` | 128 | [anthropic-messages.md](anthropic-messages.md) [docs] |

Server names can contain anything; server-everything calls itself `mcp-servers/everything` [verified].

Recommended mapping [inferred]:
- **Name:** `{server}__{tool}`. Replace any character outside `[A-Za-z0-9_-]`, including `.` and `/`, with `_`. If the name exceeds 64 characters, truncate it and add a short hash. Keep a map from the exposed name back to `(server, original tool name)`, because the original name is what goes into `tools/call`. The server name should be given by the user; fall back to a sanitized `serverInfo.name`.
- **Schema:**
  - Pass the input schema through mostly unchanged.
  - Strip `$schema`.
  - Make sure `type: object` and a `properties` object are present (fill in `{}`).
  - **Never mark MCP tools `strict`.** They fail the strict-mode rules (all properties required, `additionalProperties: false`) on both OpenAI and Anthropic.
  - The OpenAI Responses `FunctionToolParam.strict` field is `Required[Optional[bool]]` [docs, openai-python], so send `strict: false` explicitly. Whether OpenAI rejects `$schema`, `format` or `default` in non-strict mode needs a live call (see Open questions).
- **Description:** `title` and `description`. Keep `annotations` as metadata on the nucleus `Tool`, and pass `destructive_hint` and `read_only_hint` to the approval hook as context. Never use them to skip approval, because they are untrusted.
- **Result:**
  - Text content blocks → the tool-result text.
  - `structured_content` → JSON, when present.
  - Image blocks → image tool results where the provider supports them.
  - Resource links and embedded resources → text with the URI.
  - `is_error: true` → a tool error sent back to the model, not a nucleus `Err`.

### 7. Alternatives

Download counts are from crates.io on 2026-10-11 [verified].

| Crate | Latest | Recent (90-day) downloads | Notes |
|---|---|---|---|
| **`rmcp`** | 3.5.1 (2026-10-05) | **18.9M** | Official, Tier 1, Apache-2.0 |
| `rust-mcp-sdk` | 2.0.0 (2026-08-27) | 118k | Community (rust-mcp-stack), MIT |
| `pmcp` | 2.22.7 (2026-10-04) | 54k | Community (paiml), MIT |
| `turbomcp` | 3.6.0 (2026-10-01) | 9k | Community, MIT |
| `mcp-core`, `async-mcp`, `mcpr`, `mcp-sdk-rs`, `ultrafast-mcp` | last released in 2025 | ≤7k | Inactive |
| `mcp-client` 0.1.0 | 2025-02 | — | The old prototype in the same repo, superseded by rmcp |

- None of the alternatives is a better fit. They have a fraction of the users, are maintained by a single org or person, and are not tracked by the spec's conformance suite [inferred].
- **Writing our own client** is plausible for stdio only, since newline-delimited JSON-RPC with list and call is a few hundred lines. But streamable HTTP sessions and re-init, SSE resume, the 2026-07-28 stateless lifecycle and OAuth are where the real work is, and rmcp already passes the conformance suites for them [inferred].
- **Conclusion: use rmcp, fully wrapped.**

## Impact on the design

Concrete changes to propose for ROADMAP.md (not edited by this spike):

1. **Decisions:**
   - Confirm rmcp as the MCP client. Depend on `rmcp = { version = "3.5", default-features = false, features = ["client", "transport-child-process", "transport-streamable-http-client-reqwest", "reqwest"] }`, plus `process-wrap = "10"` for process groups.
   - Add `which-command` for Windows.
   - Legacy SSE is out of scope, because rmcp dropped it.
   - OAuth (the rmcp `auth` feature, +20 crates) is deferred past 0.2 or goes behind its own `mcp-oauth` feature. 0.2 supports static bearer tokens and custom headers on HTTP servers, and tokens from nucleus's own auth layer can be passed through as the bearer header.
2. **Phase 1:**
   - Move nucleus to **reqwest 0.13**, so `mcp` doesn't compile a second reqwest.
   - The MSRV must be **≥ 1.88** if the feature-combination check builds `mcp` (rmcp's MSRV), and edition 2024 works.
3. **Target API:**
   - `.mcp_stdio("npx", [...])` has no server name to prefix tool names with. Change it to `.mcp(McpServer::stdio("fs", "npx", [...]))` and `.mcp(McpServer::http("search", url).bearer(..))`, where `McpServer` is a serde spec usable from `AgentSpec`.
   - Alternatively keep `.mcp_stdio` and add `.mcp_stdio_named(..)`, falling back to the sanitized `serverInfo.name`.
4. **`Agent` statelessness:**
   - An MCP connection is a live child process or session, so the Agent rules need one explicit exception: an `Agent` built with MCP servers holds `Arc` handles to them. Clones share the connection, and dropping the last clone shuts the servers down.
   - `AgentSpec` keeps only the data (`McpServerSpec`) and connects at `build().await`.
   - Write this into the Phase 2 core-api note, even though MCP itself is Phase 5.
5. **Phase 5 checklist additions:**
   - No rmcp types in the public API; map errors to `nucleus::Error`; snapshot the API to enforce it.
   - Spawn stdio servers in their own process group, so shutdown kills `npx`/`uvx` grandchildren.
   - Pipe server stderr into `tracing`.
   - Configurable env: inherit by default but allow `env_clear` and an allow-list, and document that servers see API keys.
   - Async `shutdown()` with a bounded wait, plus drop as a fallback.
   - A default per-call timeout (rmcp has none), with progress-reset.
   - Calls go through `RequestHandle`, so a timeout or abort sends `notifications/cancelled`.
   - Stdio reconnect is ours: on `TransportClosed`, respawn, re-list and retry once, with backoff and a cap. HTTP relies on rmcp's session re-init plus our capped backoff.
   - Re-list tools on `tools/list_changed`.
   - Name sanitizing and prefixing with a 64-character limit and a reverse map.
   - Schema normalization: strip `$schema`, never `strict`.
   - Annotations go to the approval hook as untrusted context.
   - `is_error` and `structured_content` mapping.
6. **Testing (Phase 5 "integration test against a test server"):**
   - Prefer a tiny Rust MCP test server built with rmcp's `server` + `transport-io` features as a dev-dependency or test binary, so CI needs no node.
   - Keep an optional, ignored-by-default test against `npx @modelcontextprotocol/server-everything` for stdio and HTTP. The spike's scenarios (drop, close while busy, kill -9, HTTP restart) make good test cases.
7. **Upgrades:** expect an rmcp major with each MCP spec revision. Pin to a caret range within one major and do upgrades as deliberate PRs (the `cargo-deny` and public-API gates already make this visible).

## Open questions

- **Windows:** `npx.cmd` resolution (`which-command`), and whether process-wrap's `JobObject` kills the tree the way `ProcessGroup` does on Unix. Untested.
- **A server on the 2026-07-28 protocol** (stateless, no `initialize`, `server/discover`): server-everything negotiated 2025-11-25, so rmcp's modern lifecycle path was not exercised.
- **`tools/list_changed` over streamable HTTP:** no notification arrived in the spike. Is this a server-side timing issue or a client gap? It needs a server that sends the notification later in the session.
- **Pagination against a server that actually pages:** not exercised.
- **OpenAI:** does non-strict mode accept `$schema`, `format` and `default` in function `parameters`, and is the Responses API's `strict` default `true`? This needs a live call or the Responses spike. Anthropic accepts general JSON Schema when not strict [docs, anthropic-messages.md].
- Whether `ChildWithCleanup::drop` panics when a transport is dropped outside a Tokio runtime (relevant if users drop an `Agent` after the runtime has shut down). Inferred from source, not run.
- **Security:** #1030 (unbounded stdio line buffer) is open. Wait for the fix, or cap it in our own transport wrapper if a release is needed first.
