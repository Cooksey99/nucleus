# Spike: mistral.rs (latest crates.io release)

Date: 2026-10-11 to 2026-10-12. No API keys were used and no paid calls were made.

Versions and sources checked:
- `mistralrs` / `mistralrs-core` / `mistralrs-mcp` **0.8.1** from crates.io (published 2026-04-02, the latest on crates.io). Read from the registry source (`~/.cargo/registry/src/.../mistralrs-*-0.8.1/`). Paths below such as `core/src/pipeline/sampling.rs:50` mean `mistralrs-core-0.8.1/src/...`, and `sdk/src/...` means `mistralrs-0.8.1/src/...`.
- GitHub `EricLBuehler/mistral.rs` at tag **v0.9.4** (2026-09-24, the latest GitHub release), via a shallow clone, plus `git log v0.8.0..v0.9.4`. The GitHub releases API, the crates.io API and sparse index, and docs.rs build logs.
- Spike code: `spikes/mistralrs/` (a standalone cargo project with an empty `[workspace]` table and `mistralrs = "=0.8.1"`, never merged). Its outputs are in `spikes/mistralrs/out/` and are cited below as "spike".
- Machine: Apple M4 Max (16 cores, 64 GB), macOS, Rust 1.94.0, Command Line Tools only (no full Xcode).
- Models (all already in the local HF cache, so nothing new was downloaded): `MaziyarPanahi/Qwen3-0.6B-GGUF:Qwen3-0.6B.Q4_K_M.gguf` (chat), `Qwen/Qwen3-0.6B` safetensors with ISQ Q4K (chat, used for comparison), and `Qwen/Qwen3-Embedding-0.6B` (embeddings).

How each claim was established: **[verified]** means I ran it in the spike or read it directly in the named source file. **[docs]** means I read it in docs, the changelog, issues or the README. **[inferred]** means it is my conclusion from those.

## Questions

From the roadmap (Phase 0.5, Phase 3 "Local provider", Phase 4 RAG), as assigned:
1. What is the latest crates.io release, and can nucleus move off the git pin to it? What are the release cadence, MSRV, edition and license?
2. Tool calling: how are tools passed per request, does the request's own tool list work, and what shape is the response?
3. Streamed tool calls: do they arrive as argument deltas or whole? What do the chunk types look like?
4. Embedding models: which are supported, what is the API, can it batch, and what are the dims and speed?
5. JSON-schema constrained output: does it work, and how reliable is it on a small model?
6. Metal: what does it need, does it work here, and how much faster is it?
7. Sharing a loaded model: can one `Arc<Model>` serve concurrent tasks? What happens with N concurrent requests? Can a chat model and an embedding model be loaded together?
8. Footprint: clean compile time, dependency count, and which features can be turned off.
9. Other things that affect `Local`: per-request sampling, chat templates and Qwen3 thinking, cancellation, timeouts, download and cache location, and loading GGUF from a path, an HF repo or `repo:file.gguf`.

Found along the way:
10. GitHub releases went on to 0.9.4, but crates.io stopped at 0.8.1.
11. In 0.8.1, the SDK helper that replays assistant tool calls into history drops them silently.
12. A slow stream consumer stalls every other request on the same model.
13. On Metal: wrong embeddings for short inputs, NaN logits with the prefix cache, and a hang with paged attention at 16 concurrent requests.
14. `with_logging()` installs a global tracing subscriber that writes to stderr. The current nucleus code calls it.

## Findings

### 1. Release, cadence, MSRV, edition, license

- **crates.io's latest is 0.8.1** (2026-04-02). Only 5 versions were ever published there: 0.7.0-alpha.3, 0.7.0-alpha.4, 0.7.0, 0.8.0 and 0.8.1. **[verified]** crates.io API and `index.crates.io/mi/st/mistralrs`.
- 0.8.1 has MSRV (`rust-version`) **1.88**, edition **2021** and license **MIT**, and **no default features**. **[verified]** crates.io API, `mistralrs-0.8.1/Cargo.toml`.
- **GitHub has released much further than crates.io:** 36 GitHub releases since 0.8.1, from v0.8.2 (2026-06-01) to **v0.9.4 (2026-09-24)**, with 278 commits from v0.8.0 to v0.9.4. Cadence: bursts (14 tags from 06-16 to 06-25), then roughly every 2–4 weeks. **[verified]** GitHub releases API, `git log`.
- **Why crates.io is stuck** **[verified]** / **[inferred]**: at v0.9.4 the workspace depends on candle through a git rev (`candle-core = { git = ".../candle.git", version = "0.11.0", rev = "66a8cf1" }`, `Cargo.toml:42-45`), and `release.yml` publishes to PyPI, Docker and GitHub binaries, never to crates.io. The README still says `cargo add mistralrs` (`README.md:225`). At v0.9.4, MSRV is 1.94.
- **docs.rs has no docs for 0.8.x.** The 0.8.0 and 0.8.1 builds failed because docs.rs builds with `all-features` on Linux, which turns on `metal` and pulls in `objc2`. 0.7.0 is the newest version documented there. **[verified]** [docs.rs build 3095014](https://docs.rs/crate/mistralrs/0.8.1/builds/3095014). Read the source instead.
- The current pin `2ed5575` is dated 2026-02-27, between 0.7.0 and 0.8.0. **[verified]** GitHub commits API.
- **Can we move to 0.8.1? Yes.** The spike uses the same API surface nucleus uses today (`GgufModelBuilder`, `TextModelBuilder`, `EmbeddingModelBuilder`, `RequestBuilder`, `TextMessages`, `Response::Chunk`). It builds and runs on macOS, CPU and Metal. **[verified]** spike. I did not compile `nucleus-core` itself against 0.8.1. **[inferred]** The migration is small.
- **The catch:** many fixes we want landed after 0.8.1 and exist only on git: parallel Qwen tool calls, `ToolChoice::Required` and `AllowedTools`, tool-call grammar and strict mode, separating content from tool calls, and the history fix in finding 2. Example commits: `c88363cc`, `a7156921`, `1e3c73b6`. **[verified]** `git log`, v0.9.4 `mistralrs-core/src/tools/`.

### 2. Tool calling (per request)

- Tools go on each request: `RequestBuilder::set_tools(Vec<Tool>)` and `.set_tool_choice(ToolChoice::{None, Auto, Tool(Tool)})`. There is no global registry. A request with no tools on the same `Model` got plain text (`finish=stop`, `tool_calls=None`). **[verified]** spike `out/tools-none.txt`.
- 0.8.1 has no `"required"` tool choice. `Tool(..)` only raises an error after generation if no call was made; it does not force one. **[verified]** `core/src/tools/request.rs`, `core/src/tools/mod.rs:430-476`.
- `Tool { tp: ToolType::Function, function: Function { name, description: Option<String>, parameters: Option<HashMap<String, Value>> } }` comes from `mistralrs-mcp/src/tools.rs:34-48`. Pass the **whole JSON Schema object** (`type`, `properties`, `required`) as the map. Today's nucleus code passes only `properties`. **[verified]** spike: the full schema produced correct calls.
- **Response shape** **[verified]** `out/tools-cpu.txt`:
  ```json
  {"finish_reason":"tool_calls","message":{
    "content":"<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}\n</tool_call>",
    "role":"assistant",
    "tool_calls":[{"index":0,"id":"call-<uuid4>","type":"function",
                   "function":{"name":"get_weather","arguments":"{\"city\":\"Paris\"}"}}],
    "reasoning_content":"…only when thinking is on…"}}
  ```
  - `arguments` is a JSON **string**.
  - `content` still holds the raw `<tool_call>` text, so it must be dropped when `tool_calls` is present. Upstream issue #2427 reports the same leak in reasoning mode.
- Calls are parsed from the generated text; decoding is not constrained. **[verified]** `core/src/tools/mod.rs`.
- **Only one tool call per turn:** generation stops at the first complete call (`core/src/pipeline/sampling.rs:50-64`). I asked for weather *and* time; the reasoning planned both calls, but only `get_weather` came back. **[verified]** spike. v0.9.4 keeps buffering for parallel Qwen calls (`tools/mod.rs:221`).
- **The history-replay bug (0.8.1):** `RequestBuilder::add_message_with_tool_call` stores the calls under the key `"function"` instead of `"tool_calls"` (`sdk/src/messages.rs:568-600`), so the chat template never renders them. **[verified]** Prompt token counts:
  - helper with the calls: 193
  - plain empty assistant message: 193
  - raw `<tool_call>` text as the assistant content: 212

  (`out/history.txt`.) This is fixed in v0.9.4 (`mistralrs/src/messages.rs:851`). `add_tool_message(content, tool_call_id)` works.

### 3. Streamed tool calls

- **Tool calls arrive whole, not as argument deltas.** **[verified]** Text that could be the start of a call is held back (`prefix_could_be_tool`). The complete call then arrives in a single chunk with `finish_reason: "tool_calls"`, and that same chunk carries `usage` (`out/tools-stream-cpu.txt`, `core/src/pipeline/sampling.rs:69-190`). The chunk's `delta.content` also repeats the raw `<tool_call>` text. v0.9.4 has the same `Delta` struct, so still no argument deltas. **[verified]** v0.9.4 `response.rs:48`.
- Chunk types **[verified]** `core/src/response.rs`:
  - The stream yields `Response`. The variants that matter are `Chunk(ChatCompletionChunkResponse)`, `ModelError(String, _)`, `InternalError` and `ValidationError`.
  - `ChatCompletionChunkResponse { id, choices: Vec<ChunkChoice>, created, model, system_fingerprint, object, usage: Option<Usage> }`.
  - `ChunkChoice { finish_reason: Option<String>, index, delta: Delta, logprobs }`.
  - `Delta { content, role, tool_calls: Option<Vec<ToolCallResponse>>, reasoning_content }`.
  - `Usage` has token counts plus tok/s and timings.
- The stream ends when the channel closes. No `Response::Done` arrives in streaming. Some chunks have both `content` and `reasoning_content` as `None`. **[verified]** spike.
- The types are `Serialize` only and not `#[non_exhaustive]`. **[verified]** They stay private behind our `Event`.

### 4. Embedding models

- **Only two architectures are supported:** EmbeddingGemma (`Gemma3TextModel`) and Qwen3-Embedding (`Qwen3ForCausalLM`), in both 0.8.1 and v0.9.4 (`core/src/pipeline/loaders/embedding_loaders.rs:91-108`). **[verified]**
  - No BERT family (BGE, nomic-embed, MiniLM).
  - No GGUF embeddings: `EmbeddingModelBuilder` loads safetensors from an HF repo.
  - The current nucleus docs mention `nomic-embed-text`, which cannot work.
- **API:** `EmbeddingModelBuilder::new(repo).build()` returns a `Model`. Then:
  - `generate_embedding(text)` returns `Vec<f32>`.
  - `generate_embeddings(EmbeddingRequest::builder().add_prompts(iter))` returns `Vec<Vec<f32>>` in input order.

  The "batch" sends one request per input concurrently (`join_all`), and the engine batches them (`sdk/src/model.rs:502-578`). **[verified]**
- Qwen3-Embedding-0.6B produces **1024 dims**, L2-normalised (norm ≈ 1.00). **[verified]** spike.

  | Qwen3-Embedding-0.6B, ~60-token inputs | CPU | Metal |
  |---|---|---|
  | load (cached) | 2.2 s | 0.33 s |
  | one at a time | 115 ms each | 16.6 ms each |
  | batch of 16 | 41 ms each | 9.7 ms each |
  | batch of 64 | 31.9 ms each | 5.9 ms each |

  **[verified]** `out/embed-cpu.txt`, `out/embed-metal.txt`.
- CPU quality sanity check: cos(query, relevant) = 0.508, cos(query, irrelevant) = 0.287. **[verified]**
- **Metal bug: inputs of 7 words or fewer give wrong vectors.** Their cosine against the CPU vector is about 0; from 8 words up it is 1.00. This holds with f32, f16 and bf16. On Metal, the same check gives 0.012 vs −0.015. **[verified]** `embed_dump` sweep over 1–48 words, `out/ed-*.json`. RAG queries are usually short, so **Metal embeddings can't be used in 0.8.1**, while CPU is fast enough. I did not test whether v0.9.4 fixes it.
- EmbeddingGemma is gated on HF, so I didn't test it. **[docs]**

### 5. JSON-schema constrained output

`.set_constraint(Constraint::JsonSchema(schema))` uses llguidance (`core/src/request.rs:21`). Test setup: a schema with 5 required fields (an enum, an integer range, an array with `maxItems`), 10 prompts, temperature 0.8, `max_len` 200, Qwen3-0.6B, Metal. **[verified]** `out/json-metal-*.txt`.

| setup | raw content parses | schema-valid (after stripping EOS text) | tokens used |
|---|---|---|---|
| unconstrained (prompted "JSON only"; GGUF / ISQ) | 1/10 / 0/10 | 0/10 / 0/10 (markdown fences, invented keys) | – |
| GGUF Q4_K_M + schema | **0/10** | 8/10 | 2000 (every run hit `max_len`) |
| safetensors + ISQ Q4K + schema | 8/10 | 8/10 | 802 |
| GGUF + schema + `"x-guidance": {"whitespace_flexible": false}` | 0/10 | **10/10** | 2000 |
| ISQ + schema + `whitespace_flexible: false` | **10/10** | **10/10** | 360 |

- **GGUF EOS bug:** once the JSON is complete, the GGUF model emits `<|endoftext|>` until `max_len`, so the result ends with `finish=length`. The chat stop token is `<|im_end|>`. It is related to upstream issue #2204 (open, same model family and version, and it can also fail with a hard error at temperature 0). **[verified]** / **[docs]**
- **The 2/10 failures were whitespace runaway inside arrays.** `whitespace_flexible: false` fixes it. **[verified]**
- The constraint guarantees shape, not content: values from a 0.6B model can be nonsense (`"name": "strconv(42, int)"`). **[verified]**

### 6. Metal

- The `metal` feature compiles `.metal` kernels with the Metal compiler at build time. It worked here with **Command Line Tools plus the Metal toolchain** (`xcrun --find metal` → `/Volumes/MetalToolchainCryptex/...`), with no full Xcode. **[verified]** `ps` during the build. A clean release build takes 3m21s, against 1m40s without Metal (see 8).
- Without `metal`, macOS runs on CPU: `best_device` calls `cuda_if_available` (`sdk/src/model.rs:17-29`). **[verified]**
- **Speed** (Qwen3-0.6B Q4_K_M, 24-token prompt, about 170 output tokens) **[verified]** `out/speed.txt`:
  - CPU: prefill 88 tok/s, decode 37 tok/s
  - Metal: prefill 1600 tok/s, decode 255 tok/s
  - That is **about 7× faster decode and 18× faster prefill**. Load takes about 1 s.
- Metal-only bugs **[verified]**:
  - the short-input embedding bug (finding 4)
  - **NaN logits with the prefix cache:** the 3rd request in `thinking` fails deterministically with "Invalid sampling probability … NaN". It passes as a fresh request, and passes with `with_prefix_cache_n(None)` (twice each way).
  - the paged-attention hang (finding 7)

### 7. Sharing a loaded model, concurrency, chat plus embedding

- `Model` methods take `&self`. `Arc<Model>` shared across `tokio::spawn` tasks works. **[verified]** `out/concurrent-*.txt`.

  | requests (Metal, 128 max tokens) | sequential | concurrent | aggregate tok/s |
  |---|---|---|---|
  | 4 | 1.15 s | 0.84 s | 243 → 333 |
  | 16 | 4.71 s | 3.15 s | 245 → 367 |
  | 8 (CPU) | 12.75 s | 8.89 s | 37 → 56 |

  - About **1.4–1.5× throughput**, but each request's latency grows: on CPU all 8 finished together at 8.9 s.
  - Requests beyond `max_num_seqs` (default 32, `sdk/src/gguf.rs:44-60`) queue.
  - Batched greedy output can differ slightly from sequential output (CPU: 496 vs 475 tokens). **[verified]**
- **Paged attention (opt-in) on Metal:** 1 or 4 concurrent requests worked. **16 concurrent requests hung** for more than 115 s, logging thousands of `Receiver disconnected` warnings. **[verified]** `out/paged-conc16-log.txt`. Keep it off.
- **Head-of-line blocking:** each response channel has capacity 1 (`sdk/src/model.rs:120`), and the engine awaits the send for each chunk (`core/src/sequence.rs:1520-1541`). A request that took 0.62 s alone took **7.51 s** while another stream was read at 50 ms per chunk. **[verified]** `out/misc-metal.txt`.
- **Chat plus embedding in one handle:** `MultiModelBuilder::new().add_model_with_alias("chat", GgufModelBuilder…).add_model_with_alias("embed", EmbeddingModelBuilder…)` loaded both in 2.6 s. Calls are routed with `send_chat_request_with_model(.., Some("chat"))` and `generate_embedding_with_model(.., Some("embed"))`. A wrong-category call errors cleanly ("Received a request incompatible for this model's category"). **[verified]** Two separate `Model`s would also work (separate engines). **[inferred]**

### 8. Footprint

Spike crate depending only on `mistralrs` plus tokio, anyhow, serde_json and futures, which mistralrs already pulls in:

| | default features | `metal` |
|---|---|---|
| `cargo tree --prefix none \| sort -u \| wc -l` | **665** | **681** |
| same, `-e normal` | 647 | 663 |
| same, `-e normal --target x86_64-unknown-linux-gnu` | 634 | – |
| clean `cargo build --release` | **1m40s** (user 886 s), 1.4 GB target dir | **3m21s** (user 1078 s), 2.0 GB |
| clean `cargo build` (debug) | 52 s, 4.3 GB | – |

**[verified]** `out/build-*.log`.

- **Debug builds are about 10× slower at inference** (CPU decode 3.5 tok/s, against 37 in release). **[verified]**
- Features: `mistralrs` has none on by default. The optional ones are `metal`, `cuda`, `cudnn`, `flash-attn`, `accelerate`, `mkl`, `nccl` and `ring` (use ring instead of aws-lc-rs for TLS). **[verified]** `mistralrs-0.8.1/Cargo.toml`.
- **These cannot be turned off:**
  - an MCP client (`rust-mcp-schema`, `tokio-tungstenite`)
  - `clap`
  - `image` with 14 formats
  - `symphonia` and `rubato` (audio)
  - `hf-hub`
  - reqwest 0.12 *and* 0.13
  - tokenizers 0.21 *and* 0.22
  - `llguidance`, `openai-harmony`, `onig`, `aws-lc-sys`, `sysinfo`, `zip`, `tracing-subscriber`

  **[verified]** `cargo tree`.

### 9. Other things that affect `Local`

- **Sampling:** every `RequestBuilder` starts from `SamplingParams::deterministic()`, which is `top_k = 1` (`sdk/src/messages.rs:510-528`). As a result, **`set_sampler_temperature(1.5)` alone does nothing**: it gave 4 identical outputs. `set_sampling(SamplingParams { temperature, top_k: None, top_p, max_len, .. })` gave 4 different ones. `max_len` works (`finish_reason "length"`). **[verified]** `out/misc-metal.txt`. The model's `generation_config` defaults are applied only on opt-in (`SamplingParams::apply_model_defaults`). **[verified]** source.
- **Thinking (Qwen3):**
  - **On by default** (`DEFAULT_ENABLE_THINKING = true`, `core/src/pipeline/chat_template.rs:27`). It can be set per request with `.enable_thinking(bool)`.
  - Reasoning comes back separately in `reasoning_content`, and in streaming as `delta.reasoning_content` (CPU: 172 reasoning chunks, then 15 content chunks).
  - It is expensive on small models: 499 tokens of thinking for "17 × 23" (`content` was `"\n\n391"`), against 12 tokens with thinking off. Content can start with `"\n\n"`.

  **[verified]**
- **Chat templates:** read from GGUF metadata ("Using literal chat template"; bos/eos taken from it), or overridden with `with_chat_template(path)` or `with_jinja_explicit`. `with_chat_template` **panics** unless the path ends in `.json` or `.jinja` (`core/src/pipeline/macros.rs:440`). **[verified]**
- **Cancellation:**
  - Dropping a *stream* cancels the sequence on its next send (`core/src/pipeline/sampling.rs:200`): the next request took 64 ms, against 59 ms baseline.
  - Dropping a *non-streaming* future (`tokio::time::timeout`) does **not** cancel. Generation keeps running and slows later requests (272 ms, then 365 ms one second later).

  **[verified]** `cancel`.
- **Timeouts:** there are none built in. **[verified]** source.
- **Download and cache:** hf-hub, with this precedence: `HF_HUB_CACHE`, then `HF_HOME/hub`, then `~/.cache/huggingface/hub` (`core/src/pipeline/hf.rs:24-62`). Repo file lists are cached as `<org>-<repo>_repo_list.json` in that directory. The token source defaults to `CacheToken` (`HF_HOME/token`). **[verified]**
  - With the network unreachable (a dead proxy), a cached `repo:file` loaded and ran normally.
  - An uncached repo failed with a clear error.

  **[verified]**
- **Logging:** `with_logging()` calls `initialize_logging()` (`sdk/src/model_builder_trait.rs:269`), which installs a global `tracing-subscriber` writing to stderr. Without it the library printed nothing (cached model). **[verified]** Progress bars during a first download were not observed. **[inferred]** They are tied to the same flag (`with_progress(!silent)`).
- **Loading:**
  - Local path: `GgufModelBuilder::new(dir, vec![file])` works (1.07 s).
  - HF GGUF (`repo:file.gguf`): `GgufModelBuilder::new(repo, vec![file])` works (~1 s cached).
  - HF safetensors repo: `TextModelBuilder::new(repo).with_isq(IsqType::Q4K)` works (2.7 s for 0.6B, cached).

  All three model-string forms map cleanly. **[verified]**
- **Panics in library code:** `with_chat_template` (above), and `.expect` in `generate_embedding` (`sdk/src/model.rs:603`). **[verified]**

## Impact on the design

1. **Open decision "Move mistral.rs to a crates.io release": resolve it as yes, `mistralrs = "0.8.1"`** (pinned `=0.8.1` until we re-check). Also:
   - Open an upstream issue asking for 0.9.x on crates.io. It's blocked by the candle git dependency.
   - Re-check before the 0.2 release. If 0.9.x reaches crates.io, take it: it has parallel calls, `Required` tool choice and the history fix.
   - Record in ROADMAP: MSRV ≥ 1.88 (≥ 1.94 for 0.9.x), and docs.rs has no 0.8.x docs.
2. **Phase 3 Local provider: add these requirements:**
   - Always use `stream_chat_request` internally, even for non-streaming calls. A dedicated task drains it into our own buffer, which avoids head-of-line blocking (finding 7), and dropping it is our cancellation. Timeouts are ours: `tokio::time::timeout` around each `next()`.
   - Map tool calls to a complete `ToolCall` event. Strip `<tool_call>` text from `content` when `tool_calls` is present.
   - The `Event` design must not assume argument deltas. Make `ToolCallDelta` optional per provider, or emit only complete tool calls everywhere.
   - Expect at most one tool call per local turn; the loop must not depend on parallel calls.
   - Replay assistant tool calls ourselves. With 0.8.1, send the raw `<tool_call>…</tool_call>` text as the assistant content. That workaround is verified; using the lower-level `MistralRs` request with a `"tool_calls"` key is cleaner but untested. Never use `add_message_with_tool_call` on 0.8.1.
   - Always set a full `SamplingParams`: temperature → `temperature` with `top_k: None`. Never use `set_sampler_temperature` alone.
   - Expose thinking as a `RunOptions` setting mapped to `enable_thinking`, with an explicit nucleus default. I recommend off for `local/`, because it multiplies tokens on small models.
   - Never call `with_logging()` or `with_throughput_logging()`; this is the library-never-prints rule. Today's `mistralrs.rs` calls both.
   - `Local::builder()` defaults: prefix cache **off on Metal** (`with_prefix_cache_n(None)`), paged attention **off**, and `max_num_seqs` exposed.
   - Pass the full JSON Schema for `parameters`, not just `properties`.
3. **`agent.extract::<T>()` on local:**
   - Use `Constraint::JsonSchema` with `"x-guidance": {"whitespace_flexible": false}` added, and a `max_len` cap.
   - Strip trailing EOS text before parsing.
   - Validate, and return a typed error on failure.
   - Document that safetensors+ISQ works better than GGUF here (the GGUF EOS bug).
4. **Phase 4 RAG:**
   - The default local embedder is **`Qwen/Qwen3-Embedding-0.6B`** (1024 dims, ~1.1 GB). The only other option is EmbeddingGemma (gated).
   - **Run the embedder on CPU even with `metal`** (`with_force_cpu`) until a test proves Metal correct for short inputs. Add that test: compare the embedding of a ≤ 7-word query against a stored reference.
   - `embed_batch` should send all inputs in one `generate_embeddings` call.
   - Store the model id and dims in the index.
   - Remove the `nomic-embed-text` and `mxbai` references.
5. **Footprint and CI:**
   - `local` adds about 630 crates, a 1m40s release build or 52 s debug build, and 1.4–4.3 GB of target dir.
   - Local tests need optimised dependencies (`[profile.dev.package."*"] opt-level = 3`) or `--release`. Otherwise the tiny-model tests run 10× slower.
   - A macOS `metal` CI job needs the Metal toolchain.
   - `cargo-deny` must cover about 650 crates (aws-lc, onig, …). Consider the `ring` feature to match the hosted TLS stack.
6. **Batch (local):** concurrent requests on one `Arc<Model>` are verified (~1.5× throughput, bounded by `max_num_seqs`). Paged attention stays off (it hangs at 16).
7. **Model-string grammar:** the three local forms (path, `repo:file.gguf`, repo) all load. No change needed.

## Phase 1 pin

Date: 2026-10-15. Nucleus now depends on git tag **v0.9.4** behind the `local` feature, not commit `2ed5575`.

What was checked:
- The current `MistralRsProvider` compiles against that tag. The only API break was `Function.strict: Option<bool>` (left unset). **[verified]** `cargo check --features local,rag,experimental`.
- v0.9.4's MSRV is 1.94, so that is now the crate `rust-version`. **[verified]** v0.9.4 `Cargo.toml` and a local check on Rust 1.94.

What was not re-run: the 0.8.1 behavior checks (tool-call history, one call per turn, GGUF JSON that never stops, Metal short embeddings, prefix-cache NaNs, paged attention at 16). The spike crate in `spikes/mistralrs` is still `mistralrs = "=0.8.1"`. Those stay open until that spike is pointed at v0.9.4 and run again.

## Open questions

- Will upstream publish 0.9.x to crates.io, and when? Raise it on GitHub.
- Does v0.9.4 fix the Metal short-input embeddings, the prefix-cache NaN, the paged-attention hang at 16 concurrent requests, and the GGUF EOS padding under JSON constraints? Not re-run against the tag. The source at v0.9.4 still has the history-replay fix noted above.
- Linux x86 CPU and CUDA behaviour, and Windows, are untested. The Linux dependency tree resolves without `objc2`.
- Tool-calling quality on 1.7B or 4B models (only 0.6B was tested).
- EmbeddingGemma (gated on HF) is untested.
- Whether a first download prints progress bars when `with_logging` is off. It wasn't observed because every model was already cached.
- The raw `MistralRs` request path with a `"tool_calls"` message key, as the cleaner history workaround, is unverified.
