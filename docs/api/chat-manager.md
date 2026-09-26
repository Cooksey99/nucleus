# ChatManager

`ChatManager` is the primary entry point for interacting with nucleus. It orchestrates conversations between users, the LLM, and available plugins.

## Overview

```rust
pub struct ChatManager {
    // Internal fields omitted
}
```

**Module**: `nucleus-core`

## Responsibilities

- Manage multi-turn conversations with the LLM
- Detect and execute tool calls requested by the LLM
- Handle streaming responses
- Integrate RAG for context enrichment
- Keep one conversation on the manager and continue it across queries

## Construction

### `new(config: Config, registry: PluginRegistry) -> Result<Self>`

Creates a new `ChatManager` with default configuration.

```rust
use nucleus_core::{ChatManager, Config};
use nucleus_plugin::{PluginRegistry, Permission};

let config = Config::load_or_default();
let registry = PluginRegistry::new(Permission::READ_ONLY);
let manager = ChatManager::new(config, registry).await?;
```

**Note**: This creates a non-persistent RAG manager (in-memory only). For persistent storage, use `with_rag()`.

### Builder Methods

#### `with_provider(provider: Arc<dyn Provider>) -> Result<Self>`

Replace the LLM provider with a custom implementation.

```rust
use nucleus_core::provider::MistralRsProvider;

let custom_provider = Arc::new(MistralRsProvider::new(&config, registry).await?);
let manager = ChatManager::new(config, registry).await?
    .with_provider(custom_provider).await?;
```

#### `with_rag(rag: Rag) -> Self`

Replace the RAG system with a custom configuration.

```rust
use nucleus_core::Rag;

let custom_rag = Rag::new(&config, provider).await?;
let manager = ChatManager::new(config, registry).await?
    .with_rag(custom_rag);
```

## Core Methods

### `query(&self, user_message: &str) -> Result<String>`

Send a query and continue the stored conversation.

```rust
let response = manager.query("What files are in the src/ directory?").await?;
let follow_up = manager.query("Open the first one").await?;
println!("AI: {}", follow_up);
```

**Behavior**:
1. Appends the user message to stored history (seeding the system prompt on an empty history)
2. Sends that history to the LLM with available tool definitions
3. If the LLM requests tools, records those turns, executes them, and continues
4. Records the final assistant reply and returns its text

A later `query` sees the earlier turns. Call `clear_history()` to start over.

### `query_stream(&self, user_message: &str, on_chunk: F) -> Result<String>`

Streaming version of `query`. The callback receives each text chunk. The completed turn is still stored.

```rust
let response = manager.query_stream("Explain this codebase", |chunk| {
    print!("{}", chunk);
}).await?;
```

### `query_with(&self, messages: &[Message], user_message: &str) -> Result<String>`

Run one turn against a supplied history without reading or writing the stored conversation. `query_with_stream` is the streaming form. Use this for a trial, a branch, or a caller that owns the vec.

```rust
let trial = manager.query_with(&[], "What is 2 + 2?").await?;
```

## RAG / Knowledge Base Methods

### `knowledge_base_count(&self) -> usize`

Returns the number of documents currently in the knowledge base.

```rust
let count = manager.knowledge_base_count().await;
println!("Knowledge base contains {} documents", count);
```

### `load_knowledge_base(&self) -> Result<usize>`

**(Planned)** Load previously indexed documents from persistent storage.

```rust
let loaded = manager.load_knowledge_base().await?;
println!("Loaded {} documents from disk", loaded);
```

### `index_directory(&self, path: &Path) -> Result<usize>`

**(Planned)** Index a directory for semantic search.

```rust
use std::path::Path;

let indexed = manager.index_directory(Path::new("./src")).await?;
println!("Indexed {} files", indexed);
```

## Tool Execution Flow

When the LLM requests a tool:

1. **Detection**: ChatManager detects `tool_calls` in LLM response
2. **Execution**: Each tool is executed via `PluginRegistry::execute()`
3. **Injection**: Tool results are added as messages to conversation history
4. **Continuation**: Conversation continues until LLM returns a non-tool response

```text
User: "What's in main.rs?"
  ↓
LLM: [tool_call: read_file("main.rs")]
  ↓
Plugin Execution: ReadFilePlugin → file contents
  ↓
LLM: "The file contains..."
  ↓
User receives final response
```

## History

One `ChatManager` is one conversation. History stays in memory for the manager's lifetime. It is not written to disk.

```rust
let first = manager.query("Remember the number 7").await?;
let second = manager.query("What number did I just give you?").await?;

let snapshot = manager.history().await;
manager.set_history(snapshot); // replace; caller owns system prompt and prior turns
manager.clear_history().await; // next query starts clean and re-seeds the system prompt
```

`history`, `set_history`, and `clear_history` wait if a turn is in progress. Overlapping `query` calls on the same manager cannot interleave messages.

RAG context is retrieved for the new user text only. Older turns are not rewritten.

## Design Decisions to Consider

### 1. **Streaming API**

`query` returns the final string. `query_stream` invokes a callback for each chunk and still stores the completed turn.

### 2. **Tool Call Visibility**

Should tool calls be observable by the API consumer?

**Options**:
- A: Silent execution (current approach)
- B: Callback: `on_tool_call(|name, input, output| { ... })`
- C: Return `Response { text: String, tools_used: Vec<ToolCall> }`

### 3. **Error Handling**

What happens when a tool execution fails?

**Options**:
- A: Fail fast (return `Err` immediately)
- B: Inject error as tool result, let LLM handle it
- C: Retry with exponential backoff

## Examples

### Basic Query

```rust
use nucleus_core::{ChatManager, Config};
use nucleus_plugin::{PluginRegistry, Permission};

let config = Config::load_or_default();
let registry = PluginRegistry::new(Permission::READ_ONLY);
let manager = ChatManager::new(config, registry).await?;

let response = manager.query("Hello!").await?;
let follow_up = manager.query("What did I just say?").await?;
println!("AI: {}", follow_up);
```

### With File Reading Plugin

```rust
use nucleus_std::ReadFilePlugin;
use std::sync::Arc;

let mut registry = PluginRegistry::new(Permission::READ_ONLY);
registry.register(Arc::new(ReadFilePlugin::new()));

let manager = ChatManager::new(config, registry).await?;
let response = manager.query("What's in Cargo.toml?").await?;
println!("{}", response);
```

### Custom Provider

```rust
use nucleus_core::provider::OllamaProvider;

let custom_provider = Arc::new(OllamaProvider::new("llama3.2"));
let manager = ChatManager::new(config, registry).await?
    .with_provider(custom_provider).await?;
```

## See Also

- [Plugin Trait](./plugin-trait.md) - Creating custom tools
- [PluginRegistry](./plugin-registry.md) - Managing tools
- [Configuration](./configuration.md) - Configuring ChatManager behavior
- [RAG System](./rag.md) - Knowledge base integration
