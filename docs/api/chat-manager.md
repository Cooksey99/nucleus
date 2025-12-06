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
- Maintain conversation history

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

Send a query to the AI and get a response.

```rust
let response = manager.query("What files are in the src/ directory?").await?;
println!("AI: {}", response);
```

**Behavior**:
1. Sends user message to LLM with available tool definitions
2. If LLM requests tools, executes them and continues conversation
3. Returns final text response when LLM is satisfied
4. Automatically handles multi-turn tool execution loops

### `stream_query(&self, user_message: &str) -> impl Stream<Item = Result<String>>`

**(To be implemented)** Stream responses token-by-token for real-time UX.

```rust
let mut stream = manager.stream_query("Explain this codebase").await?;
while let Some(chunk) = stream.next().await {
    print!("{}", chunk?);
}
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

## State Management

**Current State**: Conversation history is maintained in memory during the `ChatManager` lifetime.

**Planned**: Persistent conversation storage with loading/saving capabilities.

## Design Decisions to Consider

### 1. **Streaming API**

Should `query()` return a stream by default, or keep separate `query()` and `stream_query()` methods?

**Options**:
- A: Single `query()` returns `Stream<String>` (always streaming)
- B: Separate `query()` (blocking) and `stream_query()` (streaming)
- C: Generic `query<R: ResponseType>()` with trait-based selection

### 2. **History Management**

How should users access/modify conversation history?

**Options**:
- A: `get_history() -> &[Message]` (read-only)
- B: `clear_history()`, `append_history()`, `set_history()` (mutable)
- C: Separate `ConversationSession` type that wraps history

### 3. **Tool Call Visibility**

Should tool calls be observable by the API consumer?

**Options**:
- A: Silent execution (current approach)
- B: Callback: `on_tool_call(|name, input, output| { ... })`
- C: Return `Response { text: String, tools_used: Vec<ToolCall> }`

### 4. **Error Handling**

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
println!("AI: {}", response);
```

### With File Reading Plugin

```rust
use nucleus_std::ReadFilePlugin;
use std::sync::Arc;

let mut registry = PluginRegistry::new(Permission::READ_ONLY);
registry.register(Arc::new(ReadFilePlugin::new()));

let manager = ChatManager::new(config, registry).await?;
let response = manager.query("What's in Cargo.toml?").await?;
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
