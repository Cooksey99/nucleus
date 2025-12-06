# API Overview

Nucleus exposes a clean, composable API for building AI-powered developer tools. The API is organized around a few core types that you compose together based on your needs.

## Core API Surface

### Primary Entry Points

- **[`ChatManager`](./chat-manager.md)** - Orchestrates conversations with tool execution (`nucleus-core`)
- **[`Config`](./configuration.md)** - Configures LLM, RAG, storage, and permissions (`nucleus-core`)
- **[`Plugin`](./plugin-trait.md)** - Trait for extending AI capabilities (`nucleus-plugin`)
- **[`PluginRegistry`](./plugin-registry.md)** - Manages available tools and permissions (`nucleus-plugin`)
- **[`Rag`](./rag.md)** - Knowledge base and semantic search (`nucleus-core`)
- **[`Provider`](./providers.md)** - LLM backend abstraction (`nucleus-core`)

### Standard Plugins

Pre-built plugins in `nucleus-std`:
- `ReadFilePlugin` - Read file contents
- `WriteFilePlugin` - Write/modify files
- `SearchPlugin` - Semantic codebase search
- `ExecPlugin` - Execute shell commands

### Developer Plugins

Advanced integrations in `nucleus-dev`:
- `GitPlugin` - Git operations
- `LspPlugin` - Language server integration

## Basic Usage Pattern

```rust
use nucleus_core::{ChatManager, Config};
use nucleus_plugin::{PluginRegistry, Permission};
use nucleus_std::ReadFilePlugin;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // 1. Load configuration
    let config = Config::load_or_default();
    
    // 2. Create plugin registry with permissions
    let mut registry = PluginRegistry::new(Permission::READ_ONLY);
    registry.register(ReadFilePlugin::new());
    
    // 3. Create chat manager
    let manager = ChatManager::new(config, registry).await?;
    
    // 4. Query the AI
    let response = manager.query("Summarize main.rs").await?;
    println!("AI: {}", response);
    
    Ok(())
}
```

## Design Principles

The API follows these design principles:

1. **Composability** - Build managers, registries, and plugins independently
2. **Type Safety** - Use Rust's type system to prevent misuse
3. **Explicit Control** - No hidden magic; you control configuration and permissions
4. **Async-First** - All I/O operations are async for performance
5. **Builder Pattern** - Chain configuration methods for ergonomic setup

## Integration Points

Nucleus is designed to be embedded in other applications:

- **Terminal Wrappers** - Intercept shell commands and provide AI assistance
- **IDE Plugins** - Add AI capabilities to code editors
- **CLI Tools** - Build standalone AI-powered developer utilities
- **Web Services** - Expose AI capabilities through HTTP APIs

## Module Organization

```
nucleus/
├── nucleus-core/       # Core orchestration and LLM integration
│   ├── ChatManager     # Main API entry point
│   ├── Config          # Configuration system
│   ├── Rag             # Knowledge base / RAG
│   └── Provider        # LLM backend abstraction
│
├── nucleus-plugin/     # Plugin system foundation
│   ├── Plugin          # Core trait all plugins implement
│   ├── PluginRegistry  # Tool management and execution
│   └── Permission      # Security boundaries
│
├── nucleus-std/        # Standard library of plugins
│   ├── ReadFilePlugin
│   ├── WriteFilePlugin
│   ├── SearchPlugin
│   └── ExecPlugin
│
└── nucleus-dev/        # Developer-focused plugins
    ├── GitPlugin
    └── LspPlugin
```

## Next Steps

- [ChatManager API](./chat-manager.md) - Learn about conversation management
- [Plugin Trait](./plugin-trait.md) - Build custom capabilities
- [Configuration](./configuration.md) - Understand configuration options
- [Integration Guide](../guides/integration.md) - Embed nucleus in your app
