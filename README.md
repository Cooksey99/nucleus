# Nucleus

A privacy-first, modular AI engine for building customizable developer tooling. Nucleus provides the infrastructure to integrate local or self-hosted LLMs with tool-based capabilities, enabling AI assistants that can interact with files, execute commands, and understand codebases - all without sending data to external services.

## What It Does

Nucleus is a **library**, not an application. It provides the building blocks for creating AI-powered developer tools:

- **Tool-Augmented LLM**: Execute real actions (read files, search code, run commands) instead of just text generation
- **Plugin System**: Extensible tools with permission controls
- **Local-First**: Works with local LLMs (Ollama, llama.cpp, etc.) - no data leaves your machine
- **Modular Architecture**: Use only what you need

## Example Usage

Here's a minimal example that lets an AI read and analyze files:

```rust
use nucleus_core::{ChatManager, Config};
use nucleus_plugin::{PluginRegistry, Permission};
use nucleus_std::ReadFilePlugin;
use std::sync::Arc;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let config = Config::load_or_default();
    
    // Register plugins the AI can use
    let mut registry = PluginRegistry::new(Permission::READ_ONLY);
    registry.register(Arc::new(ReadFilePlugin::new()));
    
    let manager = ChatManager::new(config, Arc::new(registry)).await;
    
    // Ask the AI a question - it will use plugins to answer
    let response = manager.query(
        "What's on line 7 of Cargo.toml?"
    ).await?;
    
    println!("AI: {}", response);
    Ok(())
}
```

The AI will:
1. Recognize it needs to read a file
2. Use the `ReadFilePlugin` to get the content
3. Analyze and respond with the answer

## Building

```bash
cargo build --release
```

Run examples:
```bash
cargo run --example read_file_line
cargo run --example write_file
```

## Architecture

Nucleus is structured as a workspace with clear separation:

- **nucleus-core**: LLM orchestration, chat management, configuration
- **nucleus-plugin**: Plugin trait and registry system
- **nucleus-std**: Standard plugins (file I/O, search, exec)
- **nucleus-dev**: Developer-specific plugins (git, LSP integration)
- **nucleus**: Convenience wrapper with feature flags

## Dependencies

```toml
[dependencies]
nucleus = "0.1"  # Includes core + std plugins by default
```

For minimal setup:
```toml
nucleus = { version = "0.1", default-features = false }
nucleus-core = "0.1"
```

### Vector Database (for RAG)

Nucleus uses [Qdrant](https://qdrant.tech/) for persistent vector storage in RAG (Retrieval Augmented Generation):

```bash
# Using Docker
docker run -p 6334:6334 qdrant/qdrant

# Or install locally: https://qdrant.tech/documentation/quick-start/
```

Qdrant provides:
- **Automatic deduplication**: Re-indexing replaces old documents
- **Persistent storage**: Data survives restarts
- **Fast vector search**: HNSW indexing for millions of documents

## Privacy

All data stays on your machine:
- Connects only to local/self-hosted LLM backends
- No telemetry or analytics
- All storage under user control

## What You Can Build

Nucleus is infrastructure for building tools like:
- AI-enhanced terminal wrappers
- Code review assistants
- Interactive documentation systems
- Project-specific AI helpers
- Development environment integrations
