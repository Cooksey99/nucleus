# Nucleus

[![Documentation](https://docs.rs/nucleus/badge.svg)](https://docs.rs/nucleus)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

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

## Installation

### 1. Build Nucleus

```bash
cargo build --release
```

### 2. Run Examples

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

## Installation

Add Nucleus to your `Cargo.toml`:

```toml
[dependencies]
nucleus = "0.1"  # Includes core + std plugins by default
```

For minimal setup without standard plugins:
```toml
[dependencies]
nucleus = { version = "0.1", default-features = false }
nucleus-core = "0.1"
```

For full functionality including dev tools:
```toml
[dependencies]
nucleus = { version = "0.1", features = ["full"] }
```

### Vector Database (for RAG)

Nucleus supports vector databases for persistent storage in RAG (Retrieval Augmented Generation):

- **Automatic deduplication**: Re-indexing replaces old documents
- **Persistent storage**: Data survives restarts
- **Fast vector search**: Efficient indexing for large codebases

See the configuration documentation for supported backends and setup instructions.
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

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

Areas where we'd love help:
- New plugins for the tool system
- Support for additional LLM backends
- Documentation and examples
- Testing and bug fixes

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Status

⚠️ **Note**: Nucleus is under active development. APIs may change as we work toward a stable 1.0 release.
