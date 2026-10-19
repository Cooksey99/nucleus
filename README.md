# Nucleus

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

A privacy-first, modular AI engine for building customizable developer tooling. Nucleus provides the infrastructure to integrate local or self-hosted LLMs with tool-based capabilities, enabling AI assistants that can interact with files, execute commands, and understand codebases - all without sending data to external services.

## What It Does

Nucleus is a **library**, not an application. It provides the building blocks for creating AI-powered developer tools:

- **Tool-Augmented LLM**: Execute real actions (read files, search code, run commands) instead of just text generation
- **Plugin System**: Extensible tools the model can call
- **Local-First**: Works with a local mistral.rs model - no data leaves your machine
- **Modular Architecture**: Use only what you need

## Example Usage

Here's a minimal example that makes a single hosted completion:

```rust
use nucleus::Error;

#[tokio::main]
async fn main() -> Result<(), Error> {
    let summary = nucleus::complete("openai/gpt-5.1", "Summarize the design in one line.").await?;
    println!("{summary}");
    Ok(())
}
```

The agent API - tools, retrieval and conversations - is specified in
[`docs/design/core-api.md`](docs/design/core-api.md). The example above is the
`one_shot` example; it is written against the new API and does not run yet
because the agent internals are still being built.

## Installation

### 1. Build Nucleus

```bash
cargo build --release
```

### 2. Run Examples

```bash
cargo run --example one_shot
cargo run --example read_file
```

## Architecture

Nucleus is structured as a workspace with clear separation:

- **nucleus**: the library. Chat, providers, retrieval, and built-in tools live here.
- **nucleus-cli**: `login` and `logout` for a Grok account

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
```

### Vector Database (for RAG)

Nucleus uses embedded LanceDB for persistent storage in RAG (Retrieval Augmented Generation):

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
