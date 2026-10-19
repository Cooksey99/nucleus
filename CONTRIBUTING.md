# Contributing to Nucleus

Thank you for your interest in contributing to Nucleus! This document provides guidelines for contributing to the project.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/YOUR_USERNAME/nucleus.git`
3. Create a feature branch: `git checkout -b feature/your-feature-name`
4. Make your changes
5. Run tests: `cargo test --workspace`
6. Commit your changes: `git commit -m "Add your feature"`
7. Push to your fork: `git push origin feature/your-feature-name`
8. Open a Pull Request

## Development Setup

### Prerequisites

- Rust (latest stable)
- `protoc`, the Protocol Buffers compiler, needed to build LanceDB (`brew install protobuf` or `apt install protobuf-compiler`)

### Building

```bash
cargo build
```

### Testing

These are the commands CI runs (see `.github/workflows/ci.yml`):

```bash
cargo build --workspace --all-targets
cargo test --workspace --all-targets
cargo test --workspace --doc
cargo clippy --workspace --all-targets
```

### Running Examples

```bash
cargo run --example read_file_line
```

## Project Structure

Nucleus is a Cargo workspace with these crates:

- **nucleus**: the library. Chat, providers, retrieval, and built-in tools.
- **nucleus-cli**: `login` and `logout` for a Grok account

## Code Guidelines

### Style

- Follow standard Rust formatting: `cargo fmt`
- Run clippy: `cargo clippy -- -D warnings`
- Write idiomatic Rust code

### Documentation

- Document all public APIs with doc comments
- Include examples in doc comments where helpful
- Update README.md if adding user-facing features

### Testing

- Write unit tests for new functionality
- Add integration tests for cross-crate features
- Ensure all tests pass before submitting PR

## What to Contribute

### Areas We Need Help With

- **Plugin Development**: New tools for the plugin system
- **Backend Support**: Local inference is mistral.rs. Hosted providers are added behind the same provider trait.
- **Documentation**: Tutorials, guides, and API docs
- **Testing**: Improve test coverage
- **Examples**: Real-world usage examples
- **Performance**: Optimize hot paths

### Before Starting Large Changes

For significant changes, please open an issue first to discuss:
- New features or architectural changes
- Breaking API changes
- Large refactors

This ensures your work aligns with project goals and avoids duplicate effort.

## Plugin Development

If you're building a plugin:

1. Implement the `Plugin` trait in `nucleus::tool`
2. Handle permissions appropriately
3. Write comprehensive tests
4. Document expected behavior and limitations
5. Consider adding it to `nucleus::tools` if generally useful

Example plugin structure:

```rust
use nucleus_plugin::{Plugin, PluginResult, Context};
use async_trait::async_trait;

pub struct MyPlugin;

#[async_trait]
impl Plugin for MyPlugin {
    fn name(&self) -> &str {
        "my_plugin"
    }
    
    fn description(&self) -> &str {
        "Does something useful"
    }
    
    async fn execute(&self, ctx: &Context) -> PluginResult {
        // Implementation
    }
}
```

## Pull Request Process

1. Ensure your code builds and all tests pass
2. Update documentation if needed
3. Add tests for new functionality
4. Keep PRs focused on a single feature/fix

### PR Review Criteria

- Code quality and style
- Test coverage
- Documentation completeness
- Backward compatibility
- Performance impact

## Privacy and Security

Nucleus is privacy-first by design. When contributing:

- **Never add telemetry or external network calls** (except to user-configured LLM backends or optional plugins)
- Be cautious with file system operations
- Implement proper permission checks for plugins
- Consider security implications of LLM-driven actions

## License

By contributing to Nucleus, you agree that your contributions will be licensed under the MIT License.

## Questions?

- Open an issue for bugs or feature requests
- Start a discussion for questions or ideas
- Check existing issues before creating new ones

## Code of Conduct

Be respectful and constructive. We're building this together.
