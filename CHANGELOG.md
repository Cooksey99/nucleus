# Changelog

All notable changes to this project are documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
0.2 is a rewrite; see [ROADMAP.md](ROADMAP.md) for the plan.

## [Unreleased]

### Added
- CI workflow (`.github/workflows/ci.yml`): build, test and doctest on Linux and macOS, plus a clippy report. Clippy warnings don't fail the build yet.

### Fixed
- Examples that called the async `PluginRegistry::register` without awaiting it, so no plugin was registered (`read_file_line`, `write_file`, and the `nucleus-std` `read_file` and `search_text` examples, which also failed to compile).
- Registry tests that never registered a plugin: one always failed, the other passed without testing anything.
- Doc examples in `nucleus-core` that no longer compiled against the current `ChatManagerBuilder`, `RagEngine` and `MistralRsProvider` APIs.
