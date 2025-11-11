# LLM Workspace

> **Work in Progress** - This project is in early development

A privacy-first AI terminal assistant that runs entirely on local or self-hosted LLMs. Built as a PTY wrapper, it brings intelligent assistance directly into your terminal workflow without sending your data to external services.

![Demo](demo.gif)

## Current Features (Partially Working)

- **Standalone AI Server** - Local LLM server powered by Ollama
- **PTY Terminal Wrapper** - AI-enhanced terminal session
- **Interactive AI Chat** - Ask questions and get help without leaving your terminal
- **RAG (Retrieval Augmented Generation)** - Context-aware responses from your codebase
- **File Operations** - AI-assisted file editing and command execution

## Planned Features

- **Terminal Auto-Prediction** - Intelligent command suggestions as you type
- **Personalized AI** - Learns your coding style and preferences
- **Context-Aware Assistance** - Understands your project structure and history
- **Optional Server Integration** - Self-hosted remote capabilities (local-first priority)
- **Zero Data Leakage** - Complete privacy with local-only LLM execution

## Build

```bash
make build
```

This builds:
- `llm-server` - Go AI server (connects to Ollama)
- `llm-workspace` - Rust PTY wrapper

## Run

```bash
./llm-workspace
```

Or:

```bash
make run
```

The PTY will automatically start the AI server in the background.

## AI Commands

Once in the PTY, use these commands:

- `/ai <question>` - Chat with AI
- `/edit <request>` - Use AI with file editing capabilities
- `/add <text>` - Add knowledge to vector database
- `/index <path>` - Index a directory for RAG
- `/stats` - Show knowledge base statistics

## Standalone AI Server

Run the AI server in interactive mode:

```bash
go run ai/server.go interactive
```

## Install

Install to `/usr/local/bin`:

```bash
make install
```

Then run from anywhere:

```bash
llm-workspace
```

## Configuration

Edit `ai/config.yaml` to configure:
- Model selection
- RAG settings
- File operation preferences

## Development

```bash
make dev    # Quick dev build
make clean  # Clean all artifacts
```

## Logs

Server logs: `/tmp/llm-workspace.log`
