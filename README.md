# Local LLM with RAG - Portable Setup

A lightweight, portable local LLM environment with RAG (Retrieval Augmented Generation) capabilities. Built with Go for easy deployment across ARM-based systems.

## Features

- Local LLM powered by Ollama (Qwen 2.5 Coder 32B)
- RAG with vector database (chromem-go)
- Persistent conversation history
- Learns from your interactions
- Single binary - no runtime dependencies
- Easy to port to new machines

## Quick Start

### First Time Setup

```bash
chmod +x setup.sh
./setup.sh
```

This will:
1. Install Ollama (if needed)
2. Pull the Qwen 2.5 Coder 14B model
3. Install Go dependencies
4. Initialize the project

### Run the Application

```bash
go run main.go
```

Or build a binary:

```bash
go build -o llm-app
./llm-app
```

## Moving to a New Machine

1. Copy this entire directory
2. Run `./setup.sh`
3. Your data directory will preserve knowledge and history

Or use Git:

```bash
git init
git add .
git commit -m "Initial LLM setup"
git push origin main

# On new machine:
git clone <your-repo>
cd llm-workspace
./setup.sh
```

## Configuration

Edit `config.yaml` to customize:

- **Model**: Change to any Ollama model
- **System Prompt**: Customize AI behavior
- **RAG Settings**: Adjust chunk size, top-k results
- **Storage Paths**: Change where data is stored

## Usage

### Basic Chat

```
> What's the difference between channels and mutexes in Go?
```

### Add Knowledge

```
> /add My project uses Cobra for CLI and stores data in SQLite
```

The system will remember this for future queries.

### Commands

- `/add <text>` - Add information to knowledge base
- `/quit` - Exit application

## Project Structure

```
llm-workspace/
├── setup.sh           # Installation script
├── config.yaml        # Configuration
├── main.go           # Main application
├── go.mod            # Go dependencies
├── README.md         # This file
└── data/             # Generated at runtime
    ├── vectordb/     # RAG knowledge base
    ├── history/      # Chat history
    └── preferences.json
```

## Customization

### Change Model

```yaml
llm:
  model: "llama3.1:8b"  # or "qwen2.5-coder:32b", etc.
```

Then: `ollama pull llama3.1:8b`

### Add Embedding Model for Better RAG

```bash
ollama pull nomic-embed-text
```

Update `config.yaml`:
```yaml
rag:
  embedding_model: "nomic-embed-text"
```

## Requirements

- Go 1.21+
- Ollama
- macOS (ARM) or Linux (ARM)

