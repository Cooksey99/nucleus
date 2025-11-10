# LLM Workspace

A local LLM-powered assistant with RAG (Retrieval-Augmented Generation) capabilities for code understanding and file operations.

## Features

- Local LLM inference using Ollama
- Vector database for knowledge storage (RAG)
- Code indexing and semantic search
- AI-powered file editing
- Interactive chat interface with context awareness

## Prerequisites

- macOS (tested on latest versions)
- Go 1.24.3 or higher
- Homebrew (for automatic dependency installation)

## Quick Start

```bash
# Clone the repository
git clone <your-repo-url>
cd llm-workspace

# Run automated setup (installs all dependencies including Ollama)
make setup

# Start the application
make run
```

The `make setup` command will automatically:
1. Check for and install Ollama (via Homebrew if not present)
2. Install Go dependencies
3. Pull the required LLM and embedding models from config.yaml
4. Start Ollama service if not running

## Manual Installation

If you prefer to install dependencies manually:

### 1. Install Ollama

```bash
# Using Homebrew
brew install ollama

# Start Ollama service
ollama serve &
```

Or download directly from [ollama.ai](https://ollama.ai)

### 2. Install Go Dependencies

```bash
make install-deps
```

### 3. Pull Models

```bash
# Pull both LLM and embedding models from config.yaml
make pull-all-models

# Or pull individually
make pull-model          # Pulls the LLM model
make pull-embedding-model # Pulls the embedding model

# Or manually pull a different model
ollama pull qwen2.5-coder:32b
```

### 4. Run the Application

```bash
make run
```

## Configuration

Edit `config.yaml` to customize:

```yaml
llm:
  model: "qwen2.5-coder:32b"      # LLM model to use
  base_url: "http://localhost:11434"
  temperature: 0.7
  context_length: 8192

rag:
  embedding_model: "nomic-embed-text"  # For vector embeddings
  chunk_size: 512
  chunk_overlap: 50
  top_k: 5                         # Number of relevant docs to retrieve

storage:
  vector_db_path: "./data/vectordb"
  chat_history_path: "./data/history"
```

## Usage

Once running, you can use these commands:

### Chat Commands

- **Chat normally**: Just type your question
- `/add <text>` - Add knowledge to the vector database
- `/index <path>` - Index a directory (recursively scans code files)
- `/edit <request>` - Enable file editing mode with AI assistance
- `/stats` - Show knowledge base statistics
- `/quit` - Exit the application

### Examples

```
> /index ./myproject
📚 Indexing directory: ./myproject

> What functions are defined in auth.go?
[AI responds with context from indexed files]

> /edit Refactor the login function to use better error handling
🔧 File editing mode enabled
[AI analyzes and suggests/makes changes]
```

## Available Models

The application supports any Ollama-compatible model. Popular choices:

- `qwen2.5-coder:14b` - Fast, good for most tasks
- `qwen2.5-coder:32b` - More capable, requires more resources
- `codellama:13b` - Alternative coding model
- `deepseek-coder:6.7b` - Lightweight option

Pull a model with:
```bash
ollama pull <model-name>
```

## Make Commands

```bash
make setup               # Complete first-time setup
make install-deps        # Install Go dependencies only
make pull-model          # Pull the LLM model from config.yaml
make pull-embedding-model # Pull the embedding model from config.yaml
make pull-all-models     # Pull both LLM and embedding models
make run                 # Run the application
make build               # Build binary (creates ./llm-app)
make clean               # Remove built files only (SAFE - keeps your data)
make clean-data          # Remove vector DB and chat history (DESTRUCTIVE)
make clean-all           # Remove everything including data (DESTRUCTIVE)
make test                # Run tests
make help                # Show all commands
```

## Troubleshooting

### Ollama not found
```bash
# Install via Homebrew
brew install ollama

# Or install manually from ollama.ai
```

### Ollama service not running
```bash
# Start the service
ollama serve &

# Or check if it's already running
ps aux | grep ollama
```

### Model not available
```bash
# Pull models from your config.yaml
make pull-all-models

# Or pull a specific model manually
ollama pull qwen2.5-coder:14b
```

### Port 11434 already in use
```bash
# Check what's using the port
lsof -i :11434

# Kill the process or configure a different port in config.yaml
```

## Project Structure

```
.
├── config/          # Configuration management
├── data/            # Vector DB and chat history (auto-created)
├── docs/            # Documentation
├── fileops/         # File operations and AI editing
├── rag/             # RAG implementation
├── config.yaml      # Main configuration
├── main.go          # Entry point
├── Makefile         # Build and setup automation
└── README.md        # This file
```
