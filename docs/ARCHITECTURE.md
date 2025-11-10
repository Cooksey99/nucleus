# Architecture

The codebase is organized into clean, modular packages for maintainability.

## Project Structure

```
llm-workspace/
├── main.go              # CLI interface and orchestration
├── config/
│   └── config.go        # Configuration loading and types
├── rag/
│   └── rag.go           # RAG: indexing, retrieval, embeddings
├── fileops/
│   └── fileops.go       # File operations and tool calling
├── config.yaml          # User settings
├── Makefile             # Build and setup commands
└── data/                # Runtime data (gitignored)
    ├── vectordb/        # Vector embeddings
    └── history/         # Chat history
```

## Package Responsibilities

### `config/`
- Load and parse `config.yaml`
- Define all configuration structs
- **Modify this** to add new settings

### `rag/`
- Manage vector database
- Generate embeddings via Ollama
- Index directories and files  
- Retrieve relevant context for queries
- **Modify this** to improve RAG/indexing

### `fileops/`
- Read and write files
- Tool calling (function calling for LLM)
- Chat interface with and without tools
- **Modify this** to add new file operations or tools

### `main.go`
- CLI interface only
- Wire up packages
- Handle user commands
- **Modify this** to add new CLI commands

## Adding New Features

### Add a new CLI command
Edit `main.go` → Add case in the input loop

### Add a new configuration option
1. Edit `config/config.go` → Add field to struct
2. Edit `config.yaml` → Add the setting
3. Use it in the relevant package

### Improve RAG
Edit `rag/rag.go` → Modify indexing, chunking, or retrieval logic

### Add a new tool for the LLM
Edit `fileops/fileops.go` → Add to `tools` array in `ChatWithTools()`

## How It Works

1. **Startup**:
   - `main.go` loads config
   - Creates Ollama client
   - Initializes RAG manager
   - Initializes file operations manager

2. **User Query**:
   - User types message
   - `main.go` routes to appropriate handler
   - For regular chat: `fileops.Chat()` → retrieves context → LLM
   - For edits: `fileops.ChatWithTools()` → LLM can call read_file/write_file

3. **RAG Flow**:
   - Query → embedding → vector search → top-K chunks
   - Chunks appended to prompt
   - LLM generates answer with context

## Dependencies

- `github.com/ollama/ollama/api` - LLM client
- `github.com/philippgille/chromem-go` - Vector database
- `gopkg.in/yaml.v3` - Config parsing

## Data Flow

```
User Input
    ↓
main.go (routing)
    ↓
fileops.Chat() or fileops.ChatWithTools()
    ↓
rag.RetrieveContext() (retrieves relevant chunks)
    ↓
Ollama API (LLM generation)
    ↓
Response to user
```

## Best Practices

1. **Keep packages focused** - Each package does one thing well
2. **Configuration in config.yaml** - No hardcoded settings
3. **Errors bubble up** - Let main.go handle error display
4. **Tool calling is optional** - Regular chat doesn't need tools
