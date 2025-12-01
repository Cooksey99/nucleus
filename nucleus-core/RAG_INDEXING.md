# RAG Indexing and Persistence Guide

This guide explains how to index files and directories into the RAG vector database and configure persistent storage.

## Overview

The RAG system now supports:
- **Persistent storage**: Vector database saved to disk
- **Batch indexing**: Index multiple directories at once
- **File-level indexing**: Index individual files
- **Configurable storage path**: Specify where to store the vector database
- **Auto-save**: Automatically saves after each indexing operation
- **Auto-load**: Loads previously indexed documents on startup

## Quick Start

### 1. Configure Storage Location

By default, the vector database is stored in `./data/vectordb/`. You can customize this in your `config.yaml`:

```yaml
storage:
  vector_db_path: "./my_custom_path"
  chat_history_path: "./data/history"
```

Or programmatically:

```rust
use nucleus_core::Config;

let mut config = Config::default();
config.storage.vector_db_path = "/path/to/storage".to_string();
```

### 2. Create a Manager with Persistence

```rust
use nucleus_core::{Config, rag::Manager, ollama::Client};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = Config::default();
    let client = Client::new(&config.llm.base_url);
    
    // Create manager with persistent storage
    let manager = Manager::with_persistence(&config, client);
    
    // Load previously indexed documents
    let loaded = manager.load().await?;
    println!("Loaded {} documents", loaded);
    
    Ok(())
}
```

### 3. Index Content

#### Index a Single Directory

```rust
// Recursively indexes all files in the directory
let count = manager.index_directory("./src").await?;
println!("Indexed {} files", count);
```

#### Index Multiple Directories

```rust
let dirs = vec!["./src", "./docs", "./examples"];
let total = manager.index_directories(&dirs).await?;
println!("Total files indexed: {}", total);
```

#### Index a Specific File

```rust
let chunks = manager.index_file("./README.md").await?;
println!("Created {} chunks", chunks);
```

### 4. Save Manually (Optional)

The vector store auto-saves after each indexing operation, but you can also save manually:

```rust
manager.save().await?;
```

## Configuration

### Indexer Configuration

Control which files are indexed by configuring extensions and exclude patterns:

```yaml
rag:
  embedding_model: "nomic-embed-text"
  chunk_size: 512
  chunk_overlap: 50
  top_k: 5
  indexer:
    # Only index these file types (empty = all text files)
    extensions: ["rs", "md", "toml", "py", "js", "ts"]
    
    # Exclude these patterns
    exclude_patterns:
      - "node_modules"
      - ".git"
      - "target"
      - "dist"
      - "build"
```

Or programmatically:

```rust
use nucleus_core::{Config, IndexerConfig};

let mut config = Config::default();
config.rag.indexer = IndexerConfig {
    extensions: vec!["rs".into(), "py".into()],
    exclude_patterns: vec!["target".into(), "__pycache__".into()],
};
```

## Advanced Usage

### Using the Server API

The Server API provides high-level methods with automatic persistence:

```rust
use nucleus_core::{Config, Server};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = Config::default();
    
    // Create server with persistence (auto-loads existing data)
    let server = Server::new_with_persistence(config).await?;
    
    // Now you can use the IPC server with persistent RAG
    server.start().await?;
    
    Ok(())
}
```

### Finding Subdirectories to Index

Use the utility functions to discover directories:

```rust
use nucleus_core::rag::utils;

// Find all subdirectories up to 2 levels deep
let dirs = utils::find_subdirectories("./workspace", 2).await?;

for dir in dirs {
    if utils::contains_indexable_files(&dir, &["rs".into()]).await {
        println!("Indexing: {:?}", dir);
        manager.index_directory(dir.to_str().unwrap()).await?;
    }
}
```

### Retrieving Context

After indexing, query the knowledge base:

```rust
let query = "How do I implement authentication?";
let context = manager.retrieve_context(query).await?;

if !context.is_empty() {
    println!("Found relevant context:");
    println!("{}", context);
}
```

## Storage Format

The vector database is stored as JSON in `<storage_path>/vector_store.json`:

```json
{
  "documents": [
    {
      "id": "src/main.rs_chunk_0",
      "content": "...",
      "embedding": [0.1, 0.2, ...],
      "metadata": {
        "source": "src/main.rs",
        "chunk": "0"
      }
    }
  ],
  "version": 1
}
```

## API Reference

### Manager Methods

- `Manager::with_persistence(config, client)` - Create manager with persistent storage
- `load() -> Result<usize>` - Load documents from disk
- `save() -> Result<()>` - Save documents to disk
- `index_directory(path) -> Result<usize>` - Index all files in directory
- `index_directories(paths) -> Result<usize>` - Index multiple directories
- `index_file(path) -> Result<usize>` - Index a single file
- `add_knowledge(content, source) -> Result<()>` - Add text directly
- `retrieve_context(query) -> Result<String>` - Query for relevant context
- `count() -> usize` - Get document count
- `clear()` - Remove all documents

### Utility Functions

- `utils::find_subdirectories(dir, depth) -> Result<Vec<PathBuf>>` - Find subdirectories
- `utils::contains_indexable_files(dir, extensions) -> bool` - Check if directory has indexable files
- `utils::get_relative_path(base, full) -> PathBuf` - Get relative path

## Example: Complete Workflow

```rust
use nucleus_core::{Config, rag::Manager, ollama::Client};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Setup
    let config = Config::default();
    let client = Client::new(&config.llm.base_url);
    let manager = Manager::with_persistence(&config, client);
    
    // 2. Load existing data
    let loaded = manager.load().await?;
    println!("Loaded {} existing documents", loaded);
    
    // 3. Index new content
    println!("\nIndexing project files...");
    let count = manager.index_directory("./src").await?;
    println!("✓ Indexed {} files", count);
    
    // 4. Query
    println!("\nQuerying knowledge base...");
    let context = manager.retrieve_context("error handling").await?;
    
    if !context.is_empty() {
        println!("Found relevant context!");
    }
    
    // 5. Check stats
    println!("\nTotal documents: {}", manager.count());
    println!("Storage location: {}", config.storage.vector_db_path);
    
    Ok(())
}
```

## Performance Considerations

- **Chunk Size**: Default 512 bytes. Larger chunks = more context but less granular retrieval
- **Overlap**: Default 50 bytes. More overlap = better boundary coverage but more storage
- **Storage**: JSON format is human-readable but not space-efficient. Consider compression for large datasets
- **In-Memory**: All documents are kept in memory. For very large datasets (>100k documents), consider external vector databases

## Migration from In-Memory

If you're upgrading from the in-memory version:

1. Replace `Manager::new()` with `Manager::with_persistence()`
2. Call `load()` after creating the manager
3. That's it! Auto-save handles persistence automatically

```rust
// Old (in-memory)
let manager = Manager::new(&config, client);

// New (persistent)
let manager = Manager::with_persistence(&config, client);
manager.load().await?;
```

## Troubleshooting

### Documents not persisting
- Ensure the storage directory is writable
- Check that `with_persistence()` was used instead of `new()`
- Verify the storage path in config is correct

### Large file sizes
- The JSON format is verbose. This is by design for now
- Future versions may support binary formats or compression

### Slow indexing
- Embedding generation is the bottleneck
- Consider indexing in batches during off-hours
- Use more specific file extensions to reduce file count
