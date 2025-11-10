# RAG Setup Guide

## What is RAG?

**RAG (Retrieval Augmented Generation)** allows your LLM to answer questions based on YOUR data, not just its training.

## How It Works

```
1. INDEXING (one-time setup)
   Your code → Split into chunks → Convert to vectors → Store in DB

2. QUERY (every time you ask)
   Question → Convert to vector → Find similar chunks → Add to prompt → LLM answers

3. RESULT
   LLM answers using YOUR actual code/docs as context
```

## Setup Complete! ✅

You now have:
- ✅ Qwen 2.5 Coder 32B (main model)
- ✅ Nomic Embed Text (embedding model)
- ✅ ChromemGo vector database
- ✅ Automatic retrieval on every query

## How to Use

### 1. Start the App

```bash
make run
```

### 2. Index Your Code

Index a project directory:
```
> /index /Users/cooksey/becon
```

This will:
- Find all `.go`, `.py`, `.js`, `.ts`, `.md` files
- Split them into 512-character chunks (with 50-char overlap)
- Generate embeddings for each chunk
- Store in vector database

### 3. Ask Questions

Now your questions will automatically search your indexed code:

```
> How does the authentication work in my project?
```

The system will:
1. Convert your question to an embedding
2. Search for the 5 most similar code chunks
3. Add those chunks to the prompt
4. LLM answers based on YOUR code

### 4. Add Knowledge Manually

```
> /add My API uses JWT tokens with RS256 algorithm
> /add The database connection pool size is 20
```

### 5. Check Stats

```
> /stats
Knowledge base contains 847 documents
```

## Configuration

Edit `config.yaml` to customize:

```yaml
rag:
  embedding_model: "nomic-embed-text"
  chunk_size: 512              # Characters per chunk
  chunk_overlap: 50            # Overlap between chunks
  top_k: 5                     # Number of results to retrieve
```

## Example Session

```bash
$ make run
🤖 Local LLM with RAG Ready!
Model: qwen2.5-coder:32b
Knowledge Base: 0 documents

> /index /Users/cooksey/my-project
📚 Indexing directory: /Users/cooksey/my-project
✓ Indexed: main.go (3 chunks)
✓ Indexed: handlers.go (5 chunks)
✓ Indexed: README.md (2 chunks)

✅ Indexed 3 files

> How does the authentication handler work?

[System retrieves relevant chunks from handlers.go]

The authentication handler uses JWT tokens. It validates the token
from the Authorization header, checks expiration, and extracts the
user ID. Here's the code:

[Shows actual code from your project]

> /quit
Goodbye!
```

## Advanced Usage

### Index Multiple Projects

```
> /index /Users/cooksey/project1
> /index /Users/cooksey/project2
```

All indexed code is searchable together!

### Add Documentation

```
> /add Our API follows REST principles. POST /auth/login returns JWT token
> /add Database uses PostgreSQL 15 with connection pooling
```

### Supported File Types

Currently indexes:
- `.go` - Go
- `.py` - Python
- `.js` - JavaScript
- `.ts` - TypeScript
- `.md` - Markdown

Want more? Edit line 210 in `main.go`.

## How RAG Improves Answers

**Without RAG:**
```
> What port does my API use?
< I don't have information about your specific API configuration.
```

**With RAG (after indexing):**
```
> What port does my API use?
< Your API uses port 8080, as configured in main.go line 45.
```

## Storage

Vector database is stored at:
```
./data/vectordb/
```

To backup your knowledge:
```bash
cp -r data/ backup/
```

To start fresh:
```bash
rm -rf data/
```

## Performance

- Indexing: ~1-2 seconds per file
- Embedding generation: ~50-200ms per query
- Retrieval: ~10-50ms per query
- Total overhead: ~100-300ms (negligible compared to LLM generation)

## Troubleshooting

### "Embedding failed"
Make sure nomic-embed-text is pulled:
```bash
ollama pull nomic-embed-text
```

### "Directory does not exist"
Use absolute paths:
```
> /index /Users/cooksey/project
```

### Slow indexing
For large codebases (>1000 files), this is normal. Indexing happens once.

## Next Steps

1. Index your main projects
2. Add project-specific knowledge with `/add`
3. Ask questions and watch RAG retrieve relevant context
4. Adjust `chunk_size` and `top_k` in config for better results
