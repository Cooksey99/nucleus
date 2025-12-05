# RAG Database Scaling and Performance

## Overview

This document explains the capacity limits of the current RAG implementation and provides detailed guidance on scaling to handle large datasets with optimal performance.

## Current Implementation Capacity

### Files vs Documents
- **Files**: The number of source files indexed
- **Documents/Chunks**: Each file is split into 512-byte chunks (default)
  - Example: 18 files → 210 documents
  - Average: ~10-15 chunks per file

### Real-World Reference (nucleus-core/src)
**Actual indexed data from this project:**
- **Source files**: 18 files totaling **133 KB** (0.13 MB)
- **Indexed documents**: 210 chunks
- **Vector database size**: **3.5 MB** (JSON format)
- **Storage expansion**: ~26x (133 KB → 3.5 MB)
  - Raw text: 133 KB
  - + Embeddings: ~630 KB (210 docs × 3 KB each)
  - + JSON overhead: ~2.7 MB (formatting, field names)
- **Compression potential**: With binary format, could reduce to ~0.8-1 MB

**Key insight**: The vector database is ~26x larger than the source code due to embeddings and JSON formatting.

### Storage Per Document
Each document in the vector database stores:
- **Text content**: ~512 bytes (configurable via `chunk_size`)
- **Embedding vector**: 768 dimensions × 4 bytes = ~3 KB (for `nomic-embed-text`)
- **Metadata**: Source path, chunk index, custom metadata (~100-200 bytes)
- **JSON overhead**: Field names, formatting (~500 bytes in current format)

**Total per document**: ~4-5 KB in memory, ~16-17 KB on disk (JSON)

## Practical Limits

### Memory Usage (In-Memory Storage)

| Documents | Source Code | Memory (RAM) | Disk (JSON) | Status | Use Case |
|-----------|-------------|--------------|-------------|--------|----------|
| 210 | 133 KB | ~1 MB | 3.5 MB | ✅ Current | Small module |
| 1,000 | ~0.6 MB | ~5 MB | ~17 MB | ✅ Excellent | Small project |
| 10,000 | ~6 MB | ~50 MB | ~170 MB | ✅ Great | Medium project |
| 50,000 | ~30 MB | ~250 MB | ~850 MB | ✅ Good | Large monorepo |
| 100,000 | ~60 MB | ~500 MB | ~1.7 GB | ⚠️ Okay | Multiple projects |
| 500,000 | ~300 MB | ~2.5 GB | ~8.5 GB | ⚠️ Challenging | Enterprise |
| 1,000,000+ | ~600+ MB | ~5+ GB | ~17+ GB | ❌ Not recommended | Needs external DB |

### Search Performance (Linear Scan - O(n))

Current implementation uses **cosine similarity** with full linear scan:

```rust
// Current: O(n) - checks every document
for doc in all_documents {
    score = cosine_similarity(query_embedding, doc.embedding);
}
```

| Documents | Search Time | User Experience | Notes |
|-----------|-------------|-----------------|-------|
| 210 | <1 ms | ✅ Instant | Current system |
| 1,000 | <10 ms | ✅ Instant | Imperceptible delay |
| 10,000 | 50-100 ms | ✅ Fast | Barely noticeable |
| 50,000 | 250-500 ms | ⚠️ Noticeable | Slight pause |
| 100,000 | 0.5-1 s | ⚠️ Slow | User notices wait |
| 500,000 | 2.5-5 s | ❌ Too slow | Frustrating |
| 1,000,000+ | 5-10+ s | ❌ Unacceptable | Need optimization |

### Disk Storage (JSON Format)

| Documents | File Size | Load Time | Save Time |
|-----------|-----------|-----------|-----------|
| 210 | 3.5 MB | <50 ms | <100 ms |
| 1,000 | ~17 MB | <200 ms | <300 ms |
| 10,000 | ~170 MB | 1-2 s | 2-3 s |
| 100,000 | ~1.7 GB | 10-20 s | 20-30 s |
| 1,000,000+ | ~17+ GB | Minutes | Minutes |

## Recommended Operational Limits

### Current Implementation
- **Optimal**: 10,000-50,000 documents (~6-30 MB source code)
- **Maximum comfortable**: 100,000 documents (~60 MB source code)
- **Beyond**: Requires optimization or external solution

### Real-World Examples
- **Single large project** (Linux kernel size): ~50,000-100,000 documents
- **Company monorepo** (Google/Meta scale): 1,000,000+ documents
- **Personal projects** (typical dev): 1,000-10,000 documents
- **This project** (nucleus-core/src): 210 documents from 133 KB

## Scaling Strategies

### Level 1: Optimize Current Implementation (10K → 100K documents)

#### 1.1 Binary Storage Format
Replace JSON with binary serialization:

```rust
// Instead of JSON, use bincode
use bincode;

pub async fn save_to_disk_binary(documents: &[Document], path: impl AsRef<Path>) -> Result<()> {
    let binary = bincode::serialize(documents)?;
    fs::write(path, binary).await?;
    Ok(())
}
```

**Benefits**:
- **5-10x smaller file size** (3.5 MB → 0.4-0.7 MB for our example)
- **10-20x faster load/save times**
- Same memory usage

**Estimated gains for our example**:
- 210 docs: 3.5 MB → 0.5 MB on disk
- Load time: 50ms → 5ms

**Estimated gains at scale**:
- 100,000 docs: 1.7 GB → 200-300 MB on disk
- Load time: 10-20s → 0.5-1s

#### 1.2 Memory-Mapped Files
Use `memmap2` for lazy loading:

```rust
use memmap2::MmapOptions;

// Map file to memory, OS handles paging
let file = File::open(path)?;
let mmap = unsafe { MmapOptions::new().map(&file)? };
let documents: &[Document] = bincode::deserialize(&mmap)?;
```

**Benefits**:
- Reduced startup time (no upfront load)
- OS manages memory pressure
- Scales to disk size, not RAM size

#### 1.3 Compression
Add gzip/zstd compression:

```rust
use flate2::write::GzEncoder;

// Compress before writing
let mut encoder = GzEncoder::new(file, Compression::default());
encoder.write_all(&binary_data)?;
```

**Benefits**:
- 3-5x smaller disk usage
- Slightly slower load/save (acceptable tradeoff)

**Our example**:
- 3.5 MB (JSON) → 0.5 MB (binary) → 0.15 MB (binary + gzip)
- ~23x total compression

**Implementation**:
```toml
# Cargo.toml
[dependencies]
flate2 = "1.0"
bincode = "1.3"
memmap2 = "0.9"
```

### Level 2: Algorithmic Optimization (100K → 1M documents)

#### 2.1 HNSW (Hierarchical Navigable Small World)
Implement approximate nearest neighbor (ANN) search:

**Algorithm**: O(log n) instead of O(n)

```rust
use hnsw_rs::prelude::*;

// Build HNSW index
let mut hnsw = Hnsw::<f32, DistCosine>::new(
    16,  // max connections per layer
    768, // embedding dimension
    16,  // ef_construction
    200, // max elements
    DistCosine,
);

// Insert embeddings
for (id, embedding) in embeddings.iter().enumerate() {
    hnsw.insert((embedding, id));
}

// Search (much faster!)
let results = hnsw.search(&query_embedding, 5, 50);
```

**Performance improvement**:
- 210 docs: <1ms → <1ms (no benefit at small scale)
- 10,000 docs: 100ms → 2-5ms (20-50x faster)
- 100,000 docs: 500ms → 5-10ms (50-100x faster)
- 1,000,000 docs: 5s → 10-20ms (250-500x faster)

**Tradeoffs**:
- ~95-99% accuracy (vs 100% with linear scan)
- Additional memory for graph structure (~20-30% overhead)
- Complex implementation

**Dependencies**:
```toml
[dependencies]
hnsw_rs = "0.3"
```

#### 2.2 Product Quantization (PQ)
Compress embedding vectors:

```rust
// Instead of 768 floats (3KB), use quantized codes
// Reduces to ~96 bytes (32x compression)
struct QuantizedEmbedding {
    codes: [u8; 96], // 768 dimensions → 96 bytes
}
```

**Benefits**:
- 10-32x memory reduction for embeddings
- Faster similarity computation
- Slight accuracy loss (1-2%)

**Our example**:
- 210 docs: 630 KB embeddings → 20 KB (31x compression)
- 100,000 docs: 300 MB embeddings → 10 MB

**Best for**: Very large datasets where memory is primary constraint

#### 2.3 Inverted File Index (IVF)
Cluster embeddings, search only relevant clusters:

```rust
// 1. Cluster embeddings into groups
let clusters = kmeans(&all_embeddings, num_clusters: 100);

// 2. Search: find nearest clusters, search only those
let relevant_clusters = find_nearest_clusters(&query, k: 10);
let candidates = get_documents_in_clusters(&relevant_clusters);
let results = search_within(candidates, query);
```

**Benefits**:
- 10-50x faster search
- Configurable accuracy vs speed tradeoff

**Best for**: When you can tolerate searching 10% of data instead of 100%

### Level 3: External Vector Database (1M+ documents)

For production-scale applications, integrate purpose-built vector databases:

#### 3.1 Qdrant (Recommended for Rust)
**Pros**:
- Written in Rust (excellent performance)
- Built-in HNSW indexing
- Filtering and payload support
- Easy to embed or run standalone

**Integration**:
```rust
use qdrant_client::{prelude::*, qdrant::vectors_config::Config};

// Connect to Qdrant
let client = QdrantClient::from_url("http://localhost:6334").build()?;

// Create collection
client.create_collection(&CreateCollection {
    collection_name: "nucleus_rag".to_string(),
    vectors_config: Some(VectorsConfig {
        config: Some(Config::Params(VectorParams {
            size: 768,
            distance: Distance::Cosine.into(),
        })),
    }),
    ..Default::default()
}).await?;

// Insert documents
for doc in documents {
    client.upsert_points_blocking(
        "nucleus_rag",
        vec![PointStruct::new(
            doc.id.clone(),
            doc.embedding.clone(),
            doc.to_payload(),
        )],
    ).await?;
}

// Search
let results = client.search_points(&SearchPoints {
    collection_name: "nucleus_rag".to_string(),
    vector: query_embedding,
    limit: 5,
    ..Default::default()
}).await?;
```

**Capacity**: Billions of vectors
**Performance**: <10ms for millions of documents

#### 3.2 Milvus
**Pros**:
- Production-grade
- Supports multiple index types (HNSW, IVF, etc.)
- Distributed architecture
- Good for very large scale

**Best for**: Enterprise deployments with millions+ documents

#### 3.3 ChromaDB
**Pros**:
- Simple API
- Good Python integration
- Embedded or client-server

**Best for**: Python-heavy environments

#### 3.4 Weaviate
**Pros**:
- GraphQL API
- Built-in hybrid search
- Good for complex queries

**Best for**: When you need both vector and structured search

### Level 4: Distributed Architecture (10M+ documents)

#### 4.1 Sharding
Split documents across multiple databases:

```rust
// Hash document ID to determine shard
fn get_shard(doc_id: &str, num_shards: usize) -> usize {
    let hash = hash(doc_id);
    hash % num_shards
}

// Search all shards in parallel
let results: Vec<_> = shards.par_iter()
    .map(|shard| shard.search(query))
    .flatten()
    .collect();
```

**Benefits**:
- Near-linear scaling with shard count
- Can distribute across machines

#### 4.2 Hierarchical Indexing
Multi-level search:

```rust
// Level 1: Directory-level summaries (fast)
let relevant_dirs = search_directory_summaries(query);

// Level 2: File-level within relevant dirs (medium)
let relevant_files = search_files_in_dirs(relevant_dirs, query);

// Level 3: Chunk-level within relevant files (precise)
let results = search_chunks_in_files(relevant_files, query);
```

**Benefits**:
- Dramatically reduces search space
- Better for very large codebases

## Migration Path

### Phase 1: Current → 100K documents
**Goal**: Handle medium to large projects efficiently

1. Implement binary serialization (bincode)
2. Add optional compression (gzip/zstd)
3. Profile and optimize hotspots

**Estimated effort**: 1-2 days
**Complexity**: Low
**Gains**: 
- 5-10x storage reduction (3.5 MB → 0.4-0.7 MB for our example)
- 10-20x I/O speed improvement

### Phase 2: 100K → 1M documents
**Goal**: Enterprise-scale monorepos

1. Integrate HNSW indexing (hnsw_rs)
2. Add optional product quantization
3. Benchmark and tune parameters

**Estimated effort**: 1-2 weeks
**Complexity**: Medium
**Gains**: 50-100x search speed

### Phase 3: 1M+ documents
**Goal**: Multi-repository organizations

1. Integrate Qdrant or similar
2. Implement sharding strategy
3. Add monitoring and metrics

**Estimated effort**: 2-4 weeks
**Complexity**: High
**Gains**: Handle billions of documents

## Configuration Tuning

### For Different Use Cases

#### Small Projects (<10K documents, <6 MB code)
```rust
// Current implementation is optimal
RagConfig {
    chunk_size: 512,
    chunk_overlap: 50,
    // JSON is fine, human-readable for debugging
}
```

#### Medium Projects (10K-100K documents, 6-60 MB code)
```rust
RagConfig {
    chunk_size: 1024,  // Larger chunks = fewer documents
    chunk_overlap: 100,
    storage_format: StorageFormat::Binary, // Faster I/O
    compression: Some(Compression::Zstd), // Save disk space
}
```

#### Large Projects (100K-1M documents, 60-600 MB code)
```rust
RagConfig {
    chunk_size: 1024,
    chunk_overlap: 50,  // Less overlap = fewer documents
    index_type: IndexType::HNSW,
    storage_format: StorageFormat::Binary,
    compression: Some(Compression::Zstd),
}
```

#### Enterprise Scale (1M+ documents, 600+ MB code)
```rust
RagConfig {
    chunk_size: 2048,  // Larger chunks
    chunk_overlap: 0,   // No overlap
    index_type: IndexType::External(VectorDbConfig {
        url: "http://qdrant:6334",
        collection: "nucleus_rag",
    }),
}
```

## Benchmarking Guide

### Measure Current Performance

```rust
use std::time::Instant;

// 1. Indexing speed
let start = Instant::now();
manager.index_directory("./large_project").await?;
println!("Indexed in {:?}", start.elapsed());

// 2. Search speed
let start = Instant::now();
let results = manager.retrieve_context("query").await?;
println!("Search took {:?}", start.elapsed());

// 3. Memory usage
println!("Document count: {}", manager.count());
println!("Memory estimate: {} MB", manager.count() * 5 / 1024);

// 4. Disk usage
let metadata = fs::metadata("./data/vectordb/vector_store.json")?;
println!("Disk usage: {} MB", metadata.len() / 1024 / 1024);
```

### Performance Targets

| Operation | Target | Current (210 docs) | Projected (100K docs) |
|-----------|--------|--------------------|-----------------------|
| Indexing | >100 files/min | ✅ ~500 files/min | ✅ ~500 files/min |
| Search | <100ms | ✅ <1ms | ⚠️ ~500ms |
| Load | <1s | ✅ <50ms | ⚠️ 10-20s |
| Save | <1s | ✅ <100ms | ⚠️ 20-30s |

## Summary

### Current State
- ✅ **Optimal for**: 1K-50K documents (0.6-30 MB source code)
- ✅ **Works for**: Up to 100K documents (60 MB source code)
- ⚠️ **Struggles with**: 500K+ documents (300+ MB source code)
- ❌ **Not suitable for**: 1M+ documents (600+ MB source code)

### Real Example (This Project)
- **133 KB** source code → **210 documents** → **3.5 MB** vector DB (JSON)
- Search: <1ms (instant)
- Load: <50ms
- Plenty of room to grow!

### Growth Path
1. **Binary storage** → 10x improvement in I/O (3.5 MB → 0.5 MB)
2. **HNSW indexing** → 100x improvement in search
3. **External DB** → Infinite scalability

### Next Steps
For most users, the current implementation is sufficient. Consider optimizations when:
- Search takes >500ms consistently
- Load/save takes >5s
- You have >50,000 documents (~30 MB source code)
- You plan to index entire large organizations

The modular design makes it easy to swap storage backends and index types as your needs grow.
