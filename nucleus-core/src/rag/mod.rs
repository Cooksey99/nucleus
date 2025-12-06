//! Retrieval Augmented Generation (RAG) system.
//!
//! This module implements a complete RAG pipeline for enhancing LLM responses
//! with relevant context from a knowledge base.
//!
//! # Overview
//!
//! RAG (Retrieval Augmented Generation) is a technique that combines:
//! 1. **Retrieval**: Finding relevant documents from a knowledge base
//! 2. **Augmentation**: Adding those documents as context to the LLM prompt
//! 3. **Generation**: The LLM generates a response using the added context
//!
//! This results in more accurate, up-to-date responses grounded in your data.
//!
//! # Architecture
//!
//! The RAG system consists of four main components:
//!
//! - [`Manager`]: Orchestrates the entire RAG pipeline
//! - [`embedder`]: Converts text to vector embeddings via Ollama
//! - [`store`]: In-memory vector database with similarity search
//! - [`indexer`]: File collection and text chunking utilities
//!
//!
//! # How It Works
//!
//! 1. **Indexing Phase**:
//!    - Documents are split into chunks (default: 512 bytes with 50 byte overlap)
//!    - Each chunk is converted to a vector embedding
//!    - Embeddings are stored in the vector database
//!
//! 2. **Retrieval Phase**:
//!    - User query is converted to a vector embedding  
//!    - Vector database finds the top-k most similar document chunks
//!    - Similar chunks are returned as context
//!
//! 3. **Generation Phase** (handled by chat manager):
//!    - Context is added to the LLM prompt
//!    - LLM generates response using the context

mod embedder;
mod indexer;
mod lancedb_store;
mod qdrant_store;
mod store;
mod types;
pub mod utils;

#[allow(unused)]
pub use types::{Document, SearchResult};

use crate::config::{Config, IndexerConfig};
use crate::provider::Provider;
use embedder::Embedder;
use indexer::{chunk_text, collect_files};
use store::{create_vector_store, VectorStore};
use std::path::Path;
use std::sync::Arc;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum RagError {
    #[error("Embedder error: {0}")]
    Embedder(#[from] embedder::EmbedderError),
    
    #[error("Indexer error: {0}")]
    Indexer(#[from] indexer::IndexerError),
    
    #[error("Failed to retrieve context: {0}")]
    Retrieval(String),
}

pub type Result<T> = std::result::Result<T, RagError>;

/// The main RAG manager orchestrating all components.
///
/// The manager ties together the embedder, vector store, and indexer to provide
/// a high-level API for RAG operations. It handles the complete pipeline from
/// document ingestion to context retrieval.
///
/// # Thread Safety
///
/// The manager is `Clone` and can be safely shared across threads. The underlying
/// vector store uses `Arc<RwLock>` for thread-safe access.
///
/// # Configuration
///
/// The manager uses configuration from [`Config`]:
/// - `rag.embedding_model`: Model for generating embeddings
/// - `rag.chunk_size`: Size of text chunks in bytes
/// - `rag.chunk_overlap`: Overlap between chunks in bytes
/// - `storage.top_k`: Number of results to return from searches
#[derive(Clone)]
pub struct Rag {
    embedder: Embedder,
    store: Arc<dyn VectorStore>,
    chunk_size: usize,
    chunk_overlap: usize,
    indexer_config: IndexerConfig,
}

impl Rag {
    /// Creates a new RAG manager with vector database.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use nucleus_core::{Config, rag::Rag, provider::OllamaProvider};
    /// # use std::sync::Arc;
    /// # async fn example() {
    /// let config = Config::default();
    /// let provider = Arc::new(OllamaProvider::new(&config.llm.base_url));
    /// let manager = Rag::new(&config, provider).await.unwrap();
    /// # }
    /// ```
    pub async fn new(config: &Config, provider: Arc<dyn Provider>) -> Result<Self> {
        let embedder = Embedder::new(provider, &config.rag.embedding_model);
        let store = create_vector_store(
            config.storage.clone(),
            768,
        ).await.map_err(|e| RagError::Retrieval(e.to_string()))?;
        
        Ok(Self {
            embedder,
            store,
            chunk_size: config.rag.chunk_size,
            chunk_overlap: config.rag.chunk_overlap,
            indexer_config: config.rag.indexer.clone(),
        })
    }
    /// Adds a single piece of text to the knowledge base.
    ///
    /// The text is embedded and stored as a single document. For large texts,
    /// consider using [`index_directory`](Self::index_directory) which automatically
    /// chunks content.
    ///
    /// # Arguments
    ///
    /// * `content` - The text to add to the knowledge base
    /// * `source` - Identifier for the source (e.g., "user_input", "api", "manual")
    ///
    /// # Errors
    ///
    /// Returns an error if embedding generation fails.
    ///
    pub async fn add_knowledge(&self, content: &str, source: &str) -> Result<()> {
        let embedding = self.embedder.embed(content).await?;
        
        let count = self.store.count().await.unwrap_or(0);
        let id = format!("{}_{}", source, count);
        let document = Document::new(id, content, embedding)
            .with_metadata("source", source);
        
        self.store.add(document).await.map_err(|e| RagError::Retrieval(e.to_string()))?;
        Ok(())
    }
    
    /// Helper to process a batch of chunks and their metadata.
    async fn process_batch(
        &self,
        chunk_batch: &mut Vec<String>,
        chunk_metadata: &mut Vec<(String, String, String, usize)>,
    ) -> Result<()> {
        use tracing::info;
        
        info!("Processing batch of {} chunks", chunk_batch.len());
        let chunk_refs: Vec<&str> = chunk_batch.iter().map(|s| s.as_str()).collect();
        
        info!("Calling embed_batch for {} texts", chunk_refs.len());
        let embeddings = self.embedder.embed_batch(&chunk_refs).await?;
        info!("Received {} embeddings", embeddings.len());
        
        for (embedding, (id, content, source, chunk_idx)) in embeddings.into_iter().zip(chunk_metadata.drain(..)) {
            let document = Document::new(id, content, embedding)
                .with_metadata("source", source)
                .with_metadata("chunk", chunk_idx.to_string());
            
            self.store.add(document).await.map_err(|e| RagError::Retrieval(e.to_string()))?;
        }
        
        info!("Batch processed successfully");
        chunk_batch.clear();
        Ok(())
    }
    
    /// Recursively indexes all code files in a directory.
    ///
    /// Walks the directory tree, collecting indexable files (see [`indexer`] for
    /// supported extensions). Each file is:
    /// 1. Read and split into chunks
    /// 2. Each chunk is embedded
    /// 3. Chunks are stored with file path and chunk index metadata
    ///
    /// Progress is printed to stdout as files are indexed.
    ///
    /// # Arguments
    ///
    /// * `dir_path` - Root directory to index
    ///
    /// # Returns
    ///
    /// The number of files successfully indexed.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The directory doesn't exist or isn't accessible
    /// - Embedding generation fails for any chunk
    ///
    pub async fn index_directory(&self, dir_path: &Path) -> Result<usize> {
        let files = collect_files(dir_path, &self.indexer_config).await?;
        
        use tracing::{info, debug};
        info!("Found {} files to index", files.len());
        for file in &files {
            debug!(target: "nucleus_core::rag", file = %file.path.display(), "File queued for indexing");
        }
        info!("Starting indexing...");
        
        let mut indexed_count = 0;
        
        const BATCH_SIZE: usize = 32;
        let mut chunk_batch = Vec::new();
        let mut chunk_metadata = Vec::new();
        
        for file in files {
            if file.content.is_empty() {
                eprintln!("WARNING: File has empty content: {}", file.path.display());
                continue;
            }
            
            let chunks = chunk_text(&file.content, self.chunk_size, self.chunk_overlap);
            
            if chunks.is_empty() {
                eprintln!("WARNING: No chunks created for file: {}", file.path.display());
                continue;
            }
            
            for (i, chunk) in chunks.into_iter().enumerate() {
                chunk_batch.push(chunk.clone());
                chunk_metadata.push((
                    format!("{}_chunk_{}", file.path.display(), i),
                    chunk,
                    file.path.to_string_lossy().to_string(),
                    i,
                ));
                
                // Process batch when it reaches BATCH_SIZE
                if chunk_batch.len() >= BATCH_SIZE {
                    self.process_batch(&mut chunk_batch, &mut chunk_metadata).await?;
                }
            }
            
            indexed_count += 1;
            println!("✓ Indexed: {}", file.path.display());
        }
        
        // Process remaining chunks
        if !chunk_batch.is_empty() {
            self.process_batch(&mut chunk_batch, &mut chunk_metadata).await?;
        }
        
        Ok(indexed_count)
    }
    
    /// Indexes multiple directories in batch.
    ///
    /// This is a convenience method for indexing multiple directories at once.
    /// Each directory is processed sequentially with the same indexing configuration.
    ///
    /// # Arguments
    ///
    /// * `dir_paths` - A slice of directory paths to index
    ///
    /// # Returns
    ///
    /// The total number of files successfully indexed across all directories.
    ///
    /// # Errors
    ///
    /// Returns an error if any directory fails to index. Previously indexed
    /// directories are retained in the knowledge base.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use nucleus_core::{Config, rag::Rag, ollama::Client};
    /// # async fn example(manager: Rag) {
    /// let dirs = vec!["./src", "./docs", "./examples"];
    /// let count = manager.index_directories(&dirs).await.unwrap();
    /// println!("Indexed {} files", count);
    /// # }
    /// ```
    pub async fn index_directories(&self, dir_paths: &[&str]) -> Result<usize> {
        let mut total_count = 0;
        
        for dir_path in dir_paths {
            println!("\nIndexing directory: {}", dir_path);
            let dir_path = Path::new(dir_path);
            let count = self.index_directory(dir_path).await?;
            total_count += count;
        }
        
        println!("\nTotal files indexed: {}", total_count);
        Ok(total_count)
    }
    
    /// Indexes a single file directly.
    ///
    /// This is useful for indexing individual files outside of directory traversal.
    ///
    /// # Arguments
    ///
    /// * `file_path` - Path to the file to index
    ///
    /// # Returns
    ///
    /// The number of chunks created from the file.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The file cannot be read
    /// - Embedding generation fails
    ///
    pub async fn index_file(&self, file_path: &str) -> Result<usize> {
        use tokio::fs;
        
        let content = fs::read_to_string(file_path).await
            .map_err(|e| RagError::Indexer(indexer::IndexerError::Io(e)))?;
        
        let chunks = chunk_text(&content, self.chunk_size, self.chunk_overlap);
        let chunk_count = chunks.len();
        
        for (i, chunk) in chunks.into_iter().enumerate() {
            let embedding = self.embedder.embed(&chunk).await?;
            
            let id = format!("{}_chunk_{}", file_path, i);
            let document = Document::new(id, chunk, embedding)
                .with_metadata("source", file_path)
                .with_metadata("chunk", i.to_string());
            
            self.store.add(document).await.map_err(|e| RagError::Retrieval(e.to_string()))?;
        }
        
        println!("✓ Indexed: {} ({} chunks)", file_path, chunk_count);
        Ok(chunk_count)
    }
    
    /// Retrieves relevant context from the knowledge base for a query.
    ///
    /// Converts the query to an embedding, searches for the top-k most similar
    /// documents, and formats them as context that can be added to an LLM prompt.
    ///
    /// # Arguments
    ///
    /// * `query` - The question or text to find relevant context for
    ///
    /// # Returns
    ///
    /// A formatted string containing the most relevant document chunks, or an
    /// empty string if the knowledge base is empty or no relevant documents exist.
    ///
    /// The format is:
    /// ```text
    /// 
    /// Relevant context from your knowledge base:
    ///
    /// [1] <first most relevant chunk>
    /// [2] <second most relevant chunk>
    /// ...
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if embedding generation fails.
    ///
    pub async fn retrieve_context(&self, query: &str) -> Result<String> {
        use tracing::{debug, info};
        
        let count = self.store.count().await.unwrap_or(0);
        debug!("Knowledge base count: {}", count);
        if count == 0 {
            debug!("Knowledge base is empty, returning empty context");
            return Ok(String::new());
        }
        
        debug!("Generating query embedding for: {}", query);
        let query_embedding = self.embedder.embed(query).await?;
        debug!("Query embedding generated, dimension: {}", query_embedding.len());
        
        debug!("Searching vector store...");
        let results = self.store.search(&query_embedding)
            .await
            .map_err(|e| RagError::Retrieval(e.to_string()))?;
        
        info!("Found {} results from RAG search", results.len());
        
        if results.is_empty() {
            debug!("No results found, returning empty context");
            return Ok(String::new());
        }
        
        let mut context = String::from("\n\nRelevant context from your knowledge base:\n");
        
        for (i, result) in results.iter().enumerate() {
            debug!("Result {}: score={}, source={:?}", 
                i + 1, 
                result.score, 
                result.document.metadata.get("source"));
            context.push_str(&format!("\n[{}] {}\n", i + 1, result.document.content));
        }
        
        info!("Generated context with {} results", results.len());
        Ok(context)
    }
    
    /// Returns the total number of documents (chunks) in the knowledge base.
    ///
    /// Note: each indexed file is split into multiple chunks, so this represents
    /// chunk count, not file count.
    pub async fn count(&self) -> usize {
        self.store.count().await.unwrap_or(0)
    }
    
    /// Removes all documents from the knowledge base.
    pub async fn clear(&self) -> Result<()> {
        self.store.clear().await.map_err(|e| RagError::Retrieval(e.to_string()))?;
        Ok(())
    }
    
    /// Returns all unique file paths that have been indexed in the knowledge base.
    ///
    /// This method queries Qdrant to retrieve all unique source file paths
    /// from indexed documents. Useful for displaying indexing status in UIs.
    pub async fn get_indexed_paths(&self) -> Result<Vec<String>> {
        self.store.get_indexed_paths().await
            .map_err(|e| RagError::Retrieval(e.to_string()))
    }

    /// Removes documents from the knowledge base by source path.
    ///
    /// This method removes all documents that match the given source path.
    /// If the path is a directory, all files within that directory are removed.
    /// If the path is a file, only that specific file is removed.
    ///
    /// # Arguments
    ///
    /// * `source_path` - The file or directory path to remove from the knowledge base
    ///
    /// # Returns
    ///
    /// The number of document chunks removed.
    ///
    /// # Errors
    ///
    /// Returns an error if the removal operation fails.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use nucleus_core::{Config, rag::Manager, ollama::Client};
    /// # async fn example(manager: Manager) {
    /// // Remove a specific file
    /// let removed = manager.remove_from_knowledge_base("./src/main.rs").await.unwrap();
    /// println!("Removed {} chunks", removed);
    ///
    /// // Remove an entire directory
    /// let removed = manager.remove_from_knowledge_base("./docs").await.unwrap();
    /// println!("Removed {} chunks", removed);
    /// # }
    /// ```
    pub async fn remove_from_knowledge_base(&self, source_path: &str) -> Result<usize> {
        let removed = self.store.remove_by_source(source_path).await
            .map_err(|e| RagError::Retrieval(e.to_string()))?;
        
        if removed > 0 {
            println!("Removed {} document chunks from: {}", removed, source_path);
        } else {
            println!("No documents found for: {}", source_path);
        }
        
        Ok(removed)
    }
}
