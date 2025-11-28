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
mod store;
mod types;

#[allow(unused)]
pub use types::{Document, SearchResult};

use crate::config::{Config, IndexerConfig};
use crate::ollama::Client;
use embedder::Embedder;
use indexer::{chunk_text, collect_files};
use store::VectorStore;
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
/// - `rag.top_k`: Number of results to return from searches
#[derive(Clone)]
pub struct Manager {
    embedder: Embedder,
    store: VectorStore,
    chunk_size: usize,
    chunk_overlap: usize,
    top_k: usize,
    indexer_config: IndexerConfig,
}

impl Manager {
    /// Creates a new RAG manager with the given configuration.
    pub fn new(config: &Config, ollama_client: Client) -> Self {
        let embedder = Embedder::new(ollama_client, &config.rag.embedding_model);
        let store = VectorStore::new();
        
        Self {
            embedder,
            store,
            chunk_size: config.rag.chunk_size,
            chunk_overlap: config.rag.chunk_overlap,
            top_k: config.rag.top_k,
            indexer_config: config.rag.indexer.clone(),
        }
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
        
        let id = format!("{}_{}", source, self.store.count());
        let document = Document::new(id, content, embedding)
            .with_metadata("source", source);
        
        self.store.add(document);
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
    pub async fn index_directory(&self, dir_path: &str) -> Result<usize> {
        let files = collect_files(dir_path, &self.indexer_config).await?;
        let mut indexed_count = 0;
        
        for file in files {
            let chunks = chunk_text(&file.content, self.chunk_size, self.chunk_overlap);
            
            for (i, chunk) in chunks.into_iter().enumerate() {
                let embedding = self.embedder.embed(&chunk).await?;
                
                let id = format!("{}_chunk_{}", file.path.display(), i);
                let document = Document::new(id, chunk, embedding)
                    .with_metadata("source", file.path.to_string_lossy())
                    .with_metadata("chunk", i.to_string());
                
                self.store.add(document);
            }
            
            indexed_count += 1;
            println!("✓ Indexed: {}", file.path.display());
        }
        
        Ok(indexed_count)
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
        if self.store.count() == 0 {
            return Ok(String::new());
        }
        
        let query_embedding = self.embedder.embed(query).await?;
        let results = self.store.search(&query_embedding, self.top_k);
        
        if results.is_empty() {
            return Ok(String::new());
        }
        
        let mut context = String::from("\n\nRelevant context from your knowledge base:\n");
        
        for (i, result) in results.iter().enumerate() {
            context.push_str(&format!("\n[{}] {}\n", i + 1, result.document.content));
        }
        
        Ok(context)
    }
    
    /// Returns the total number of documents (chunks) in the knowledge base.
    ///
    /// Note: each indexed file is split into multiple chunks, so this represents
    /// chunk count, not file count.
    pub fn count(&self) -> usize {
        self.store.count()
    }
    
    /// Removes all documents from the knowledge base.
    pub fn clear(&self) {
        self.store.clear();
    }
}
