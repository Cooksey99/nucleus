//! Vector store abstraction and factory.
//!
//! This module provides a unified interface for different vector database implementations.

use super::lancedb_store::LanceDbStore;
use super::qdrant_store::QdrantStore;
use super::types::{Document, SearchResult};
use crate::config::{StorageConfig, StorageMode};
use anyhow::Result;
use async_trait::async_trait;
use std::sync::Arc;

/// Unified interface for vector database operations.
///
/// Implementations handle document storage, similarity search, and metadata queries
/// across different vector database backends (LanceDB for embedded, Qdrant for gRPC).
#[async_trait]
pub trait VectorStore: Send + Sync {
    /// Adds or updates multiple documents in the store.
    async fn add(&self, documents: Vec<Document>) -> Result<()>;
    /// Searches for the most similar documents using vector similarity.
    ///
    /// # Arguments
    ///
    /// * `query_embedding` - The embedding vector to search for
    /// * `top_k` - Maximum number of results to return
    ///
    /// # Returns
    ///
    /// A vector of search results, sorted by descending similarity score.
    async fn search(&self, query_embedding: &[f32]) -> Result<Vec<SearchResult>>;

    /// Returns the total number of documents in the store.
    async fn count(&self) -> Result<usize>;

    /// Removes all documents from the store.
    async fn clear(&self) -> Result<()>;

    /// Returns all unique source file paths that have been indexed.
    async fn get_indexed_paths(&self) -> Result<Vec<String>>;

    /// Removes all documents with a matching source path.
    ///
    /// # Arguments
    ///
    /// * `source_path` - The source path to remove (file or directory)
    ///
    /// # Returns
    ///
    /// The number of documents removed.
    async fn remove_by_source(&self, source_path: &str) -> Result<usize>;
}

/// Creates a vector store instance based on the storage mode.
///
/// - `Embedded` mode uses LanceDB for zero-setup, in-process storage
/// - `Grpc` mode uses Qdrant for remote server connectivity
///
/// # Arguments
///
/// * `storage_config` - Storage configuration including storage mode and top_k
/// * `vector_size` - Dimension of the embedding vectors
///
/// # Returns
///
/// A trait object that can be used for all vector store operations.
pub async fn create_vector_store(
    storage_config: StorageConfig,
    vector_size: u64,
) -> Result<Arc<dyn VectorStore>> {
    match storage_config.storage_mode.clone() {
        StorageMode::Embedded { path } => {
            let store = LanceDbStore::new(storage_config, &path, vector_size.into()).await?;
            Ok(Arc::new(store))
        }
        StorageMode::Grpc { .. } => {
            let store = QdrantStore::new(storage_config, vector_size).await?;
            Ok(Arc::new(store))
        }
    }
}
