//! Qdrant vector database storage implementation.
//!
//! This module provides integration with Qdrant, a high-performance vector database
//! that offers automatic deduplication, persistence, and scalability.

use super::store::VectorStore;
use super::types::{Document, SearchResult};
use crate::config::{StorageConfig, StorageMode};
use anyhow::{Context, Result};
use async_trait::async_trait;
use qdrant_client::{
    qdrant::{
        vectors_config::Config, CreateCollectionBuilder, DeletePointsBuilder, Distance,
        PointStruct, ScrollPointsBuilder, SearchPointsBuilder, UpsertPointsBuilder,
        VectorParamsBuilder, VectorsConfig,
    },
    Qdrant,
};
use serde_json::json;
use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

/// Qdrant-based vector store for document embeddings.
///
/// Provides persistent, scalable vector storage with automatic deduplication
/// through Qdrant's upsert mechanism. When a document with an existing ID is
/// added, it automatically replaces the old version.
///
/// # Features
///
/// - **Automatic deduplication**: Re-indexing replaces old documents
/// - **Persistent storage**: Data survives restarts
/// - **Scalable**: Handles millions of documents efficiently
/// - **Fast search**: Optimized vector similarity with HNSW indexing
///
#[derive(Clone)]
pub struct QdrantStore {
    storage_config: StorageConfig,
    client: Arc<Qdrant>,
    collection_name: String,
    vector_size: u64,
}

#[async_trait]
impl VectorStore for QdrantStore {
    async fn add(&self, documents: Vec<Document>) -> Result<()> {
        if documents.is_empty() {
            return Ok(());
        }

        let points: Vec<PointStruct> = documents
            .into_iter()
            .map(|document| {
                let mut hasher = DefaultHasher::new();
                document.id.hash(&mut hasher);
                let numeric_id = hasher.finish();

                let payload: HashMap<String, serde_json::Value> = document
                    .metadata
                    .iter()
                    .map(|(k, v)| (k.clone(), json!(v)))
                    .chain(vec![
                        ("content".to_string(), json!(document.content)),
                        ("id".to_string(), json!(document.id)),
                    ])
                    .collect();

                PointStruct::new(numeric_id, document.embedding, payload)
            })
            .collect();

        self.client
            .upsert_points(UpsertPointsBuilder::new(&self.collection_name, points))
            .await
            .context("Failed to upsert points")?;

        Ok(())
    }

    /// Searches for the most similar documents using cosine similarity.
    ///
    /// # Arguments
    ///
    /// * `query_embedding` - The embedding vector to search for
    /// * `top_k` - Maximum number of results to return
    ///
    /// # Returns
    ///
    /// A vector of search results, sorted by descending similarity score.
    async fn search(&self, query_embedding: &[f32]) -> Result<Vec<SearchResult>> {
        let search_result = self
            .client
            .search_points(
                SearchPointsBuilder::new(
                    &self.collection_name,
                    query_embedding.to_vec(),
                    self.storage_config.top_k as u64,
                )
                .with_payload(true),
            )
            .await
            .context("Failed to search points")?;

        let results = search_result
            .result
            .into_iter()
            .map(|point| {
                let payload = point.payload;
                let content = payload
                    .get("content")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string())
                    .unwrap_or_default();

                // Get the original ID from metadata
                let id = payload
                    .get("id")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string())
                    .unwrap_or_default();

                let metadata: HashMap<String, String> = payload
                    .iter()
                    .filter(|(k, _)| k.as_str() != "content" && k.as_str() != "id")
                    .filter_map(|(k, v)| v.as_str().map(|s| (k.clone(), s.to_string())))
                    .collect();

                let document = Document {
                    id,
                    content,
                    embedding: vec![], // Don't return embeddings in search results
                    metadata,
                };

                SearchResult {
                    document,
                    score: point.score,
                }
            })
            .collect();

        Ok(results)
    }

    /// Returns the total number of documents in the collection.
    async fn count(&self) -> Result<usize> {
        let info = self
            .client
            .collection_info(&self.collection_name)
            .await
            .context("Failed to get collection info")?;

        Ok(info
            .result
            .map(|r| r.points_count.unwrap_or(0) as usize)
            .unwrap_or(0))
    }

    /// Removes all documents from the collection.
    async fn clear(&self) -> Result<()> {
        // Delete the collection
        self.client
            .delete_collection(&self.collection_name)
            .await
            .context("Failed to delete collection")?;

        // Recreate it
        self.ensure_collection().await?;

        Ok(())
    }

    /// Returns all unique source file paths that have been indexed.
    ///
    /// Scrolls through all documents in the collection and extracts unique
    /// source paths from the metadata.
    async fn get_indexed_paths(&self) -> Result<Vec<String>> {
        use std::collections::HashSet;

        let mut unique_paths = HashSet::new();
        let mut offset: Option<qdrant_client::qdrant::PointId> = None;

        // Scroll through all points in batches
        loop {
            let mut builder = ScrollPointsBuilder::new(&self.collection_name)
                .limit(100)
                .with_payload(true);

            if let Some(off) = offset {
                builder = builder.offset(off);
            }

            let scroll_result = self
                .client
                .scroll(builder)
                .await
                .context("Failed to scroll points")?;

            // Extract source paths from this batch
            for point in &scroll_result.result {
                let payload = &point.payload;
                if let Some(source_value) = payload.get("source") {
                    if let Some(source_str) = source_value.as_str() {
                        unique_paths.insert(source_str.to_string());
                    }
                }
            }

            // Check if there are more results
            if let Some(next_offset) = scroll_result.next_page_offset {
                offset = Some(next_offset);
            } else {
                break;
            }
        }

        Ok(unique_paths.into_iter().collect())
    }

    /// Removes all documents with a matching source path.
    ///
    /// This method deletes all points where the "source" metadata field
    /// matches the provided path or starts with the provided path (for directory removal).
    ///
    /// # Arguments
    ///
    /// * `source_path` - The source path to remove (e.g., file path or directory path)
    ///
    /// # Returns
    ///
    /// The number of documents removed.
    async fn remove_by_source(&self, source_path: &str) -> Result<usize> {
        use std::path::Path;

        // Normalize the path for comparison
        let normalized_path = Path::new(source_path).to_string_lossy().replace("\\", "/");

        // First, count how many points will be deleted by scrolling
        let mut points_to_delete = Vec::new();
        let mut offset: Option<qdrant_client::qdrant::PointId> = None;

        loop {
            let mut builder = ScrollPointsBuilder::new(&self.collection_name)
                .limit(100)
                .with_payload(true);

            if let Some(off) = offset {
                builder = builder.offset(off);
            }

            let scroll_result = self
                .client
                .scroll(builder)
                .await
                .context("Failed to scroll points")?;

            // Check each point's source
            for point in &scroll_result.result {
                if let Some(point_id) = &point.id {
                    let payload = &point.payload;
                    if let Some(source_value) = payload.get("source") {
                        if let Some(source_str) = source_value.as_str() {
                            let point_source = source_str.replace("\\", "/");
                            // Match exact file or any file under directory
                            if point_source == normalized_path
                                || point_source.starts_with(&format!("{}/", normalized_path))
                            {
                                points_to_delete.push(point_id.clone());
                            }
                        }
                    }
                }
            }

            // Check if there are more results
            if let Some(next_offset) = scroll_result.next_page_offset {
                offset = Some(next_offset);
            } else {
                break;
            }
        }

        let count = points_to_delete.len();

        // Delete the points if any were found
        if !points_to_delete.is_empty() {
            self.client
                .delete_points(
                    DeletePointsBuilder::new(&self.collection_name).points(points_to_delete),
                )
                .await
                .context("Failed to delete points")?;
        }

        Ok(count)
    }
}

impl QdrantStore {
    /// Creates a new Qdrant store and ensures the collection exists.
    ///
    /// # Arguments
    ///
    /// * `storage_config` - Storage configuration including storage mode and collection name
    /// * `vector_size` - Dimension of the embedding vectors
    pub async fn new(storage_config: StorageConfig, vector_size: u64) -> Result<Self> {
        let client = match &storage_config.storage_mode {
            StorageMode::Grpc { url } => Arc::new(
                Qdrant::from_url(&url)
                    .build()
                    .context("Failed to connect to Qdrant server")?,
            ),
            _ => {
                anyhow::bail!("QdrantStore only supports Grpc mode")
            }
        };

        let collection_name = storage_config.vector_db.collection_name.clone();

        let store = Self {
            storage_config,
            client,
            collection_name,
            vector_size,
        };

        store.ensure_collection().await?;

        Ok(store)
    }

    async fn ensure_collection(&self) -> Result<()> {
        let collections = self
            .client
            .collection_exists(&self.collection_name)
            .await
            .context("Failed to check collection")?;

        if !collections {
            self.client
                .create_collection(
                    CreateCollectionBuilder::new(&self.collection_name).vectors_config(
                        VectorsConfig {
                            config: Some(Config::Params(
                                VectorParamsBuilder::new(self.vector_size, Distance::Cosine)
                                    .build(),
                            )),
                        },
                    ),
                )
                .await
                .context("Failed to create collection")?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // #[tokio::test]
    #[ignore] // Requires Qdrant server running
    async fn test_qdrant_store_grpc() {
        let mut storage_config = StorageConfig::default();
        storage_config.storage_mode = StorageMode::Grpc {
            url: "http://localhost:6334".to_string(),
        };
        storage_config.vector_db.collection_name = "test_collection_grpc".to_string();

        let store = QdrantStore::new(storage_config, 3).await.unwrap();

        let doc = Document::new("test_1", "Hello world", vec![1.0, 0.0, 0.0]);
        store.add(vec![doc]).await.unwrap();

        let count = store.count().await.unwrap();
        assert_eq!(count, 1);

        store.clear().await.unwrap();
    }
}
