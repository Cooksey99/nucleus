//! In-memory vector storage and search.
//!
//! This module provides a simple but effective vector database implementation
//! using in-memory storage and cosine similarity for search.

use super::persistence::{load_from_disk, save_to_disk, PersistenceError};
use super::types::{Document, SearchResult};
use std::path::PathBuf;
use std::sync::{Arc, RwLock};

/// An in-memory vector store for document embeddings.
///
/// The vector store maintains a collection of documents with their embeddings
/// and provides similarity-based search using cosine similarity. Documents are
/// stored in memory and protected by a RwLock for thread-safe concurrent access.
///
/// # Characteristics
///
/// - **Simple**: No external dependencies or setup required
/// - **Fast**: In-memory storage with O(n) search (linear scan)
/// - **Thread-safe**: Uses `Arc<RwLock>` for safe concurrent access
/// - **Ephemeral**: Data is lost when the process ends
///
/// # When to Use
///
/// This implementation is suitable for:
/// - Small to medium datasets (< 10,000 documents)
/// - Prototyping and development
/// - Applications where persistence isn't required
///
/// For larger datasets or persistent storage, consider:
/// - Qdrant, Milvus, or Weaviate for production workloads
/// - Pinecone or similar cloud services
///
#[derive(Clone)]
pub struct VectorStore {
    documents: Arc<RwLock<Vec<Document>>>,
    persistence_path: Option<PathBuf>,
}

impl VectorStore {
    pub fn new() -> Self {
        Self {
            documents: Arc::new(RwLock::new(Vec::new())),
            persistence_path: None,
        }
    }
    
    /// Creates a new vector store with persistent storage.
    ///
    /// # Arguments
    ///
    /// * `storage_path` - Directory where the vector database will be stored.
    ///   The file will be named "vector_store.json" within this directory.
    ///
    pub fn with_persistence(storage_path: impl Into<PathBuf>) -> Self {
        let mut path = storage_path.into();
        path.push("vector_store.json");
        
        Self {
            documents: Arc::new(RwLock::new(Vec::new())),
            persistence_path: Some(path),
        }
    }
    
    /// Loads documents from disk if persistence is enabled.
    ///
    /// This should be called after creating the store to restore previously
    /// indexed documents.
    ///
    /// # Errors
    ///
    /// Returns an error if loading fails. If no persistence path is set,
    /// this is a no-op.
    ///
    pub async fn load(&self) -> Result<usize, PersistenceError> {
        if let Some(path) = &self.persistence_path {
            let docs = load_from_disk(path).await?;
            let count = docs.len();
            
            let mut store = self.documents.write().unwrap();
            *store = docs;
            
            Ok(count)
        } else {
            Ok(0)
        }
    }
    
    /// Saves documents to disk if persistence is enabled.
    ///
    /// # Errors
    ///
    /// Returns an error if saving fails. If no persistence path is set,
    /// this is a no-op.
    ///
    pub async fn save(&self) -> Result<(), PersistenceError> {
        if let Some(path) = &self.persistence_path {
            // Clone documents to avoid holding the lock across await
            let docs = self.documents.read().unwrap().clone();
            save_to_disk(&docs, path).await?;
        }
        Ok(())
    }
    
    /// Adds a document to the store.
    ///
    /// Note: No deduplication is performed, so adding the same document multiple
    /// times will create duplicates.
    pub fn add(&self, document: Document) {
        let mut docs = self.documents.write().unwrap();
        docs.push(document);
    }
    
    /// Searches for the most similar documents using cosine similarity.
    ///
    /// Performs a linear scan over all documents, computing cosine similarity
    /// between the query embedding and each document's embedding. Results are
    /// sorted by similarity score in descending order (best matches first).
    ///
    /// # Arguments
    ///
    /// * `query_embedding` - The embedding vector to search for
    /// * `top_k` - Maximum number of results to return
    ///
    /// # Returns
    ///
    /// A vector of search results, sorted by descending similarity score.
    /// May contain fewer than `top_k` results if the store has fewer documents.
    ///
    /// # Performance
    ///
    /// Time complexity: O(n * d) where n is the number of documents and d is
    /// the embedding dimension. Space complexity: O(n) for storing results.
    ///
    pub fn search(&self, query_embedding: &[f32], top_k: usize) -> Vec<SearchResult> {
        let docs = self.documents.read().unwrap();
        
        let mut results: Vec<SearchResult> = docs
            .iter()
            .map(|doc| {
                let score = cosine_similarity(query_embedding, &doc.embedding);
                SearchResult {
                    document: doc.clone(),
                    score,
                }
            })
            .collect();
        
        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        
        results.into_iter().take(top_k).collect()
    }
    
    pub fn count(&self) -> usize {
        self.documents.read().unwrap().len()
    }
    
    pub fn clear(&self) {
        self.documents.write().unwrap().clear();
    }
}

/// Computes cosine similarity between two vectors.
///
/// Returns values from -1.0 (opposite) to 1.0 (identical), with 0.0 indicating
/// orthogonal vectors. Returns 0.0 for mismatched lengths or zero magnitude.
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return 0.0;
    }
    
    let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let magnitude_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let magnitude_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    
    if magnitude_a == 0.0 || magnitude_b == 0.0 {
        return 0.0;
    }
    
    dot_product / (magnitude_a * magnitude_b)
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert_eq!(cosine_similarity(&a, &b), 1.0);
        
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        assert_eq!(cosine_similarity(&a, &b), 0.0);
    }
    
    #[test]
    fn test_vector_store() {
        let store = VectorStore::new();
        
        let doc = Document::new("1", "test", vec![1.0, 0.0, 0.0]);
        store.add(doc);
        
        assert_eq!(store.count(), 1);
        
        let results = store.search(&[1.0, 0.0, 0.0], 5);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].score, 1.0);
    }
}
