//! In-memory vector storage and search.
//!
//! This module provides a simple but effective vector database implementation
//! using in-memory storage and cosine similarity for search.

use super::types::{Document, SearchResult};
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
/// # Example
///
/// ```no_run
/// # use core::rag::{store::VectorStore, types::Document};
/// let store = VectorStore::new();
///
/// // Add a document
/// let doc = Document::new("1", "Hello world", vec![0.1, 0.2, 0.3]);
/// store.add(doc);
///
/// // Search for similar documents
/// let query_embedding = vec![0.1, 0.2, 0.3];
/// let results = store.search(&query_embedding, 5);
/// ```
#[derive(Clone)]
pub struct VectorStore {
    /// Thread-safe storage for documents.
    ///
    /// Uses Arc for cheap cloning and RwLock for concurrent read/write access.
    documents: Arc<RwLock<Vec<Document>>>,
}

impl VectorStore {
    /// Creates a new empty vector store.
    ///
    /// # Example
    ///
    /// ```
    /// # use core::rag::store::VectorStore;
    /// let store = VectorStore::new();
    /// assert_eq!(store.count(), 0);
    /// ```
    pub fn new() -> Self {
        Self {
            documents: Arc::new(RwLock::new(Vec::new())),
        }
    }
    
    /// Adds a document to the store.
    ///
    /// Documents are appended to the internal vector. No deduplication is performed,
    /// so adding the same document multiple times will create duplicates.
    ///
    /// # Arguments
    ///
    /// * `document` - The document to add, including its embedding
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use core::rag::{store::VectorStore, types::Document};
    /// let store = VectorStore::new();
    /// let doc = Document::new("1", "content", vec![0.1, 0.2]);
    /// store.add(doc);
    /// assert_eq!(store.count(), 1);
    /// ```
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
    /// # Example
    ///
    /// ```no_run
    /// # use core::rag::{store::VectorStore, types::Document};
    /// let store = VectorStore::new();
    ///
    /// let doc1 = Document::new("1", "cats", vec![1.0, 0.0, 0.0]);
    /// let doc2 = Document::new("2", "dogs", vec![0.0, 1.0, 0.0]);
    /// store.add(doc1);
    /// store.add(doc2);
    ///
    /// // Search for documents similar to [1.0, 0.0, 0.0]
    /// let results = store.search(&[1.0, 0.0, 0.0], 5);
    /// assert_eq!(results[0].document.id, "1"); // "cats" is most similar
    /// assert_eq!(results[0].score, 1.0); // Perfect match
    /// ```
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
    
    /// Returns the total number of documents in the store.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use core::rag::{store::VectorStore, types::Document};
    /// let store = VectorStore::new();
    /// assert_eq!(store.count(), 0);
    ///
    /// store.add(Document::new("1", "content", vec![0.1]));
    /// assert_eq!(store.count(), 1);
    /// ```
    pub fn count(&self) -> usize {
        self.documents.read().unwrap().len()
    }
    
    /// Removes all documents from the store.
    ///
    /// After calling this method, the store will be empty (count returns 0).
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use core::rag::{store::VectorStore, types::Document};
    /// let store = VectorStore::new();
    /// store.add(Document::new("1", "content", vec![0.1]));
    /// assert_eq!(store.count(), 1);
    ///
    /// store.clear();
    /// assert_eq!(store.count(), 0);
    /// ```
    pub fn clear(&self) {
        self.documents.write().unwrap().clear();
    }
}

/// Computes cosine similarity between two vectors.
///
/// Cosine similarity measures the cosine of the angle between two vectors in
/// a multi-dimensional space. It is particularly useful for comparing document
/// embeddings as it is independent of vector magnitude.
///
/// # Formula
///
/// ```text
/// cosine_similarity(A, B) = (A · B) / (||A|| * ||B||)
/// ```
///
/// Where:
/// - `A · B` is the dot product of vectors A and B
/// - `||A||` is the magnitude (L2 norm) of vector A
/// - `||B||` is the magnitude (L2 norm) of vector B
///
/// # Return Value
///
/// - `1.0` - Vectors point in the same direction (identical)
/// - `0.0` - Vectors are orthogonal (unrelated)
/// - `-1.0` - Vectors point in opposite directions
/// - Returns `0.0` if vectors have different lengths or zero magnitude
///
/// # Arguments
///
/// * `a` - First vector
/// * `b` - Second vector
///
/// # Example
///
/// ```
/// # use core::rag::store::cosine_similarity;
/// let a = vec![1.0, 0.0, 0.0];
/// let b = vec![1.0, 0.0, 0.0];
/// assert_eq!(cosine_similarity(&a, &b), 1.0); // Identical
///
/// let a = vec![1.0, 0.0];
/// let b = vec![0.0, 1.0];
/// assert_eq!(cosine_similarity(&a, &b), 0.0); // Orthogonal
/// ```
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
