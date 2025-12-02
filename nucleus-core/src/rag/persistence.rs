//! Persistence layer for the vector store.
//!
//! This module provides functionality to save and load vector store data
//! to/from disk, enabling persistent storage of indexed documents and embeddings.

use super::types::Document;
use serde::{Deserialize, Serialize};
use std::path::Path;
use thiserror::Error;
use tokio::fs;

#[derive(Debug, Error)]
pub enum PersistenceError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
}

pub type Result<T> = std::result::Result<T, PersistenceError>;

/// Serializable representation of the vector store.
#[derive(Debug, Serialize, Deserialize)]
pub struct VectorStoreSnapshot {
    pub documents: Vec<Document>,
    pub version: u32,
}

impl VectorStoreSnapshot {
    pub fn new(documents: Vec<Document>) -> Self {
        Self {
            documents,
            version: 1,
        }
    }
}

/// Saves documents to a file on disk.
///
/// Documents are serialized to JSON format for human-readability and
/// ease of inspection. For large datasets, consider using a binary format.
///
/// # Arguments
///
/// * `documents` - The documents to save
/// * `path` - The file path where data should be saved
///
/// # Errors
///
/// Returns an error if:
/// - The parent directory doesn't exist and can't be created
/// - File writing fails
/// - Serialization fails
///
pub async fn save_to_disk(documents: &[Document], path: impl AsRef<Path>) -> Result<()> {
    let path = path.as_ref();
    
    // Create parent directory if it doesn't exist
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).await?;
    }
    
    let snapshot = VectorStoreSnapshot::new(documents.to_vec());
    let json = serde_json::to_string_pretty(&snapshot)?;
    
    fs::write(path, json).await?;
    
    Ok(())
}

/// Loads documents from a file on disk.
///
/// # Arguments
///
/// * `path` - The file path to load from
///
/// # Returns
///
/// A vector of documents, or an empty vector if the file doesn't exist.
///
/// # Errors
///
/// Returns an error if:
/// - The file exists but can't be read
/// - Deserialization fails
///
pub async fn load_from_disk(path: impl AsRef<Path>) -> Result<Vec<Document>> {
    let path = path.as_ref();
    
    if !path.exists() {
        return Ok(Vec::new());
    }
    
    let contents = fs::read_to_string(path).await?;
    let snapshot: VectorStoreSnapshot = serde_json::from_str(&contents)?;
    
    Ok(snapshot.documents)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rag::Document;
    use tempfile::tempdir;
    
    #[tokio::test]
    async fn test_save_and_load() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test_store.json");
        
        let doc = Document::new("test_1", "test content", vec![1.0, 2.0, 3.0])
            .with_metadata("source", "test");
        
        let docs = vec![doc.clone()];
        
        // Save
        save_to_disk(&docs, &path).await.unwrap();
        assert!(path.exists());
        
        // Load
        let loaded = load_from_disk(&path).await.unwrap();
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0].id, doc.id);
        assert_eq!(loaded[0].content, doc.content);
        assert_eq!(loaded[0].embedding, doc.embedding);
    }
    
    #[tokio::test]
    async fn test_load_nonexistent() {
        let docs = load_from_disk("nonexistent.json").await.unwrap();
        assert!(docs.is_empty());
    }
}
