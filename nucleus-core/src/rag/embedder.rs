//! Embedding generation using Ollama.
//!
//! This module provides functionality to convert text into vector embeddings
//! using Ollama's embedding models.

use crate::ollama::{Client, EmbedRequest};
use thiserror::Error;

/// Errors that can occur during embedding generation.
#[derive(Debug, Error)]
pub enum EmbedderError {
    /// The Ollama API returned an error.
    #[error("Ollama error: {0}")]
    Ollama(#[from] crate::ollama::OllamaError),
    
    /// The API response contained no embeddings.
    ///
    /// This typically indicates a problem with the model or request format.
    #[error("No embeddings returned")]
    NoEmbeddings,
}

/// Result type for embedding operations.
pub type Result<T> = std::result::Result<T, EmbedderError>;

/// Generates vector embeddings for text using Ollama's embedding models.
///
/// The embedder converts text into high-dimensional vectors that capture
/// semantic meaning. These vectors can then be compared using cosine
/// similarity to find semantically similar text.
///
/// # Supported Models
///
/// Common embedding models available through Ollama:
/// - `nomic-embed-text` - 768-dimensional embeddings, good general purpose
/// - `mxbai-embed-large` - 1024-dimensional embeddings, higher quality
///
#[derive(Clone)]
pub struct Embedder {
    client: Client,
    model: String,
}

impl Embedder {
    pub fn new(client: Client, model: impl Into<String>) -> Self {
        Self {
            client,
            model: model.into(),
        }
    }
    
    /// Generates a vector embedding for the given text.
    ///
    /// The embedding is a high-dimensional vector (typically 768 or 1024 dimensions)
    /// that represents the semantic meaning of the text. Similar texts will have
    /// similar embeddings, as measured by cosine similarity.
    ///
    /// # Arguments
    ///
    /// * `text` - The text to embed. Can be a word, sentence, paragraph, or document.
    ///
    /// # Returns
    ///
    /// A vector of floating-point values representing the embedding. The length
    /// depends on the model used.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The Ollama API is unreachable
    /// - The model is not available
    /// - The API returns no embeddings
    ///
    pub async fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let request = EmbedRequest {
            model: self.model.clone(),
            input: text.to_string(),
        };
        
        let response = self.client.embed(request).await?;
        
        response.embeddings
            .into_iter()
            .next()
            .ok_or(EmbedderError::NoEmbeddings)
    }
}
