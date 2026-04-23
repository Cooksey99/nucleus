//! Embedding generation using LLM providers.
//!
//! This module provides functionality to convert text into vector embeddings
//! using provider embedding models.

use crate::{
    models::EmbeddingModel,
    provider::{Provider, ProviderError},
};
use std::sync::Arc;
use thiserror::Error;

/// Errors that can occur during embedding generation.
#[derive(Debug, Error)]
pub enum EmbedderError {
    /// The provider API returned an error.
    #[error("Provider error: {0}")]
    Provider(#[from] ProviderError),

    /// The API response contained no embeddings.
    ///
    /// This typically indicates a problem with the model or request format.
    #[error("No embeddings returned")]
    NoEmbeddings,
}

/// Result type for embedding operations.
pub type Result<T> = std::result::Result<T, EmbedderError>;

/// Generates vector embeddings for text using LLM provider embedding models.
///
/// The embedder converts text into high-dimensional vectors that capture
/// semantic meaning. These vectors can then be compared using cosine
/// similarity to find semantically similar text.
///
/// # Supported Models
///
/// Common embedding models:
/// - `nomic-embed-text` - 768-dimensional embeddings, good general purpose
/// - `mxbai-embed-large` - 1024-dimensional embeddings, higher quality
///
#[derive(Clone)]
pub struct Embedder {
    provider: Arc<dyn Provider>,
    model: EmbeddingModel,
}

impl Embedder {
    pub fn new(provider: Arc<dyn Provider>, model: impl Into<EmbeddingModel>) -> Self {
        Self {
            provider,
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
        self.provider
            .embed(text, &self.model)
            .await
            .map_err(EmbedderError::Provider)
    }

    /// Generates embeddings for multiple texts in batch.
    ///
    /// This is more efficient than calling `embed()` repeatedly, as it can
    /// process multiple texts in a single request or pipeline them efficiently.
    ///
    /// # Arguments
    ///
    /// * `texts` - A slice of texts to embed
    ///
    /// # Returns
    ///
    /// A vector of embeddings, one for each input text.
    ///
    /// # Errors
    ///
    /// Returns an error if any embedding generation fails.
    pub async fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        use tracing::info;

        info!("Embedder::embed_batch called with {} texts", texts.len());
        let result = self
            .provider
            .embed_batch(texts, &self.model)
            .await
            .map_err(EmbedderError::Provider)?;
        info!(
            "Embedder::embed_batch completed, got {} embeddings",
            result.len()
        );

        Ok(result)
    }
}
