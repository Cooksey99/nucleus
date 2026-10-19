//! Embedding-model metadata.

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

/// An embedding model: a Hugging Face repo or a local path, plus its vector size.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingModel {
    /// Stable identifier for the model.
    pub id: String,
    /// Human-readable model name.
    pub name: String,
    /// Local path to the model, when it is stored on disk.
    pub path: Option<PathBuf>,
    /// Hugging Face repository, when the model is downloaded from the Hub.
    pub hf_repo: Option<String>,
    /// Maximum context length in tokens.
    pub context_length: usize,
    /// Size of the embedding vectors this model produces.
    pub embedding_dim: usize,
    /// Short description of the model.
    pub description: String,
}

impl Default for EmbeddingModel {
    fn default() -> Self {
        Self {
            id: "qwen3-embedding-0.6b".to_string(),
            name: "Qwen3 Embedding 0.6B".to_string(),
            path: None,
            hf_repo: Some("Qwen/Qwen3-Embedding-0.6B".to_string()),
            context_length: 32768,
            embedding_dim: 1024,
            description: "Multilingual text embedding model with MRL support".to_string(),
        }
    }
}
