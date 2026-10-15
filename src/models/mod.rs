use std::path::PathBuf;

use serde::{Deserialize, Serialize};

/// An embedding model: a Hugging Face repo or a local path, plus its vector size.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingModel {
    pub id: String,
    pub name: String,
    pub path: Option<PathBuf>,
    pub hf_repo: Option<String>,
    pub context_length: usize,
    pub embedding_dim: usize,
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
