use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;
use thiserror::Error;

use crate::models::EmbeddingModel;

#[derive(Debug, Error)]
pub enum ConfigError {
    #[error("Failed to read config file: {0}")]
    FileRead(#[from] std::io::Error),

    #[error("Failed to parse config: {0}")]
    Parse(#[from] serde_yaml::Error),
}

pub type Result<T> = std::result::Result<T, ConfigError>;

/// Configuration for the entire chat/agent
///
/// This includes the LLM model itself, as well as the features and customization you want it have
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub system_prompt: String,
    pub llm: LlmConfig,
    pub rag: RagConfig,
    pub storage: StorageConfig,
    pub personalization: PersonalizationConfig,

    #[serde(skip)]
    pub permission: Permission,
}

/// Permissions granted to the AI.
///
/// **Note**: A permission granted here does not mean it will automatically perform the actions.
/// However, if false, the functionality will not exist to begin with.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Permission {
    /// Read directories and files
    pub read: bool,
    /// Write to files
    pub write: bool,
    /// Run system commands
    pub command: bool,
}

impl Default for Permission {
    fn default() -> Self {
        Self {
            read: true,
            write: true,
            command: true,
        }
    }
}

/// Configuration for the AI model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmConfig {
    /// Provider type: "ollama", "mistralrs", or "coreml"
    #[serde(default = "default_provider")]
    pub provider: String,
    pub model: String,
    pub base_url: String,
    pub temperature: f64,
    pub context_length: usize,
    /// CoreML-specific: input feature name
    #[serde(default = "default_input_name")]
    pub coreml_input_name: String,
    /// CoreML-specific: output feature name
    #[serde(default = "default_output_name")]
    pub coreml_output_name: String,
}

fn default_provider() -> String {
    "mistralrs".to_string()
}

fn default_input_name() -> String {
    "input".to_string()
}

fn default_output_name() -> String {
    "output".to_string()
}

/// Configuration for RAG processing.
///
/// This covers embedding settings and text processing behavior (chunking, indexing).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RagConfig {
    pub embedding_model: EmbeddingModel,
    #[serde(default)]
    pub indexer: IndexerConfig,
}

/// Configuration for file indexing behavior.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexerConfig {
    /// File extensions to index (e.g., ["rs", "go", "py"])
    /// Empty list (default) means index all readable text files
    #[serde(default)]
    pub extensions: Vec<String>,

    /// Patterns to exclude - skips directories/files containing these strings
    /// Default excludes: build artifacts, version control, package managers, temp files
    #[serde(default = "default_exclude_patterns")]
    pub exclude_patterns: Vec<String>,

    /// Size of text chunks in bytes for splitting documents
    pub chunk_size: usize,

    /// Overlap between consecutive chunks in bytes
    pub chunk_overlap: usize,
}

fn default_exclude_patterns() -> Vec<String> {
    crate::patterns::default_exclude_patterns()
}

fn default_top_k() -> usize {
    5
}

impl Default for IndexerConfig {
    fn default() -> Self {
        Self {
            extensions: Vec::new(), // Empty = index all text files
            exclude_patterns: default_exclude_patterns(),
            chunk_size: 512,
            chunk_overlap: 50,
        }
    }
}

/// Vector database storage mode
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "mode", rename_all = "lowercase")]
pub enum StorageMode {
    /// Embedded storage - runs in-process with zero setup (default)
    Embedded { path: String },
    /// gRPC storage - connect to external vector database server
    Grpc { url: String },
}

impl Default for StorageMode {
    fn default() -> Self {
        Self::Embedded {
            path: "./data/nucleus_vectordb".to_string(),
        }
    }
}

impl Default for RagConfig {
    fn default() -> Self {
        let embedding_model = EmbeddingModel::default();

        let indexer = IndexerConfig {
            extensions: Vec::new(),
            exclude_patterns: default_exclude_patterns(),
            chunk_size: embedding_model.embedding_dim,
            chunk_overlap: 50,
        };

        Self {
            embedding_model,
            indexer,
        }
    }
}

/// Storage configuration for all persistence.
///
/// This includes chat history, tool state, and vector database storage.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageConfig {
    pub chat_history_path: String,
    pub tool_state_path: String,
    /// Vector database storage mode
    #[serde(default)]
    pub storage_mode: StorageMode,
    /// Vector database configuration (collection name, etc.)
    #[serde(default)]
    pub vector_db: VectorDbConfig,
    /// Number of results to return from vector similarity searches
    #[serde(default = "default_top_k")]
    pub top_k: usize,
}

/// Vector database configuration (collection/index name, etc.).
///
/// Provider-agnostic configuration that works with any vector DB backend
/// (Qdrant, LanceDB, etc.).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorDbConfig {
    /// Collection/index name for storing vectors
    pub collection_name: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersonalizationConfig {
    pub learn_from_interactions: bool,
    pub save_conversations: bool,
    pub user_preferences_path: String,
}
impl Default for PersonalizationConfig {
    fn default() -> Self {
        Self {
            learn_from_interactions: true,
            save_conversations: true,
            user_preferences_path: "./data/preferences.json".to_string(),
        }
    }
}

impl Default for VectorDbConfig {
    fn default() -> Self {
        Self {
            collection_name: "nucleus_kb".to_string(),
        }
    }
}

impl Default for StorageConfig {
    fn default() -> Self {
        Self {
            chat_history_path: "./data/history".to_string(),
            tool_state_path: "./data/tool_state".to_string(),
            storage_mode: StorageMode::default(),
            vector_db: VectorDbConfig::default(),
            top_k: default_top_k(),
        }
    }
}

impl Default for LlmConfig {
    fn default() -> Self {
        Self {
            provider: default_provider(),
            model: "MaziyarPanahi/Qwen3-0.6B-GGUF:Qwen3-0.6B.Q4_K_M.gguf".to_string(),
            base_url: "http://localhost:11434".to_string(),
            temperature: 0.6,
            context_length: 32768,
            coreml_input_name: default_input_name(),
            coreml_output_name: default_output_name(),
        }
    }
}

impl Default for Config {
    fn default() -> Self {
        Self {
            llm: LlmConfig::default(),
            system_prompt:
                "You are a helpful AI assistant specializing in programming and development tasks."
                    .to_string(),
            rag: RagConfig::default(),
            storage: StorageConfig::default(),
            personalization: PersonalizationConfig::default(),
            permission: Permission::default(),
        }
    }
}

impl Config {
    /// Load configuration from a YAML file.
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let contents = fs::read_to_string(path)?;
        let mut config: Config = serde_yaml::from_str(&contents)?;

        config.permission = Permission::default();

        Ok(config)
    }

    /// Load configuration from `config.yaml` if it exists, otherwise use defaults.
    pub fn load_or_default() -> Self {
        Self::load("config.yaml").unwrap_or_default()
    }

    /// Create a new Config with default values and builder-style configuration.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the LLM model identifier.
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.llm.model = model.into();
        self
    }

    /// Set the temperature (0.0-2.0).
    pub fn with_temperature(mut self, temperature: f64) -> Self {
        self.llm.temperature = temperature;
        self
    }

    /// Set the system prompt.
    pub fn with_system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = prompt.into();
        self
    }

    /// Set the base URL for the LLM provider.
    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.llm.base_url = url.into();
        self
    }

    /// Set the context length (maximum tokens).
    pub fn with_context_length(mut self, length: usize) -> Self {
        self.llm.context_length = length;
        self
    }

    /// Set the LLM provider type.
    pub fn with_provider(mut self, provider: impl Into<String>) -> Self {
        self.llm.provider = provider.into();
        self
    }

    /// Configure RAG settings.
    pub fn with_rag_config(mut self, rag_config: RagConfig) -> Self {
        self.rag = rag_config;
        self
    }

    /// Configure storage settings.
    pub fn with_storage_config(mut self, storage_config: StorageConfig) -> Self {
        self.storage = storage_config;
        self
    }

    /// Configure personalization settings.
    pub fn with_personalization_config(mut self, personalization_config: PersonalizationConfig) -> Self {
        self.personalization = personalization_config;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_permission_default() {
        let perm = Permission::default();
        assert!(perm.read);
        assert!(perm.write);
        assert!(perm.command);
    }

    #[test]
    fn test_vector_db_config_default() {
        let config = VectorDbConfig::default();
        assert_eq!(config.collection_name, "nucleus_kb");
    }

    #[test]
    fn test_storage_config_defaults() {
        let config = StorageConfig::default();
        assert_eq!(config.chat_history_path, "./data/history");
        assert_eq!(config.tool_state_path, "./data/tool_state");
        assert_eq!(config.vector_db.collection_name, "nucleus_kb");
        assert_eq!(config.top_k, 5);
    }

    #[test]
    fn test_rag_config_defaults() {
        let config = RagConfig::default();
        assert_eq!(config.embedding_model.name, EmbeddingModel::default().name);
    }
}
