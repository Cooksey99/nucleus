use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ConfigError {
    #[error("Failed to read config file: {0}")]
    FileRead(#[from] std::io::Error),

    #[error("Failed to parse config: {0}")]
    Parse(#[from] serde_yaml::Error),
}

pub type Result<T> = std::result::Result<T, ConfigError>;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub llm: LlmConfig,
    pub system_prompt: String,
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
    pub model: String,
    pub base_url: String,
    pub temperature: f64,
    pub context_length: usize,
}

/// Configuration for RAG.
///
/// This includes the actual RAG settings, as well as the storage type (eg. Embedded, gRPC, etc.)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RagConfig {
    pub embedding_model: String,
    pub chunk_size: usize,
    pub chunk_overlap: usize,
    pub top_k: usize,
    #[serde(default)]
    pub indexer: IndexerConfig,
    /// Vector database storage mode
    #[serde(default)]
    pub storage_mode: StorageMode,
    /// Vector database configuration (collection name, etc.)
    #[serde(default)]
    pub vector_db: VectorDbConfig,
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
}

fn default_exclude_patterns() -> Vec<String> {
    crate::patterns::default_exclude_patterns()
}

impl Default for IndexerConfig {
    fn default() -> Self {
        Self {
            extensions: Vec::new(), // Empty = index all text files
            exclude_patterns: default_exclude_patterns(),
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
        Self {
            embedding_model: "nomic-embed-text".to_string(),
            chunk_size: 512,
            chunk_overlap: 50,
            top_k: 5,
            indexer: IndexerConfig::default(),
            storage_mode: StorageMode::default(),
            vector_db: VectorDbConfig::default(),
        }
    }
}

/// General storage configuration for non-RAG persistence.
///
/// This covers chat history, user preferences, tool state, etc.
/// RAG vector database is configured separately in `RagConfig`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageConfig {
    pub chat_history_path: String,
    pub tool_state_path: String,
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
        }
    }
}

impl Default for LlmConfig {
    fn default() -> Self {
        Self {
            model: "qwen3:0.6b".to_string(),
            base_url: "http://localhost:11434".to_string(),
            temperature: 0.6,
            context_length: 32768,
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
        config.rag = RagConfig::default();
        config.personalization = PersonalizationConfig::default();

        Ok(config)
    }

    /// Load configuration from `config.yaml` if it exists, otherwise use defaults.
    pub fn load_or_default() -> Self {
        Self::load("config.yaml").unwrap_or_default()
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
    }

    #[test]
    fn test_rag_config_with_vector_db() {
        let config = RagConfig::default();
        assert_eq!(config.vector_db.collection_name, "nucleus_kb");
        assert_eq!(config.embedding_model, "nomic-embed-text");
    }
}
