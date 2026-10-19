use async_trait::async_trait;
use serde_json::Value;
use std::fmt;
use thiserror::Error;

/// Errors from a [`Plugin`] call.
#[derive(Debug, Error)]
pub enum PluginError {
    /// The arguments did not match the plugin's parameter schema.
    #[error("Invalid input: {0}")]
    InvalidInput(String),

    /// The plugin ran but failed.
    #[error("Execution failed: {0}")]
    ExecutionFailed(String),

    /// Any other plugin failure.
    #[error("Plugin error: {0}")]
    Other(String),
}

/// Result type for plugin operations.
pub type Result<T> = std::result::Result<T, PluginError>;

/// Output from plugin execution.
#[derive(Debug, Clone)]
pub struct PluginOutput {
    /// The text the model reads.
    pub content: String,
    /// Optional structured metadata produced by the plugin.
    pub metadata: Option<Value>,
}

impl PluginOutput {
    /// A plugin output carrying only text.
    pub fn new(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            metadata: None,
        }
    }

    /// Attach structured metadata to this output.
    pub fn with_metadata(mut self, metadata: Value) -> Self {
        self.metadata = Some(metadata);
        self
    }
}

impl fmt::Display for PluginOutput {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.content)
    }
}

/// Core trait that all plugins must implement.
///
/// Plugins extend nucleus with custom capabilities. From the LLM's perspective,
/// plugins appear as tools that can be called during conversation.
#[async_trait]
pub trait Plugin: Send + Sync {
    /// Unique identifier for this plugin.
    /// This is what the LLM will use to call the plugin.
    fn name(&self) -> &str;

    /// Human-readable description of what this plugin does.
    /// Included in the LLM prompt to help it decide when to use this plugin.
    fn description(&self) -> &str;

    /// JSON schema defining the plugin's parameters.
    /// This tells the LLM what arguments the plugin expects.
    fn parameter_schema(&self) -> Value;

    /// Execute the plugin with given input parameters.
    /// The input should match the parameter schema.
    async fn execute(&self, input: Value) -> Result<PluginOutput>;
}
