use async_trait::async_trait;
use serde_json::Value;
use std::fmt;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum PluginError {
    #[error("Invalid input: {0}")]
    InvalidInput(String),

    #[error("Execution failed: {0}")]
    ExecutionFailed(String),

    #[error("Permission denied: {0}")]
    PermissionDenied(String),

    #[error("Plugin error: {0}")]
    Other(String),
}

pub type Result<T> = std::result::Result<T, PluginError>;

/// Permissions required by a plugin.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Permission {
    pub read: bool,
    pub write: bool,
    pub execute: bool,
}

impl Permission {
    pub const READ_ONLY: Self = Self {
        read: true,
        write: false,
        execute: false,
    };

    pub const READ_WRITE: Self = Self {
        read: true,
        write: true,
        execute: false,
    };

    pub const ALL: Self = Self {
        read: true,
        write: true,
        execute: true,
    };

    pub const NONE: Self = Self {
        read: false,
        write: false,
        execute: false,
    };

    /// Check if this permission allows the required permission.
    pub fn allows(&self, required: &Permission) -> bool {
        (!required.read || self.read)
            && (!required.write || self.write)
            && (!required.execute || self.execute)
    }
}

/// Output from plugin execution.
#[derive(Debug, Clone)]
pub struct PluginOutput {
    pub content: String,
    pub metadata: Option<Value>,
}

impl PluginOutput {
    pub fn new(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            metadata: None,
        }
    }

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
/// plugins appear as "tools" that can be called during conversation.
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

    /// Permissions required to execute this plugin.
    /// Used to enforce security boundaries.
    fn required_permission(&self) -> Permission;

    /// Execute the plugin with given input parameters.
    /// The input should match the parameter schema.
    ///
    /// **This is the actual function the LLM will use to call a tool**
    async fn execute(&self, input: Value) -> Result<PluginOutput>;
}
