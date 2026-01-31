use async_trait::async_trait;
use nucleus_plugin::{Permission, Plugin, PluginError, PluginOutput, Result};
use serde::Deserialize;
use serde_json::Value;
use std::path::{Path, PathBuf};
use schemars::{JsonSchema, schema_for};

/// Plugin for reading file contents.
pub struct ReadFilePlugin;
pub struct WriteFilePlugin;

#[derive(Debug, Deserialize, JsonSchema)]
struct ReadFileParams {
    /// Absolute or relative path to the file to read
    path: String,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct WriteFileParams {
    /// Absolute or relative path to the file to write to
    #[serde(default)]
    path: PathBuf,
    /// Content to write to the file
    content: String,
}

impl ReadFilePlugin {
    pub fn new() -> Self {
        Self
    }

    pub async fn read(&self, path: &Path) -> Result<PluginOutput> {
        let input = serde_json::json!({
            "path": path
        });
        self.execute(input).await
    }
}

impl WriteFilePlugin {
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl Plugin for ReadFilePlugin {
    fn name(&self) -> &str {
        "read_file"
    }

    fn description(&self) -> &str {
        "Read the contents of a file"
    }

    fn parameter_schema(&self) -> Value {
        let schema = schema_for!(ReadFileParams);
        serde_json::to_value(schema).unwrap_or_default()
    }

    fn required_permission(&self) -> Permission {
        Permission::READ_ONLY
    }

    async fn execute(&self, input: Value) -> Result<PluginOutput> {
        let params: ReadFileParams = serde_json::from_value(input)
            .map_err(|e| PluginError::InvalidInput(format!("Invalid parameters: {}", e)))?;

        let path = PathBuf::from(&params.path);

        // Read file
        let content = tokio::fs::read_to_string(&path)
            .await
            .map_err(|e| PluginError::ExecutionFailed(format!("Failed to read file: {}", e)))?;

        // Log the operation
        println!("Read file: {}", path.display());

        Ok(PluginOutput::new(content))
    }
}

#[async_trait]
impl Plugin for WriteFilePlugin {
    fn name(&self) -> &str {
        "write_file"
    }

    fn description(&self) -> &str {
        "Write changes or additions to file"
    }

    fn parameter_schema(&self) -> Value {
        let schema = schema_for!(WriteFileParams);
        serde_json::to_value(schema).unwrap_or_default()
    }

    fn required_permission(&self) -> Permission {
        Permission::READ_WRITE
    }

    async fn execute(&self, input: Value) -> Result<PluginOutput> {
        let params: WriteFileParams = serde_json::from_value(input)
            .map_err(|e| PluginError::InvalidInput(format!("Invalid parameters: {}", e)))?;

        let path = PathBuf::from(&params.path);

        tokio::fs::write(&path, &params.content)
            .await
            .map_err(|e| PluginError::ExecutionFailed(format!("Failed to write file: {}", e)))?;

        println!("Wrote file: {}", path.display());

        Ok(PluginOutput::new(format!(
            "Successfully wrote {} bytes to {}",
            params.content.len(),
            path.display()
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_read_file() {
        // Create a temporary file
        let temp_dir = std::env::temp_dir();
        let test_file = temp_dir.join("nucleus_test_read.txt");
        let test_content = "Hello, nucleus!";

        std::fs::write(&test_file, test_content).unwrap();

        // Test reading it
        let plugin = ReadFilePlugin::new();
        let input = serde_json::json!({
            "path": test_file.to_str().unwrap()
        });

        let result = plugin.execute(input).await.unwrap();
        assert_eq!(result.content, test_content);

        // Cleanup
        std::fs::remove_file(test_file).ok();
    }

    #[tokio::test]
    async fn test_read_nonexistent_file() {
        let plugin = ReadFilePlugin::new();
        let input = serde_json::json!({
            "path": "/nonexistent/file.txt"
        });

        let result = plugin.execute(input).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_write_file() {
        let temp_dir = std::env::temp_dir();
        let test_file = temp_dir.join("nucleus_test_write.txt");
        let test_content = "Written by nucleus!";

        let plugin = WriteFilePlugin::new();
        let input = serde_json::json!({
            "path": test_file.to_str().unwrap(),
            "content": test_content
        });

        let result = plugin.execute(input).await.unwrap();
        assert!(result.content.contains("Successfully wrote"));

        let written_content = std::fs::read_to_string(&test_file).unwrap();
        assert_eq!(written_content, test_content);

        std::fs::remove_file(test_file).ok();
    }

    #[tokio::test]
    async fn test_write_file_creates_file() {
        let temp_dir = std::env::temp_dir();
        let test_file = temp_dir.join("nucleus_test_new_file.txt");

        std::fs::remove_file(&test_file).ok();

        let plugin = WriteFilePlugin::new();
        let input = serde_json::json!({
            "path": test_file.to_str().unwrap(),
            "content": "New file content"
        });

        plugin.execute(input).await.unwrap();
        assert!(test_file.exists());

        std::fs::remove_file(test_file).ok();
    }
}
