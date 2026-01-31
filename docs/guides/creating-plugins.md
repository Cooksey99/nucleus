# Creating Plugins

Plugins extend nucleus with custom capabilities. From the LLM's perspective, plugins appear as "tools" that can be called during conversation. This guide walks you through creating your own plugins.

## Overview

A plugin is a Rust struct that implements the `Plugin` trait. Each plugin:
- Has a unique name the LLM uses to call it
- Defines parameters via JSON Schema (auto-generated from Rust structs)
- Declares required permissions (read, write, execute)
- Implements async execution logic

## Quick Start

Let's create a simple plugin that counts lines in a file:

```rust
use async_trait::async_trait;
use nucleus_plugin::{Permission, Plugin, PluginError, PluginOutput, Result};
use schemars::{JsonSchema, schema_for};
use serde::Deserialize;
use serde_json::Value;
use std::path::PathBuf;

pub struct LineCountPlugin;

#[derive(Debug, Deserialize, JsonSchema)]
struct LineCountParams {
    /// Path to the file to count lines in
    path: String,
}

impl LineCountPlugin {
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl Plugin for LineCountPlugin {
    fn name(&self) -> &str {
        "count_lines"
    }

    fn description(&self) -> &str {
        "Count the number of lines in a file"
    }

    fn parameter_schema(&self) -> Value {
        let schema = schema_for!(LineCountParams);
        serde_json::to_value(schema).unwrap()
    }

    fn required_permission(&self) -> Permission {
        Permission::READ_ONLY
    }

    async fn execute(&self, input: Value) -> Result<PluginOutput> {
        let params: LineCountParams = serde_json::from_value(input)
            .map_err(|e| PluginError::InvalidInput(format!("Invalid parameters: {}", e)))?;

        let path = PathBuf::from(&params.path);
        let content = tokio::fs::read_to_string(&path)
            .await
            .map_err(|e| PluginError::ExecutionFailed(format!("Failed to read file: {}", e)))?;

        let line_count = content.lines().count();

        Ok(PluginOutput::new(format!("File has {} lines", line_count)))
    }
}
```

## Step-by-Step Guide

### 1. Define Your Plugin Struct

Create a struct to hold any plugin state:

```rust
pub struct MyPlugin {
    // Optional: configuration or state
    config: MyConfig,
}

impl MyPlugin {
    pub fn new() -> Self {
        Self {
            config: MyConfig::default(),
        }
    }
}
```

For stateless plugins, use a unit struct:

```rust
pub struct MyPlugin;
```

### 2. Define Parameter Schema

Create a struct for your plugin's parameters using `serde` and `schemars`:

```rust
use schemars::JsonSchema;
use serde::Deserialize;

#[derive(Debug, Deserialize, JsonSchema)]
struct MyPluginParams {
    /// Description that the LLM will see
    required_param: String,
    
    /// Optional parameter with default
    #[serde(default)]
    optional_param: Option<String>,
    
    /// Parameter with custom default value
    #[serde(default = "default_timeout")]
    timeout: u64,
}

fn default_timeout() -> u64 {
    30
}
```

**Key Points:**
- Use `///` doc comments - they become parameter descriptions in the schema
- Be specific in descriptions - help the LLM understand when to use each parameter
- Use `#[serde(default)]` for optional parameters
- Use `#[serde(default = "function")]` for custom defaults

### 3. Implement the Plugin Trait

Implement the five required methods:

```rust
#[async_trait]
impl Plugin for MyPlugin {
    fn name(&self) -> &str {
        "my_plugin"  // Used by LLM to call this plugin
    }

    fn description(&self) -> &str {
        "A clear, concise description of what this plugin does. \
         Include when the LLM should use it and what it returns."
    }

    fn parameter_schema(&self) -> Value {
        let schema = schema_for!(MyPluginParams);
        serde_json::to_value(schema).unwrap()
    }

    fn required_permission(&self) -> Permission {
        // Choose based on what your plugin does:
        // Permission::READ_ONLY - only reads data
        // Permission::READ_WRITE - reads and writes files
        // Permission::ALL - reads, writes, and executes commands
        Permission::READ_ONLY
    }

    async fn execute(&self, input: Value) -> Result<PluginOutput> {
        // Parse parameters
        let params: MyPluginParams = serde_json::from_value(input)
            .map_err(|e| PluginError::InvalidInput(format!("Invalid parameters: {}", e)))?;

        // Your plugin logic here
        let result = do_something(&params).await
            .map_err(|e| PluginError::ExecutionFailed(e.to_string()))?;

        // Return output
        Ok(PluginOutput::new(result))
    }
}
```

### 4. Handle Errors Properly

Use the appropriate error types:

```rust
// Invalid input from LLM
PluginError::InvalidInput("Expected positive number".to_string())

// Execution failed (file not found, network error, etc.)
PluginError::ExecutionFailed("Failed to connect to database".to_string())

// Permission denied
PluginError::PermissionDenied("Cannot write to system directory".to_string())

// Other errors
PluginError::Other("Unexpected error".to_string())
```

### 5. Return Rich Output

You can return plain text or include metadata:

```rust
// Simple text output
Ok(PluginOutput::new("Operation completed"))

// With metadata (useful for tool chaining)
Ok(PluginOutput::new("Found 5 matches")
    .with_metadata(serde_json::json!({
        "count": 5,
        "matches": ["file1.rs", "file2.rs"]
    })))
```

## Real-World Examples

### File Reader Plugin

See `nucleus-std/src/files.rs` for a complete example:

```rust
#[derive(Debug, Deserialize, JsonSchema)]
struct ReadFileParams {
    /// Absolute or relative path to the file to read
    path: String,
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
        serde_json::to_value(schema).unwrap()
    }

    fn required_permission(&self) -> Permission {
        Permission::READ_ONLY
    }

    async fn execute(&self, input: Value) -> Result<PluginOutput> {
        let params: ReadFileParams = serde_json::from_value(input)
            .map_err(|e| PluginError::InvalidInput(format!("Invalid parameters: {}", e)))?;

        let content = tokio::fs::read_to_string(&params.path)
            .await
            .map_err(|e| PluginError::ExecutionFailed(format!("Failed to read file: {}", e)))?;

        Ok(PluginOutput::new(content))
    }
}
```

### Search Plugin

See `nucleus-std/src/search.rs` for a more complex example with multiple parameters and regex support.

## Testing Your Plugin

Write tests to verify your plugin works:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_my_plugin() {
        let plugin = MyPlugin::new();
        
        let input = serde_json::json!({
            "required_param": "test_value"
        });

        let result = plugin.execute(input).await;
        assert!(result.is_ok());
        
        let output = result.unwrap();
        assert!(output.content.contains("expected"));
    }

    #[tokio::test]
    async fn test_invalid_input() {
        let plugin = MyPlugin::new();
        
        let input = serde_json::json!({
            // Missing required parameter
        });

        let result = plugin.execute(input).await;
        assert!(result.is_err());
    }
}
```

## Registering Your Plugin

To make your plugin available to the LLM, register it with the `PluginRegistry`:

```rust
use nucleus_plugin::PluginRegistry;

let mut registry = PluginRegistry::new();
registry.register(Box::new(MyPlugin::new()));
```

## Best Practices

### 1. Clear Descriptions

Write descriptions that help the LLM understand:
- **What** the plugin does
- **When** to use it
- **What** it returns

```rust
fn description(&self) -> &str {
    "Search for text patterns in files. Use this when you need to find \
     specific code, function names, or text across multiple files. \
     Returns matching lines with file paths and line numbers."
}
```

### 2. Useful Parameter Descriptions

```rust
/// The shell command to execute (e.g., "git status", "ls -la")
command: String,

/// Working directory for command execution (defaults to current directory)
cwd: Option<PathBuf>,
```

### 3. Validate Input

Check parameters before execution:

```rust
if params.timeout == 0 {
    return Err(PluginError::InvalidInput("Timeout must be greater than 0".to_string()));
}

if !params.path.exists() {
    return Err(PluginError::ExecutionFailed("Path does not exist".to_string()));
}
```

### 4. Use Appropriate Permissions

- `READ_ONLY` - File/data reading, search operations
- `READ_WRITE` - File modifications, database writes
- `ALL` - Command execution, system operations

### 5. Handle Async Properly

Use `tokio` for async operations:

```rust
use tokio::fs;
use tokio::process::Command;

// Async file I/O
let content = fs::read_to_string(&path).await?;

// Async command execution
let output = Command::new("git")
    .args(["status"])
    .output()
    .await?;
```

### 6. Provide Context in Errors

```rust
.map_err(|e| PluginError::ExecutionFailed(
    format!("Failed to read file '{}': {}", path.display(), e)
))?;
```

## Plugin Locations

- **`nucleus-std`** - General-purpose plugins (files, search, commands)
- **`nucleus-dev`** - Developer-specific plugins (LSP, git, testing)
- **Custom crates** - Your own plugin collections

## Next Steps

- Check out existing plugins in `nucleus-std/src/` for examples
- Read the [Plugin System Architecture](../architecture/plugin-system.md)
- Learn about [Tool-Augmented LLM](../architecture/tool-augmented.md) design
- See [API Reference: Plugin Trait](../api/plugin-trait.md) for complete details
