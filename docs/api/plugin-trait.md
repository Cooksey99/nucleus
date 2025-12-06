# Plugin Trait

The `Plugin` trait is the foundation of nucleus's extensibility. By implementing this trait, you create custom tools that the AI can use during conversations.

## Overview

```rust
#[async_trait]
pub trait Plugin: Send + Sync {
    fn name(&self) -> &str;
    fn description(&self) -> &str;
    fn parameter_schema(&self) -> Value;
    fn required_permission(&self) -> Permission;
    async fn execute(&self, input: Value) -> Result<PluginOutput>;
}
```

**Module**: `nucleus-plugin`

## Trait Methods

### `name(&self) -> &str`

Unique identifier for this plugin. This is what the LLM uses to call the plugin.

```rust
fn name(&self) -> &str {
    "read_file"
}
```

**Guidelines**:
- Use `snake_case` naming
- Keep names concise but descriptive
- Names should be unique across all registered plugins

### `description(&self) -> &str`

Human-readable description of what the plugin does. This is included in the LLM prompt to help it decide when to use this plugin.

```rust
fn description(&self) -> &str {
    "Read the contents of a file from the filesystem"
}
```

**Guidelines**:
- Be specific about what the plugin does
- Mention key constraints or limitations
- Help the LLM understand when to use this tool

### `parameter_schema(&self) -> Value`

JSON schema defining the plugin's input parameters. This tells the LLM what arguments the plugin expects.

```rust
fn parameter_schema(&self) -> Value {
    json!({
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to the file to read"
            }
        },
        "required": ["path"]
    })
}
```

**Format**: Standard JSON Schema (draft-07)

**Guidelines**:
- Include descriptions for all parameters
- Mark required vs optional fields
- Provide examples in descriptions when helpful

### `required_permission(&self) -> Permission`

Permissions required to execute this plugin. Used to enforce security boundaries.

```rust
fn required_permission(&self) -> Permission {
    Permission::READ_ONLY
}
```

**Available Permissions**:
- `Permission::READ_ONLY` - Read files and directories
- `Permission::READ_WRITE` - Read + write files
- `Permission::ALL` - Read + write + execute commands
- `Permission::NONE` - No special permissions

### `execute(&self, input: Value) -> Result<PluginOutput>`

Executes the plugin with given input parameters. This is the actual implementation of your tool.

```rust
async fn execute(&self, input: Value) -> Result<PluginOutput> {
    let path = input["path"].as_str()
        .ok_or_else(|| PluginError::InvalidInput("Missing 'path' parameter".into()))?;
    
    let contents = tokio::fs::read_to_string(path).await
        .map_err(|e| PluginError::ExecutionFailed(e.to_string()))?;
    
    Ok(PluginOutput::new(contents))
}
```

**Parameters**:
- `input`: JSON value matching your `parameter_schema()`

**Returns**:
- `Ok(PluginOutput)`: Successful execution with result
- `Err(PluginError)`: Execution failed

## PluginOutput

```rust
pub struct PluginOutput {
    pub content: String,
    pub metadata: Option<Value>,
}
```

### Creating Output

```rust
// Simple text output
PluginOutput::new("Result text")

// With metadata
PluginOutput::new("Result text")
    .with_metadata(json!({
        "lines": 42,
        "size_bytes": 1024
    }))
```

The `content` field is returned to the LLM for processing. The optional `metadata` can be used for logging, debugging, or returning structured data to the API consumer (future feature).

## Complete Example

Here's a complete plugin implementation:

```rust
use nucleus_plugin::{Plugin, PluginOutput, Permission, Result};
use async_trait::async_trait;
use serde_json::{json, Value};

pub struct ReadFilePlugin;

impl ReadFilePlugin {
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
        "Read the complete contents of a file. Returns the file contents as text."
    }
    
    fn parameter_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file (relative or absolute)"
                }
            },
            "required": ["path"]
        })
    }
    
    fn required_permission(&self) -> Permission {
        Permission::READ_ONLY
    }
    
    async fn execute(&self, input: Value) -> Result<PluginOutput> {
        let path = input["path"].as_str()
            .ok_or_else(|| PluginError::InvalidInput(
                "Missing 'path' parameter".into()
            ))?;
        
        let contents = tokio::fs::read_to_string(path).await
            .map_err(|e| PluginError::ExecutionFailed(
                format!("Failed to read file: {}", e)
            ))?;
        
        Ok(PluginOutput::new(contents))
    }
}
```

## Usage

Once implemented, register your plugin with the `PluginRegistry`:

```rust
use nucleus_plugin::{PluginRegistry, Permission};
use std::sync::Arc;

let mut registry = PluginRegistry::new(Permission::READ_ONLY);
registry.register(Arc::new(ReadFilePlugin::new()));
```

## Best Practices

### 1. **Error Handling**

Provide clear error messages that help the LLM understand what went wrong:

```rust
Err(PluginError::ExecutionFailed(
    format!("File not found: {} (current directory: {:?})", path, env::current_dir()?)
))
```

### 2. **Input Validation**

Validate inputs thoroughly before execution:

```rust
let path = input["path"].as_str()
    .ok_or_else(|| PluginError::InvalidInput("'path' must be a string".into()))?;

if path.is_empty() {
    return Err(PluginError::InvalidInput("'path' cannot be empty".into()));
}
```

### 3. **Async-First**

Use async I/O for all operations:

```rust
// Good: async I/O
let contents = tokio::fs::read_to_string(path).await?;

// Bad: blocking I/O
let contents = std::fs::read_to_string(path)?;
```

### 4. **Structured Metadata**

Use metadata for debugging and logging:

```rust
Ok(PluginOutput::new(result)
    .with_metadata(json!({
        "execution_time_ms": elapsed.as_millis(),
        "result_size": result.len()
    })))
```

## Design Decisions to Consider

### 1. **Parameter Schema Format**

Should we support alternatives to JSON Schema?

**Options**:
- A: JSON Schema only (current approach)
- B: Add support for TypeScript types
- C: Rust macro to auto-generate schema from struct

### 2. **Streaming Execution**

Should plugins support streaming output for long-running operations?

**Options**:
- A: Add `async fn execute_stream(&self, input: Value) -> impl Stream<Item = String>`
- B: Keep execution atomic (current approach)
- C: Add `Progress` callback parameter

### 3. **Plugin State**

Should plugins have mutable state between executions?

**Options**:
- A: Stateless plugins (current approach)
- B: Add `&mut self` for stateful plugins
- C: Separate `StatefulPlugin` trait

### 4. **Permission Granularity**

Are current permissions sufficient?

**Current**: `read`, `write`, `execute`

**Possible additions**:
- Network access permission
- File path restrictions (allow list/deny list)
- Resource limits (CPU, memory, time)

## See Also

- [PluginRegistry](./plugin-registry.md) - Managing and executing plugins
- [Creating Plugins Guide](../guides/creating-plugins.md) - Step-by-step plugin development
- [Standard Plugins](../guides/standard-plugins.md) - Included plugin implementations
