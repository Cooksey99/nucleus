//! Refactor Assistant - Real-World Integration Example
//!
//! This example demonstrates how a developer would actually use nucleus
//! to build an AI-powered refactoring assistant.
//!
//! ## What It Demonstrates
//!
//! - Creating a `ChatManager` with custom plugins
//! - Registering tools (file operations, search, etc.)
//! - Sending queries that trigger tool execution
//! - The LLM automatically deciding which tools to call
//! - Tool chaining (search → read → analyze → write)
//!
//! ## Real API Usage
//!
//! This uses nucleus's actual public API:
//! - `ChatManager::new()` - Core conversation orchestration
//! - `PluginRegistry` - Tool/plugin management
//! - Built-in plugins from `nucleus_std`
//! - Tool-augmented LLM flow with automatic tool calling

use async_trait::async_trait;
use nucleus_core::{ChatManager, Config};
use nucleus_plugin::{Permission, Plugin, PluginRegistry};
use nucleus_std::{ReadFilePlugin, SearchPlugin, WriteFilePlugin};
use serde_json::json;
use std::path::{Path, PathBuf};
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Refactor Assistant Example ===\n");

    let test_dir = setup_test_environment()?;
    println!("Created test environment at: {}\n", test_dir.display());

    // Step 1: Load configuration (model, temperature, etc.)
    let config = Config::default();
    println!("Configuration:");
    println!("  Model: {}", config.llm.model);
    println!("  Temperature: {}\n", config.llm.temperature);

    // Step 2: Create plugin registry with appropriate permissions
    let mut registry = PluginRegistry::new(Permission::READ_WRITE);

    // Step 3: Register plugins (tools the LLM can use)
    registry.register(Arc::new(ReadFilePlugin::new()));
    registry.register(Arc::new(WriteFilePlugin::new()));

    println!("Registered plugins:");
    for plugin in registry.all() {
        println!("  - {}: {}", plugin.name(), plugin.description());
    }
    println!();

    // Step 4: Create ChatManager (orchestrates LLM + tools)
    let manager = ChatManager::new(config, registry).await?;

    // Step 5: Ask the AI to perform refactoring
    println!("Query: Read config.rs and create a backup\n");

    let query = format!(
        "Read the file at {} and create a backup at {}",
        test_dir.join("config.rs").display(),
        test_dir.join("config.rs.bak").display()
    );

    match manager.query(&query).await {
        Ok(response) => {
            println!("AI Response:\n{}", response);
        }
        Err(e) => {
            eprintln!("Error: {}", e);
            eprintln!("\nNote: This example requires Ollama running with a model that supports tool calling.");
            eprintln!(
                "Install Ollama from https://ollama.ai and run: ollama pull qwen2.5-coder:7b"
            );
        }
    }

    cleanup_test_environment(&test_dir)?;

    Ok(())
}

fn setup_test_environment() -> Result<PathBuf, Box<dyn std::error::Error>> {
    let test_dir = std::env::temp_dir().join("nucleus_refactor_demo");
    std::fs::create_dir_all(&test_dir)?;

    std::fs::write(
        test_dir.join("config.rs"),
        r#"
pub struct Config {
    pub host: String,
    pub port: u16,
    pub timeout: u64,
}

impl Config {
    pub fn new(host: String, port: u16, timeout: u64) -> Self {
        Self { host, port, timeout }
    }
    
    pub fn default() -> Self {
        Self {
            host: "localhost".to_string(),
            port: 8080,
            timeout: 30,
        }
    }
}
"#,
    )?;

    std::fs::write(
        test_dir.join("server.rs"),
        r#"
pub struct ServerBuilder {
    host: String,
    port: u16,
}

impl ServerBuilder {
    pub fn new() -> Self {
        Self {
            host: "localhost".to_string(),
            port: 3000,
        }
    }
    
    pub fn host(mut self, host: impl Into<String>) -> Self {
        self.host = host.into();
        self
    }
    
    pub fn port(mut self, port: u16) -> Self {
        self.port = port;
        self
    }
    
    pub fn build(self) -> Server {
        Server {
            host: self.host,
            port: self.port,
        }
    }
}

pub struct Server {
    host: String,
    port: u16,
}
"#,
    )?;

    Ok(test_dir)
}

fn register_plugins(registry: &mut PluginRegistry) {
    registry.register(Arc::new(ReadFilePlugin::new()));
    registry.register(Arc::new(WriteFilePlugin::new()));
    registry.register(Arc::new(SearchPlugin::new()));
    registry.register(Arc::new(MockSemanticSearchPlugin));
    registry.register(Arc::new(MockCommandExecutor));
}

fn list_plugins(registry: &PluginRegistry) {
    for plugin in registry.all() {
        let perm = plugin.required_permission();
        let perm_str = format_permission(&perm);
        println!("  - {} [{}]", plugin.name(), perm_str);
        println!("    {}", plugin.description());
    }
}

fn format_permission(perm: &Permission) -> String {
    match (perm.read, perm.write, perm.execute) {
        (true, false, false) => "READ",
        (true, true, false) => "READ+WRITE",
        (true, true, true) => "ALL",
        _ => "NONE",
    }
    .to_string()
}

async fn demo_file_operations(
    registry: &PluginRegistry,
    test_dir: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("--- Demo: File Operations ---");

    let read_plugin = registry
        .get("read_file")
        .expect("read_file plugin not found");

    let config_path = test_dir.join("config.rs");
    let input = json!({
        "path": config_path.to_str().unwrap()
    });

    println!("Reading file: {}", config_path.display());
    let output = read_plugin.execute(input).await?;
    println!("Read {} bytes\n", output.content.len());

    let backup_path = test_dir.join("config.rs.bak");
    let write_plugin = registry
        .get("write_file")
        .expect("write_file plugin not found");

    let backup_input = json!({
        "path": backup_path.to_str().unwrap(),
        "content": output.content
    });

    println!("Creating backup: {}", backup_path.display());
    write_plugin.execute(backup_input).await?;
    println!("Backup created successfully\n");

    Ok(())
}

async fn demo_search_patterns(
    registry: &PluginRegistry,
    test_dir: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("--- Demo: Search for Patterns ---");

    let search_plugin = registry
        .get("semantic_search")
        .expect("semantic_search plugin not found");

    let input = json!({
        "query": "builder pattern",
        "path": test_dir.to_str().unwrap()
    });

    println!("Searching for: builder pattern");
    let output = search_plugin.execute(input).await?;
    println!("Found matches:\n{}\n", output.content);

    Ok(())
}

async fn demo_tool_composition(
    registry: &PluginRegistry,
    test_dir: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("--- Demo: Tool Composition (Search → Read → Analyze) ---");

    println!("Step 1: Search for builder patterns");
    let search_plugin = registry.get("semantic_search").unwrap();
    let search_result = search_plugin
        .execute(json!({
            "query": "builder pattern",
            "path": test_dir.to_str().unwrap()
        }))
        .await?;

    println!("Found: {}", search_result.content);

    println!("\nStep 2: Read the matching file");
    let read_plugin = registry.get("read_file").unwrap();
    let server_path = test_dir.join("server.rs");
    let read_result = read_plugin
        .execute(json!({
            "path": server_path.to_str().unwrap()
        }))
        .await?;

    println!("Read {} bytes from server.rs", read_result.content.len());

    println!("\nStep 3: Propose refactoring (simulated)");
    println!("In real implementation, LLM would:");
    println!("  - Analyze config.rs structure");
    println!("  - Compare with ServerBuilder pattern from server.rs");
    println!("  - Generate ConfigBuilder implementation");
    println!("  - Write refactored code with proper error handling");

    println!("\nStep 4: Validate with tests (simulated)");
    let exec_plugin = registry.get("execute_command").unwrap();
    let exec_result = exec_plugin
        .execute(json!({
            "command": "cargo test",
            "cwd": test_dir.to_str().unwrap()
        }))
        .await?;

    println!("{}", exec_result.content);

    Ok(())
}

fn cleanup_test_environment(test_dir: &Path) -> Result<(), Box<dyn std::error::Error>> {
    std::fs::remove_dir_all(test_dir)?;
    println!("\nCleaned up test environment");
    Ok(())
}

struct MockSemanticSearchPlugin;

#[async_trait]
impl Plugin for MockSemanticSearchPlugin {
    fn name(&self) -> &str {
        "semantic_search"
    }

    fn description(&self) -> &str {
        "Search codebase for semantically similar patterns using vector embeddings"
    }

    fn parameter_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "required": ["query", "path"],
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Semantic search query (e.g. 'builder pattern', 'error handling')"
                },
                "path": {
                    "type": "string",
                    "description": "Directory to search in"
                }
            }
        })
    }

    fn required_permission(&self) -> Permission {
        Permission::READ_ONLY
    }

    async fn execute(
        &self,
        _input: serde_json::Value,
    ) -> nucleus_plugin::Result<nucleus_plugin::PluginOutput> {
        use nucleus_plugin::PluginOutput;

        Ok(PluginOutput::new(
            "Found 1 match: server.rs (ServerBuilder pattern)",
        ))
    }
}

struct MockCommandExecutor;

#[async_trait]
impl Plugin for MockCommandExecutor {
    fn name(&self) -> &str {
        "execute_command"
    }

    fn description(&self) -> &str {
        "Execute shell commands with permission controls"
    }

    fn parameter_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "required": ["command"],
            "properties": {
                "command": {
                    "type": "string",
                    "description": "Command to execute"
                },
                "cwd": {
                    "type": "string",
                    "description": "Working directory for command execution"
                }
            }
        })
    }

    fn required_permission(&self) -> Permission {
        Permission::ALL
    }

    async fn execute(
        &self,
        _input: serde_json::Value,
    ) -> nucleus_plugin::Result<nucleus_plugin::PluginOutput> {
        use nucleus_plugin::PluginOutput;

        Ok(PluginOutput::new("✓ cargo test passed (simulated)\n   Running 12 tests\n   test result: ok. 12 passed; 0 failed"))
    }
}
