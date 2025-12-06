//! Example: Using WriteFilePlugin with the AI
//!
//! Demonstrates how the LLM can write files using the plugin system.

use nucleus_core::{ChatManager, Config};
use nucleus_plugin::{PluginRegistry, Permission};
use nucleus_std::WriteFilePlugin;
use std::sync::Arc;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    println!("Nucleus - WriteFile Plugin Example\n");

    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive("nucleus_core=debug".parse().unwrap())
                .add_directive("mistralrs_core=debug".parse().unwrap())
        )
        .init();
    
    let config = Config::load_or_default();
    
    // Create registry with WRITE permission (required for WriteFilePlugin)
    let mut registry = PluginRegistry::new(Permission::READ_WRITE);
    registry.register(Arc::new(WriteFilePlugin::new()));
    let registry = registry;

    // Use the local Q4_K_M GGUF from Ollama's blob storage
    let manager = ChatManager::builder(config, registry)
        .with_llm_model("Qwen/Qwen3-8B")
        .build()
        .await?;

    println!("Question: Create a file called 'hello.txt' with the content 'Hello from nucleus!'\n");
    
    let response = manager.query(
        "Create a file called 'seth.txt' and put a random a joke in it."
    ).await?;
    
    println!("AI Response:\n{}", response);
    
    Ok(())
}
