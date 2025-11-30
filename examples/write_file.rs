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
    
    let config = Config::load_or_default();
    
    // Create registry with WRITE permission (required for WriteFilePlugin)
    let mut registry = PluginRegistry::new(Permission::READ_WRITE);
    registry.register(Arc::new(WriteFilePlugin::new()));
    let registry = Arc::new(registry);

    let manager = ChatManager::new(config, registry);

    println!("Question: Create a file called 'hello.txt' with the content 'Hello from nucleus!'\n");
    
    let response = manager.query(
        "Create a file called 'seth.txt' and put a random a joke in it."
    ).await?;
    
    println!("AI Response:\n{}", response);
    
    Ok(())
}
