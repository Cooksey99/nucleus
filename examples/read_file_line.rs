//! Example: Using ChatManager to have AI analyze files
//!
//! Demonstrates the complete flow:
//! 1. User asks a question about a file
//! 2. AI decides to use the read_file plugin
//! 3. Plugin reads the file
//! 4. AI analyzes and responds

use nucleus_core::{ChatManager, Config};
use nucleus_plugin::{Permission, PluginRegistry};
use nucleus_std::ReadFilePlugin;
use std::sync::Arc;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize tracing subscriber
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive("nucleus_core=debug".parse().unwrap()),
        )
        .init();

    println!("Nucleus - AI + Plugin Example\n");

    println!("Current dir: {:?}\n", std::env::current_dir()?);

    let config = Config::load_or_default();

    let mut registry = PluginRegistry::new(Permission::READ_ONLY);
    registry.register(ReadFilePlugin::new());

    let manager = ChatManager::new(config, registry).await?;

    println!("Question: What's on line 7 of config.yaml?\n");

    let response = manager.query("What's on line 7 of config.yaml?").await?;

    println!("AI Response:\n{}", response);

    Ok(())
}
