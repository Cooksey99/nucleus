//! Example demonstrating RAG indexing with embedded storage.
//!
//! This example shows how to:
//! - Index directories into the RAG vector database
//! - Configure persistent storage location
//! - Query the indexed knowledge base

use nucleus_core::{ChatManager, Config};
use nucleus_plugin::{Permission, PluginRegistry};
use std::path::Path;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    println!("Nucleus - RAG Indexing Example");
    println!("==============================\n");

    let config = Config::load_or_default();
    print_rag_config(&config);

    let registry = PluginRegistry::new(Permission::READ_WRITE);
    let manager = ChatManager::new(config.clone(), registry).await?;

    let doc_count = manager.knowledge_base_count().await;
    println!("Current knowledge base: {} documents\n", doc_count);

    if doc_count == 0 {
        index_example_directory(&manager).await;
    }

    query_example(&manager).await?;

    print_summary(&config, manager.knowledge_base_count().await);

    Ok(())
}

fn print_rag_config(config: &Config) {
    println!("RAG Configuration:");
    match &config.storage.storage_mode {
        nucleus_core::config::StorageMode::Embedded { path } => {
            println!("  Storage: Embedded at {}", path);
        }
        nucleus_core::config::StorageMode::Grpc { url } => {
            println!("  Storage: Remote gRPC @ {}", url);
        }
    }
    println!("  Collection: {}", config.storage.vector_db.collection_name);
    println!("  Embedding: {}", config.rag.embedding_model.name);
    println!();
}

async fn index_example_directory(manager: &ChatManager) {
    println!("=== Indexing Example ===");
    println!("Indexing nucleus-core/src directory...\n");

    let path = Path::new("./nucleus-core/src");
    match manager.index_directory(path).await {
        Ok(count) => {
            let total = manager.knowledge_base_count().await;
            println!("\n✓ Indexed {} files ({} total documents)\n", count, total);
        }
        Err(e) => {
            eprintln!("⚠ Could not index directory: {}", e);
            eprintln!("  Make sure Ollama is running and the embedding model is installed.\n");
        }
    }
}

async fn query_example(manager: &ChatManager) -> anyhow::Result<()> {
    println!("=== Query Example ===");
    println!("Query: 'Where is the index_directory function implemented?'\n");

    let response = manager.query(
        "In which file and module is the index_directory function implemented? What does it do?"
    ).await?;

    println!("Response:\n{}\n", response);
    Ok(())
}

fn print_summary(config: &Config, doc_count: usize) {
    println!("=== Summary ===");
    match &config.storage.storage_mode {
        nucleus_core::config::StorageMode::Embedded { path } => {
            println!(
                "Collection '{}' at {}",
                config.storage.vector_db.collection_name, path
            );
        }
        nucleus_core::config::StorageMode::Grpc { url } => {
            println!(
                "Collection '{}' @ {}",
                config.storage.vector_db.collection_name, url
            );
        }
    }
    println!("{} documents indexed", doc_count);
    println!("Data persists across restarts");
}
