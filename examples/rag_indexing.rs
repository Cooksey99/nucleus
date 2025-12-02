//! Example demonstrating RAG indexing with persistent storage.
//!
//! This example shows how to:
//! - Index directories into the RAG vector database
//! - Configure persistent storage location
//! - Query the indexed knowledge base

use nucleus_core::{ChatManager, Config};
use nucleus_plugin::{PluginRegistry, Permission};
use std::sync::Arc;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    println!("Nucleus - RAG Indexing Example");
    println!("==============================\n");
    
    // Load or create config
    let mut config = Config::load_or_default();
    
    // Configure where to store the vector database
    config.storage.vector_db_path = "./data/vectordb".to_string();
    
    println!("Configuration:");
    println!("  Vector DB Path: {}/vector_store.json", config.storage.vector_db_path);
    println!("  Embedding Model: {}", config.rag.embedding_model);
    println!("  Chunk Size: {} bytes", config.rag.chunk_size);
    println!("  Chunk Overlap: {} bytes\n", config.rag.chunk_overlap);
    
    // Create chat manager with empty plugin registry
    let registry = Arc::new(PluginRegistry::new(Permission::READ_WRITE));
    let manager = ChatManager::new(config.clone(), registry);
    
    // Load previously indexed documents
    println!("Loading knowledge base...");
    let loaded = manager.load_knowledge_base().await?;
    if loaded > 0 {
        println!("✓ Loaded {} documents from previous session\n", loaded);
    } else {
        println!("  No existing documents (first run)\n");
    }
    
    // Check current count
    let doc_count = manager.knowledge_base_count();
    println!("Current knowledge base: {} documents\n", doc_count);
    
    // Example: Index the src directory
    if doc_count == 0 {
        println!("=== Indexing Example ===");
        println!("Indexing nucleus-core/src directory...");
        println!("This may take a minute...\n");
        
        match manager.index_directory("./nucleus-core/src").await {
            Ok(count) => {
                println!("✓ Indexed {} files!", count);
                println!("  Total documents: {}\n", manager.knowledge_base_count());
            }
            Err(e) => {
                eprintln!("Warning: Could not index directory: {}", e);
                eprintln!("Make sure Ollama is running and the embedding model is installed.\n");
            }
        }
    }
    
    // Example: Query the knowledge base
    println!("=== Query Example ===");
    println!("Asking: 'What is the RAG system?'\n");
    
    let response = manager.query(
        "Based on the codebase, what is the RAG system and how does it work?"
    ).await?;
    
    println!("AI Response:\n{}\n", response);
    
    println!("\n✓ Example complete!");
    println!("\nVector database location: {}/vector_store.json", config.storage.vector_db_path);
    println!("Knowledge base: {} documents", manager.knowledge_base_count());
    println!("All indexed data persists across runs.");
    
    Ok(())
}
