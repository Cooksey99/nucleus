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
    
    // Check if Qdrant is running, try to start it if not
    match nucleus_core::qdrant_helper::ensure_qdrant_running(&config.storage.qdrant.url).await {
        Ok(_) => println!("✓ Connected to Qdrant at {}\n", config.storage.qdrant.url),
        Err(e) => {
            eprintln!("{}", e);
            
            // Try to auto-start on Windows
            #[cfg(target_os = "windows")]
            {
                let qdrant_path = "C:\\Users\\frigont\\Desktop\\qdrant.exe";
                if let Ok(_) = nucleus_core::qdrant_helper::try_start_qdrant(qdrant_path) {
                    // Wait a bit more for startup
                    tokio::time::sleep(tokio::time::Duration::from_secs(3)).await;
                    
                    // Try connecting again
                    nucleus_core::qdrant_helper::ensure_qdrant_running(&config.storage.qdrant.url).await?;
                    println!("✓ Connected to Qdrant\n");
                } else {
                    return Err(e);
                }
            }
            
            #[cfg(not(target_os = "windows"))]
            return Err(e);
        }
    }
    
    println!("Configuration:");
    println!("  Qdrant URL: {}", config.storage.qdrant.url);
    println!("  Collection: {}", config.storage.qdrant.collection_name);
    println!("  Embedding Model: {}", config.rag.embedding_model);
    println!("  Chunk Size: {} bytes", config.rag.chunk_size);
    println!("  Chunk Overlap: {} bytes\n", config.rag.chunk_overlap);
    
    // Create chat manager with empty plugin registry
    let registry = Arc::new(PluginRegistry::new(Permission::READ_WRITE));
    let manager = ChatManager::new(config.clone(), registry).await;
    
    // Check current count
    let doc_count = manager.knowledge_base_count().await;
    println!("Current knowledge base: {} documents\n", doc_count);
    
    // Example: Index the src directory
    if doc_count == 0 {
        println!("=== Indexing Example ===");
        println!("Indexing nucleus-core/src directory...");
        println!("This may take a minute...\n");
        
        match manager.index_directory("./nucleus-core/src").await {
            Ok(count) => {
                println!("✓ Indexed {} files!", count);
                println!("  Total documents: {}\n", manager.knowledge_base_count().await);
            }
            Err(e) => {
                eprintln!("Warning: Could not index directory: {}", e);
                eprintln!("Error chain: {:?}", e);
                eprintln!("Make sure Ollama is running and the embedding model is installed.\n");
            }
        }
    }
    
    // Example: Query the knowledge base
    println!("=== Query Example ===");
    println!("Asking: 'Where is the index_directory function implemented?'\n");
    
    let response = manager.query(
        "In which file and module is the index_directory function implemented? What does it do?"
    ).await?;
    
    println!("AI Response:\n{}\n", response);
    
    println!("\n✓ Example complete!");
    println!("\nVector database: {} @ {}", config.storage.qdrant.collection_name, config.storage.qdrant.url);
    println!("Knowledge base: {} documents", manager.knowledge_base_count().await);
    println!("All indexed data persists in Qdrant.");
    
    Ok(())
}
