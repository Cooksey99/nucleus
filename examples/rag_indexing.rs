//! Example demonstrating RAG indexing with 5 specific knowledge items.
//!
//! This example indexes exactly 5 distinct pieces of data and demonstrates
//! querying them using RAG (Retrieval-Augmented Generation).

use anyhow::Context;
use nucleus_core::{ChatManager, Config};
use nucleus_plugin::{Permission, PluginRegistry};

/// The 5 specific bits of data to index for RAG
const KNOWLEDGE_ITEMS: [(&str, &str); 5] = [
    ("Nucleus Architecture",
     "Nucleus is built on a modular plugin architecture. Core components include the ChatManager for session management, PluginRegistry for plugin discovery and loading, and Config for runtime settings. The system uses an event-driven design with async/await throughout for high concurrency."),
    ("Vector Database Storage",
     "Nucleus supports embedded Qdrant for vector storage with persistence to disk. Documents are chunked into overlapping windows of 512 tokens with a 50-token overlap, then embedded using sentence-transformers models. The default embedding model is all-minilm:l6-v2 with 384 dimensions."),
    ("RAG Pipeline",
     "The RAG pipeline in Nucleus involves: 1) Document ingestion and chunking, 2) Embedding generation via Ollama, 3) Vector storage in Qdrant, 4) Semantic search for relevant chunks, 5) Context augmentation of LLM prompts. Hybrid search combining semantic and keyword matching is available."),
    ("Plugin System",
     "Plugins can extend Nucleus with new capabilities. Each plugin must implement the Plugin trait with name(), description(), and execute() methods. Plugins can be written in Rust and loaded dynamically, or connected via gRPC for remote plugins. Permission levels control read/write access."),
    ("Configuration Options",
     "Nucleus configuration is YAML-based. Key settings include: storage mode (embedded/grpc), vector DB collection name, embedding model selection, chunk size and overlap, and the list of enabled plugins. The config file is typically located at ~/.nucleus/config.yaml or ./nucleus.yaml."),
];

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    println!("Nucleus - RAG Indexing Example (5 Specific Knowledge Items)");
    println!("===========================================================\n");

    let config = Config::load_or_default();
    print_rag_config(&config);

    let registry = PluginRegistry::new(Permission::READ_WRITE);
    let manager = ChatManager::new(config.clone(), registry).await?;

    // Clear existing knowledge base for clean demo
    manager.clear_knowledge_base().await?;

    // Index our 5 specific items
    println!("=== Indexing 5 Specific Knowledge Items ===\n");
    for (title, content) in KNOWLEDGE_ITEMS.iter() {
        println!("Indexing: {}...", title);
        manager.index_text(content, title).await
            .context("Unable to index knowledge base")?;
        println!(" Indexed successfully");
    }

    let total = manager.knowledge_base_count().await;
    println!("\n All 5 knowledge items indexed ({} total documents)\n", total);

    // Query each of the 5 items
    println!("=== Querying Each Knowledge Item ===\n");
    let queries = [
        "What is the architecture of Nucleus?",
        "How does Nucleus handle vector storage?",
        "Describe the RAG pipeline steps.",
        "How do plugins work in Nucleus?",
        "Where is the Nucleus configuration stored and what options are available?",
    ];

    for (i, query) in queries.iter().enumerate() {
        println!("Query {}: {}", i + 1, query);
        let response = manager.query(None, query).await?;
        println!("  Response: {}\n", response.trim());
    }

    // Final summary
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
    if let Some(rag_config) = &config.rag {
        println!("  Embedding: {}", rag_config.embedding_model.name);
    }
    println!();
}

fn print_summary(config: &Config, doc_count: usize) {
    println!("\n=== Summary ===");
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
    println!("All 5 specific knowledge items are now available for RAG queries!");
}
