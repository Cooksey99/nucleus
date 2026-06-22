//! Example demonstrating RAG indexing with 5 specific knowledge items.
//!
//! Keeps setup intentionally minimal: load config, enable default RAG, index data, query data.

use nucleus_core::config::StorageMode;
use nucleus_core::{ChatManager, Config};
use nucleus_plugin::{Permission, PluginRegistry};

const VECTOR_DB_PATH: &str = "./data/nucleus_vectordb_rag_indexing";
const COLLECTION_NAME: &str = "nucleus_kb_rag_indexing_example";

const KNOWLEDGE_ITEMS: [(&str, &str); 5] = [
    (
        "Nucleus Architecture",
        "Nucleus is built on a modular plugin architecture. Core components include the ChatManager for session management, PluginRegistry for plugin discovery and loading, and Config for runtime settings. The system uses an event-driven design with async/await throughout for high concurrency.",
    ),
    (
        "Vector Database Storage",
        "Nucleus supports embedded Qdrant for vector storage with persistence to disk. Documents are chunked into overlapping windows of 512 tokens with a 50-token overlap, then embedded using sentence-transformers models. The default embedding model is all-minilm:l6-v2 with 384 dimensions.",
    ),
    (
        "RAG Pipeline",
        "The RAG pipeline in Nucleus involves: 1) Document ingestion and chunking, 2) Embedding generation via Ollama, 3) Vector storage in Qdrant, 4) Semantic search for relevant chunks, 5) Context augmentation of LLM prompts. Hybrid search combining semantic and keyword matching is available.",
    ),
    (
        "Plugin System",
        "Plugins can extend Nucleus with new capabilities. Each plugin must implement the Plugin trait with name(), description(), and execute() methods. Plugins can be written in Rust and loaded dynamically, or connected via gRPC for remote plugins. Permission levels control read/write access.",
    ),
    (
        "Configuration Options",
        "Nucleus configuration is YAML-based. Key settings include: storage mode (embedded/grpc), vector DB collection name, embedding model selection, chunk size and overlap, and the list of enabled plugins. The config file is typically located at ~/.nucleus/config.yaml or ./nucleus.yaml.",
    ),
];

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let mut config = Config::load_or_default().with_default_rag();
    config.storage.storage_mode = StorageMode::Embedded {
        path: VECTOR_DB_PATH.to_string(),
    };
    config.storage.vector_db.collection_name = COLLECTION_NAME.to_string();

    let _ = std::fs::remove_dir_all(VECTOR_DB_PATH);
    std::fs::create_dir_all(VECTOR_DB_PATH)?;

    let manager = ChatManager::new(config, PluginRegistry::new(Permission::READ_WRITE)).await?;
    println!("RAG enabled: {}", manager.is_rag_enabled());

    manager.clear_knowledge_base().await?;
    for (title, content) in KNOWLEDGE_ITEMS {
        manager.index_text(content, title).await?;
    }
    println!("Indexed {} documents.", manager.knowledge_base_count().await);

    let queries = [
        "What is the architecture of Nucleus?",
        "How does Nucleus handle vector storage?",
        "Describe the RAG pipeline steps.",
        "How do plugins work in Nucleus?",
        "Where is the Nucleus configuration stored and what options are available?",
    ];

    for query in queries {
        let rag_context = manager.preview_rag_context(query).await?;
        let rag_used = rag_context
            .lines()
            .any(|line| line.trim_start().starts_with("[1]"));

        println!("\nQ: {}", query);
        println!("RAG_CONTEXT_USED={}", rag_used);

        let response = manager.query(None, query).await?;
        println!("A: {}", response.trim());
    }

    Ok(())
}
