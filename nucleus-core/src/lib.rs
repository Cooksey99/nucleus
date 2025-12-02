//! nucleus-core - Core AI engine infrastructure
//!
//! Provides the foundational components for building AI-powered applications:
//! - LLM integration (Ollama)
//! - RAG (Retrieval Augmented Generation)
//! - Configuration management
//! - Server API (primary interface)
//!
//! ## Primary API
//!
//! Users should interact with nucleus via the `Server` API.

// Public modules
pub mod chat;
pub mod config;
pub mod detection;
pub mod patterns;
pub mod provider;
pub mod rag;
pub mod server;

// Public modules (for advanced use)
pub mod ollama;

// Public exports
pub use chat::ChatManager;
pub use config::{Config, IndexerConfig};
pub use detection::{check_ollama_silent, detect_ollama, DetectionError, OllamaInfo};
pub use ollama::Client;
pub use provider::Provider;
pub use rag::Rag;
pub use server::Server;
