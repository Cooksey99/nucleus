//! nucleus-core - Core AI engine infrastructure
//!
//! Provides the foundational components for building AI-powered applications:
//! - LLM provider abstraction (Ollama, mistral.rs, etc.)
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

// Public exports
pub use chat::ChatManager;
pub use config::{Config, IndexerConfig};
pub use detection::{check_ollama_silent, detect_ollama, DetectionError, OllamaInfo};
pub use rag::Rag;
pub use server::Server;

// Provider exports
pub use provider::{
    Provider,
    ChatRequest,
    ChatResponse,
    Message,
    Tool,
    ToolCall,
    ToolFunction,
    ToolCallFunction,
    ProviderError,
};
