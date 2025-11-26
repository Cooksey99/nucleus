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
pub mod rag;
pub mod server;

// Private module (implementation detail)
mod ollama;

// Public exports
pub use chat::ChatManager;
pub use config::Config;
pub use server::Server;
