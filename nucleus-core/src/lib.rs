#![recursion_limit = "256"]

//! nucleus-core - Core AI engine infrastructure
//!
//! Provides the foundational components for building AI-powered applications:
//! - LLM provider abstraction (mistral.rs, Grok, and others)
//! - RAG (Retrieval Augmented Generation)
//! - Configuration management
//!
//! ## Primary API
//!
//! Users should interact with nucleus via `ChatManager`.

// Public modules
pub mod chat;
pub mod config;
pub mod models;
pub mod patterns;
pub mod provider;
pub mod rag;

// Public exports
pub use chat::{ChatManager, ChatManagerBuilder, CommandEffect};
pub use config::{Config, IndexerConfig};
pub use rag::RagEngine;

// Provider exports
pub use provider::{
    grok_login, grok_logout, ChatRequest, ChatResponse, GrokProvider, Message, Provider,
    ProviderError, Tool, ToolCall, ToolCallFunction, ToolFunction,
};
