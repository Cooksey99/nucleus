//! LLM provider abstraction layer.
//!
//! This module defines a common interface for different LLM backends
//! (mistral.rs, Grok, and others) to provide chat completions and embeddings.

mod factory;
#[cfg(feature = "hosted")]
pub mod grok;
pub mod mock;
#[cfg(feature = "local")]
pub mod mistralrs;
mod types;

// Re-export common types
pub use types::{
    ChatRequest, ChatResponse, EmbedRequest, EmbedResponse, Message, Provider, ProviderError,
    ProviderType, Result, StructuredOutput, Tool, ToolCall, ToolCallFunction, ToolFunction,
};

// Re-export provider implementations
pub use factory::create_provider;
#[cfg(feature = "hosted")]
pub use grok::GrokProvider;
#[cfg(feature = "experimental")]
pub use grok::{login as grok_login, logout as grok_logout};
#[cfg(feature = "local")]
pub use mistralrs::MistralRsProvider;
