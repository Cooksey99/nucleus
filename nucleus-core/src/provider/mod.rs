//! LLM provider abstraction layer.
//!
//! This module defines a common interface for different LLM backends
//! (Ollama, mistral.rs, etc.) to provide chat completions and embeddings.

mod types;
pub mod ollama;
pub mod mistralrs;

// Re-export common types
pub use types::{
    Provider,
    ProviderError,
    Result,
    ChatRequest,
    ChatResponse,
    Message,
    Tool,
    ToolCall,
    ToolFunction,
    ToolCallFunction,
    EmbedRequest,
    EmbedResponse,
};

// Re-export provider implementations
pub use ollama::OllamaProvider;
// pub use mistralrs::MistralRsProvider;  // Uncomment when implemented

