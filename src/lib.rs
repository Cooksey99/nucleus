#![recursion_limit = "256"]
#![forbid(unsafe_code)]
#![cfg_attr(
    not(test),
    deny(clippy::unwrap_used, clippy::expect_used, clippy::panic)
)]

//! nucleus - Privacy-first, modular AI engine
//!
//! Library for local and hosted LLM calls, with tools and retrieval.
//!
//! # Features
//!
//! - `hosted` (default): Grok and other hosted providers
//! - `std` (default): built-in file, search, and exec tools
//! - `local`: in-process mistral.rs
//! - `rag`: retrieval. Also enables `local` and, until Phase 4, `lancedb`
//! - `lancedb`: LanceDB vector store
//! - `experimental`: Grok account login
//! - `mcp`: reserved for Phase 5
//! - `metal`: Apple Silicon acceleration for mistral.rs
//! - `cuda`: NVIDIA acceleration for mistral.rs

pub mod chat;
pub mod config;
pub mod models;
pub mod patterns;
pub mod provider;
#[cfg(feature = "rag")]
pub mod rag;
pub mod tool;

#[cfg(feature = "std")]
pub mod tools;

pub use chat::{ChatManager, ChatManagerBuilder, CommandEffect};
pub use config::{Config, IndexerConfig};
#[cfg(feature = "hosted")]
pub use provider::GrokProvider;
#[cfg(feature = "local")]
pub use provider::MistralRsProvider;
#[cfg(feature = "experimental")]
pub use provider::{grok_login, grok_logout};
pub use provider::{
    ChatRequest, ChatResponse, Message, Provider, ProviderError, Tool, ToolCall, ToolCallFunction,
    ToolFunction,
};
#[cfg(feature = "rag")]
pub use rag::RagEngine;
pub use tool::{Plugin, PluginError, PluginOutput, PluginRegistry};

#[cfg(feature = "std")]
pub use tools::{ExecPlugin, ReadFilePlugin, SearchPlugin, WriteFilePlugin};

/// Convenient imports.
pub mod prelude {
    pub use crate::{ChatManager, ChatManagerBuilder, Config, Plugin, PluginRegistry, Provider};

    #[cfg(feature = "std")]
    pub use crate::{ExecPlugin, ReadFilePlugin, SearchPlugin, WriteFilePlugin};
}
