#![recursion_limit = "256"]

//! nucleus - Privacy-first, modular AI engine
//!
//! Library for local and hosted LLM calls, with tools and retrieval.
//!
//! # Features
//!
//! - `std` (default): built-in file, search, and exec tools
//! - `metal`: Apple Silicon acceleration for mistral.rs
//! - `cuda`: NVIDIA acceleration for mistral.rs

pub mod chat;
pub mod config;
pub mod models;
pub mod patterns;
pub mod provider;
pub mod rag;
pub mod tool;

#[cfg(feature = "std")]
pub mod tools;

pub use chat::{ChatManager, ChatManagerBuilder, CommandEffect};
pub use config::{Config, IndexerConfig};
pub use provider::{
    grok_login, grok_logout, ChatRequest, ChatResponse, GrokProvider, Message, Provider,
    ProviderError, Tool, ToolCall, ToolCallFunction, ToolFunction,
};
pub use rag::RagEngine;
pub use tool::{Permission, Plugin, PluginError, PluginOutput, PluginRegistry};

#[cfg(feature = "std")]
pub use tools::{ExecPlugin, ReadFilePlugin, SearchPlugin, WriteFilePlugin};

/// Convenient imports.
pub mod prelude {
    pub use crate::{
        ChatManager, ChatManagerBuilder, Config, Permission, Plugin, PluginRegistry, Provider,
    };

    #[cfg(feature = "std")]
    pub use crate::{ExecPlugin, ReadFilePlugin, SearchPlugin, WriteFilePlugin};
}
