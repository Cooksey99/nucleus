#![recursion_limit = "256"]
#![forbid(unsafe_code)]
#![deny(missing_docs)]
// `nucleus::Error` is deliberately a large enum with structured payloads
// (core-api.md §6); boxing its variants would change the approved signatures.
#![allow(clippy::result_large_err)]
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
//! - `local`: in-process models (`local//path.gguf`)
//! - `rag`: retrieval. Also enables `local` and, until Phase 4, `lancedb`
//! - `lancedb`: LanceDB vector store
//! - `experimental`: Grok account login
//! - `mcp`: reserved for Phase 5
//! - `metal`: Apple Silicon acceleration for local models
//! - `cuda`: NVIDIA acceleration for local models
//!
//! The module layout follows `docs/design/core-api.md` §2.

pub mod agent;
pub mod batch;
pub mod config;
pub mod conversation;
pub mod error;
pub mod event;
pub mod extract;
pub mod message;
pub mod models;
pub mod options;
pub mod patterns;
pub mod provider;
#[cfg(feature = "rag")]
pub mod rag;
pub mod store;
pub mod tool;

#[cfg(feature = "std")]
pub mod tools;

// The core error type and the new conversation model.
pub use conversation::{Conversation, ConversationId};
pub use error::{Error, ProviderError, RequestId, Result};
pub use event::{Cost, CostSource, Event, ModelPrice, PriceTable, StopReason, Usage};
pub use message::{KnowledgeSource, Message, Part, ProviderItem, Role, ToolCallId, ToolOutput};
pub use options::{CacheTtl, Effort, RunOptions, Thinking, ThinkingDisplay};
pub use store::{ConversationStore, MemoryStore};

// The agent, and the turn-level data types it returns.
pub use agent::{Agent, AgentBuilder, Approval, KnowledgeMode, ToolCall, Turn};

#[cfg(any(feature = "hosted", feature = "lancedb"))]
pub use event::EventStream;

pub use config::{Config, IndexerConfig};
#[cfg(feature = "hosted")]
pub use provider::GrokProvider;
#[cfg(feature = "experimental")]
pub use provider::{grok_login, grok_logout};

// The new provider abstraction: the `Provider` trait with its request,
// capability and batch types (core-api.md §3, §7).
pub use batch::{BatchHandle, BatchItem, BatchPart, BatchProgress, BatchRequest};
pub use provider::{
    BatchSupport, Capabilities, ChatRequest, ModelString, ProviderKind, ThinkingSupport,
    ToolChoice, ToolSpec,
};
// `Provider::chat_stream` returns `EventStream`, so the trait carries the same
// feature gate and is absent from a build with neither `hosted` nor `lancedb`.
#[cfg(any(feature = "hosted", feature = "lancedb"))]
pub use provider::Provider;
// The model-facing tool trait and its adapters, and the legacy plugin surface.
pub use tool::{tool_fn, tool_fn_sync, Tool, ToolAnnotations, ToolContext};
pub use tool::{Plugin, PluginError, PluginOutput, PluginRegistry};

#[cfg(feature = "rag")]
pub use rag::RagEngine;

#[cfg(feature = "std")]
pub use tools::{ExecPlugin, ReadFilePlugin, SearchPlugin, WriteFilePlugin};

/// One-shot completion against a hosted model.
///
/// `model` is a `provider/model` string, such as `"openai/gpt-5.1"`. This is the
/// library's only top-level function; everything else hangs off the agent.
///
/// See `docs/design/core-api.md` §11.1.
///
/// # Errors
///
/// Returns [`Error`] when the model cannot be resolved or the call fails.
pub async fn complete(model: impl AsRef<str>, prompt: impl AsRef<str>) -> Result<String> {
    let _ = (model.as_ref(), prompt.as_ref());
    todo!("one-shot completion is implemented with the core agent")
}

/// Convenient imports.
pub mod prelude {
    pub use crate::{
        complete, Agent, Config, Conversation, ConversationStore, Error, Message, Plugin,
        PluginRegistry, Result, Role, RunOptions, Usage,
    };

    #[cfg(any(feature = "hosted", feature = "lancedb"))]
    pub use crate::Provider;

    #[cfg(feature = "std")]
    pub use crate::{ExecPlugin, ReadFilePlugin, SearchPlugin, WriteFilePlugin};
}
