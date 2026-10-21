//! Provider abstraction: the [`Provider`] trait and the provider-neutral
//! request, capability and batch types.
//!
//! A provider streams a [`ChatRequest`] as owned [`Event`]s and reports what it
//! and a model can do through [`Capabilities`]. Batch methods default to
//! `Unsupported`, so a provider may implement only [`Provider::chat_stream`].
//!
//! [`Event`]: crate::event::Event
//!
//! See `docs/design/core-api.md` §3 and §7. The legacy provider surface stays in
//! [`types`] until Phase 3.

mod factory;
#[cfg(feature = "hosted")]
pub mod grok;
#[cfg(feature = "local")]
pub mod mistralrs;
pub mod mock;
pub mod types;

// The legacy provider and tool types stay reachable at `provider::types` for the
// hosted and local implementations until Phase 3. They are not re-exported here
// because the new public [`Provider`] trait and request types share their names.
//
// The embedding request/response are still the legacy shapes and are re-exported
// for the new `Provider::embed` signature.
pub use types::{EmbedRequest, EmbedResponse};

// Re-export provider implementations.
pub use factory::create_provider;
#[cfg(feature = "hosted")]
pub use grok::GrokProvider;
#[cfg(feature = "experimental")]
pub use grok::{login as grok_login, logout as grok_logout};
#[cfg(feature = "local")]
pub use mistralrs::MistralRsProvider;

use serde::{Deserialize, Serialize};

// Only the gated `Provider` trait uses these batch types.
#[cfg(any(feature = "hosted", feature = "lancedb"))]
use crate::batch::{BatchHandle, BatchItem, BatchProgress, BatchRequest};
use crate::conversation::ConversationId;
use crate::error::{Error, Result};
use crate::message::Message;
use crate::options::{Effort, RunOptions};

/// Which provider serves a model. The prefix of a [`ModelString`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum ProviderKind {
    /// OpenAI (Responses API).
    OpenAI,
    /// Anthropic (Messages API).
    Anthropic,
    /// xAI Grok.
    Grok,
    /// Gloo's hosted proxy.
    Gloo,
    /// In-process mistral.rs.
    Local,
    /// A provider the caller installed with `AgentBuilder::provider`.
    Custom,
}

impl ProviderKind {
    /// The prefix that selects this provider in a model string.
    pub fn prefix(self) -> &'static str {
        match self {
            ProviderKind::OpenAI => "openai",
            ProviderKind::Anthropic => "anthropic",
            ProviderKind::Grok => "grok",
            ProviderKind::Gloo => "gloo",
            ProviderKind::Local => "local",
            ProviderKind::Custom => "custom",
        }
    }
}

/// A parsed model string, such as `openai/gpt-5.1` or `local/Qwen/Qwen3-8B`.
///
/// The string is split on the **first** `/` only: the first segment names the
/// provider, and the remainder is the provider's own model id, path or
/// `repo:file` spec, passed through verbatim.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct ModelString {
    /// Which provider serves this model.
    pub provider: ProviderKind,
    /// The provider's own model id, path or `repo:file` spec, verbatim.
    pub id: String,
}

impl ModelString {
    /// Parse `provider/id`, splitting on the first `/`. The tail is opaque.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Serialization`] when there is no `/`, the provider
    /// prefix is unknown, or the id is empty.
    pub fn parse(s: &str) -> Result<Self> {
        let (prefix, id) = s.split_once('/').ok_or_else(|| {
            Error::Serialization(format!("model string '{s}' has no provider prefix"))
        })?;
        let provider = match prefix {
            "openai" => ProviderKind::OpenAI,
            "anthropic" => ProviderKind::Anthropic,
            "grok" => ProviderKind::Grok,
            "gloo" => ProviderKind::Gloo,
            "local" => ProviderKind::Local,
            other => {
                return Err(Error::Serialization(format!(
                    "unknown provider prefix '{other}' in model string '{s}'"
                )))
            }
        };
        if id.is_empty() {
            return Err(Error::Serialization(format!(
                "model string '{s}' has an empty model id"
            )));
        }
        Ok(Self {
            provider,
            id: id.to_string(),
        })
    }

    /// Render back to `provider/id`.
    pub fn as_str(&self) -> String {
        format!("{}/{}", self.provider.prefix(), self.id)
    }
}

/// The new provider trait: stream a chat request, report capabilities, and
/// optionally submit batches.
///
/// It replaces the legacy `provider::types::Provider` and returns a stream
/// rather than taking a callback; it never prints.
#[cfg(any(feature = "hosted", feature = "lancedb"))]
#[async_trait::async_trait]
pub trait Provider: Send + Sync {
    /// Which provider this is, for diagnostics and error reporting.
    fn kind(&self) -> ProviderKind;

    /// Stream a chat request. The caller drains the returned stream.
    ///
    /// # Errors
    ///
    /// Returns [`Error`] when the request cannot be started. Errors that arrive
    /// mid-stream travel as [`Event::Error`](crate::event::Event::Error).
    async fn chat_stream(&self, request: ChatRequest) -> Result<crate::event::EventStream>;

    /// Embed texts. Implementations that cannot embed return
    /// [`Error::Unsupported`]; the `Embedder` trait arrives in Phase 4.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Unsupported`] by default.
    async fn embed(&self, request: EmbedRequest) -> Result<EmbedResponse> {
        let _ = request;
        Err(embeddings_unsupported(self.kind()))
    }

    /// What this provider and model can do. Consulted before sending, so an
    /// unsupported request fails locally rather than as a provider 400.
    fn capabilities(&self, model: &str) -> Capabilities;

    // --- Batch: optional methods with an `Unsupported` default. ---

    /// Submit a batch of requests. Unsupported by default.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Unsupported`] by default.
    async fn batch_submit(&self, request: BatchRequest) -> Result<BatchHandle> {
        let _ = request;
        Err(batch_unsupported(self.kind()))
    }

    /// Poll the status of a submitted batch. Unsupported by default.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Unsupported`] by default.
    async fn batch_status(&self, handle: &BatchHandle) -> Result<BatchProgress> {
        let _ = handle;
        Err(batch_unsupported(self.kind()))
    }

    /// Collect the results of a finished batch. Unsupported by default.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Unsupported`] by default.
    async fn batch_results(
        &self,
        handle: &BatchHandle,
    ) -> Result<Vec<(crate::error::RequestId, BatchItem)>> {
        let _ = handle;
        Err(batch_unsupported(self.kind()))
    }

    /// Cancel a submitted batch. Unsupported by default.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Unsupported`] by default.
    async fn batch_cancel(&self, handle: &BatchHandle) -> Result<()> {
        let _ = handle;
        Err(batch_unsupported(self.kind()))
    }
}

/// The `Error::Unsupported` returned by the batch defaults. A free function so
/// no extra method is added to the public trait surface.
#[cfg(any(feature = "hosted", feature = "lancedb"))]
fn batch_unsupported(kind: ProviderKind) -> Error {
    Error::Unsupported {
        feature: "batch".to_string(),
        provider: kind.prefix().to_string(),
        model: None,
    }
}

/// The `Error::Unsupported` returned by the embeddings default.
#[cfg(any(feature = "hosted", feature = "lancedb"))]
fn embeddings_unsupported(kind: ProviderKind) -> Error {
    Error::Unsupported {
        feature: "embeddings".to_string(),
        provider: kind.prefix().to_string(),
        model: None,
    }
}

/// One provider request. Built by the agent, owned, serde.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct ChatRequest {
    /// The model to call.
    pub model: ModelString,
    /// The messages, excluding the system prompt.
    pub messages: Vec<Message>,
    /// Rendered system prompt, owned by the agent and never in `messages`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub system: Option<String>,
    /// Tool definitions. Raw JSON so the provider encodes them its way.
    #[serde(default)]
    pub tools: Vec<ToolSpec>,
    /// How the model may choose a tool.
    pub tool_choice: ToolChoice,
    /// The run options for this request.
    pub options: RunOptions,
    /// The conversation id, for prompt-cache keys and routing headers.
    pub conversation_id: ConversationId,
    /// Extra body fields, passed through unchanged.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider_options: Option<serde_json::Value>,
}

/// A tool definition as sent to a provider.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct ToolSpec {
    /// The model-facing tool name.
    pub name: String,
    /// What the tool does.
    pub description: String,
    /// The full JSON Schema (schemars output, transformed per provider).
    pub parameters: serde_json::Value,
    /// Whether strict mode was requested. MCP tools are never strict.
    pub strict: bool,
}

/// How the model may choose a tool.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub enum ToolChoice {
    /// The model decides.
    Auto,
    /// The model must not call a tool.
    None,
    /// The model must call some tool.
    Required,
    /// The model must call the named tool.
    Tool {
        /// The tool's model-facing name.
        name: String,
    },
}

/// What a provider and a model can do, consulted before sending.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct Capabilities {
    /// The model's context window in tokens, when known.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub context_window: Option<u32>,
    /// The maximum output tokens, when known.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_output_tokens: Option<u32>,
    /// Whether the provider streams.
    pub streaming: bool,
    /// Whether the model accepts tools.
    pub tools: bool,
    /// Whether the model supports native structured output.
    pub structured_output: bool,
    /// How `Thinking::Off` maps on this model.
    pub thinking: ThinkingSupport,
    /// Effort levels the model accepts.
    #[serde(default)]
    pub effort_levels: Vec<Effort>,
    /// Whether the model accepts image inputs.
    pub images: bool,
    /// Whether the model accepts PDF inputs.
    pub pdf: bool,
    /// Batch support for this model, including the discount.
    pub batch: BatchSupport,
    /// Minimum cacheable prefix in tokens, when the provider caches.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_min_tokens: Option<u32>,
}

/// How `Thinking::Off` maps on a model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum ThinkingSupport {
    /// `Off` maps to a per-model value and always succeeds.
    Mappable,
    /// `Off` is an error before sending (Opus 5.5, Fable 5.1).
    AlwaysOn,
    /// The provider has no thinking to configure.
    NotApplicable,
}

/// Batch support for a model.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum BatchSupport {
    /// The model does not support batch.
    Unsupported,
    /// Supported, with the batch discount as a fraction (0.5 = 50% off).
    Supported {
        /// The discount as a fraction of the synchronous price.
        discount: f32,
    },
    /// Not known without a live call; treat as unsupported.
    Unknown,
}

#[cfg(all(test, any(feature = "hosted", feature = "lancedb")))]
mod tests {
    use super::*;

    /// A provider that implements only the required methods and relies on the
    /// `embeddings` and `batch` defaults. If this compiles, a custom provider
    /// needs no downcast and no extra trait.
    struct StubProvider;

    #[async_trait::async_trait]
    impl Provider for StubProvider {
        fn kind(&self) -> ProviderKind {
            ProviderKind::Custom
        }

        async fn chat_stream(&self, _request: ChatRequest) -> Result<crate::event::EventStream> {
            Ok(Box::pin(futures::stream::empty()))
        }

        fn capabilities(&self, _model: &str) -> Capabilities {
            Capabilities {
                context_window: None,
                max_output_tokens: None,
                streaming: true,
                tools: false,
                structured_output: false,
                thinking: ThinkingSupport::NotApplicable,
                effort_levels: Vec::new(),
                images: false,
                pdf: false,
                batch: BatchSupport::Unsupported,
                cache_min_tokens: None,
            }
        }
    }

    #[tokio::test]
    async fn a_stub_provider_only_needs_chat_stream() {
        let provider: std::sync::Arc<dyn Provider> = std::sync::Arc::new(StubProvider);
        // The batch methods default to `Unsupported` without any downcast.
        let request = BatchRequest {
            id: crate::error::RequestId("r1".to_string()),
            request: ChatRequest {
                model: ModelString::parse("local/stub").expect("parses"),
                messages: Vec::new(),
                system: None,
                tools: Vec::new(),
                tool_choice: ToolChoice::Auto,
                options: RunOptions::default(),
                conversation_id: ConversationId::new(),
                provider_options: None,
            },
        };
        let error = provider
            .batch_submit(request)
            .await
            .expect_err("batch is unsupported by default");
        assert!(matches!(error, Error::Unsupported { .. }));

        let stream = provider
            .chat_stream(ChatRequest {
                model: ModelString::parse("local/stub").expect("parses"),
                messages: Vec::new(),
                system: None,
                tools: Vec::new(),
                tool_choice: ToolChoice::Auto,
                options: RunOptions::default(),
                conversation_id: ConversationId::new(),
                provider_options: None,
            })
            .await
            .expect("chat_stream starts");
        let _ = stream;
    }

    #[test]
    fn model_string_splits_on_the_first_slash() {
        let model = ModelString::parse("local/Qwen/Qwen3-8B").expect("parses");
        assert_eq!(model.provider, ProviderKind::Local);
        assert_eq!(model.id, "Qwen/Qwen3-8B");
        assert_eq!(model.as_str(), "local/Qwen/Qwen3-8B");
    }

    #[test]
    fn model_string_rejects_a_missing_prefix() {
        assert!(ModelString::parse("gpt-5.1").is_err());
        assert!(ModelString::parse("unknown/x").is_err());
    }
}
