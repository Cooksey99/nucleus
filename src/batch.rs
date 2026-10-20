//! Batch requests: [`BatchRequest`], [`BatchHandle`] and the progress and
//! per-request result types.
//!
//! Batch support is a set of optional [`Provider`] methods that default to
//! `Unsupported`; support is reported per model in
//! [`Capabilities::batch`](crate::provider::Capabilities::batch). The types here
//! are the serializable, versioned handle shape, so a handle can be saved and a
//! batch resumed later. The provider implementations arrive in Phase 3.
//!
//! [`Provider`]: crate::provider::Provider
//!
//! See `docs/design/core-api.md` §10.2 and `docs/design/spikes/batch.md` §6.

use serde::{Deserialize, Serialize};

use crate::error::{Error, RequestId};
use crate::event::Usage;
use crate::message::Message;
use crate::provider::{ChatRequest, ProviderKind};

/// One request in a batch: an id plus the provider-neutral request to run.
///
/// The id is passed to the provider unchanged as the `custom_id`
/// (`batch_request_id` on xAI), so results match on it without a translation
/// table. IDs must be unique within a batch.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct BatchRequest {
    /// The caller's id for this request, unique within the batch.
    pub id: RequestId,
    /// The request to run.
    pub request: ChatRequest,
}

/// A serializable, versioned handle to a submitted batch.
///
/// Credentials are never stored here; resuming takes the handle plus the same
/// provider configuration. `version` lets the shape change later.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct BatchHandle {
    /// The handle format version. Starts at 1.
    pub version: u32,
    /// The provider the batch runs on.
    pub provider: ProviderKind,
    /// The full model string, e.g. `anthropic/claude-opus-5-5`.
    pub model: String,
    /// The endpoint, to detect resuming against the wrong base URL.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub base_url: Option<String>,
    /// One entry per provider batch; 0.2 creates exactly one.
    #[serde(default)]
    pub parts: Vec<BatchPart>,
    /// The submitted request ids, in submission order.
    #[serde(default)]
    pub request_ids: Vec<RequestId>,
    /// Creation time, RFC 3339.
    pub created_at: String,
    /// The processing deadline, RFC 3339, when the provider gave one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expires_at: Option<String>,
}

/// One provider-side batch inside a [`BatchHandle`].
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct BatchPart {
    /// The provider's batch id (`batch_…`, `msgbatch_…`).
    pub batch_id: String,
    /// The uploaded input file id (OpenAI, xAI file mode), for cleanup.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input_file_id: Option<String>,
    /// The provider endpoint the batch targets (OpenAI).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub endpoint: Option<String>,
}

/// How many requests in a batch have reached each state.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
#[non_exhaustive]
pub struct BatchProgress {
    /// Total requests in the batch.
    pub total: u64,
    /// Requests that succeeded.
    pub succeeded: u64,
    /// Requests that failed.
    pub failed: u64,
    /// Requests cancelled, when the provider counts them.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cancelled: Option<u64>,
    /// Requests that expired, when the provider counts them.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expired: Option<u64>,
    /// Requests still pending.
    pub pending: u64,
}

/// The result of one request in a batch.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub enum BatchItem {
    /// The request succeeded, carrying the assistant message and its usage.
    Succeeded {
        /// The assistant message the request produced.
        message: Message,
        /// Token usage and cost for the request.
        usage: Usage,
    },
    /// The request failed.
    Failed(Error),
    /// The request was cancelled.
    Cancelled,
    /// The request expired before it ran.
    Expired,
    /// The provider reported no result for the request.
    Missing,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::conversation::ConversationId;
    use crate::options::RunOptions;
    use crate::provider::{ModelString, ToolChoice};

    #[test]
    fn a_batch_handle_round_trips_through_json() {
        let handle = BatchHandle {
            version: 1,
            provider: ProviderKind::Anthropic,
            model: "anthropic/claude-opus-5-5".to_string(),
            base_url: None,
            parts: vec![BatchPart {
                batch_id: "msgbatch_1".to_string(),
                input_file_id: None,
                endpoint: None,
            }],
            request_ids: vec![RequestId("0".to_string())],
            created_at: "2026-10-20T00:00:00Z".to_string(),
            expires_at: None,
        };

        let json = serde_json::to_string(&handle).expect("serializes");
        let back: BatchHandle = serde_json::from_str(&json).expect("deserializes");
        assert_eq!(back.version, 1);
        assert_eq!(back.provider, ProviderKind::Anthropic);
        assert_eq!(back.parts.len(), 1);

        // A handle carries no credentials; the field set is exactly the design.
        let value = serde_json::to_value(&back).expect("serializes");
        let object = value.as_object().expect("an object");
        assert!(!object.contains_key("api_key"));
        assert!(!object.contains_key("base_url"));
    }

    #[test]
    fn a_batch_request_wraps_a_chat_request() {
        let request = BatchRequest {
            id: RequestId("0".to_string()),
            request: ChatRequest {
                model: ModelString::parse("openai/gpt-5.1").expect("parses"),
                messages: vec![Message::user("hi")],
                system: None,
                tools: Vec::new(),
                tool_choice: ToolChoice::Auto,
                options: RunOptions::default(),
                conversation_id: ConversationId::new(),
                provider_options: None,
            },
        };
        let json = serde_json::to_string(&request).expect("serializes");
        let back: BatchRequest = serde_json::from_str(&json).expect("deserializes");
        assert_eq!(back.id, RequestId("0".to_string()));
        assert_eq!(back.request.model.id, "gpt-5.1");
    }
}
