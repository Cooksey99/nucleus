//! The single public error type, `nucleus::Error`.
//!
//! Every fallible public method returns [`Result`], whose error is [`Error`].
//! The type is `#[non_exhaustive]`, serde-serializable and carries structured
//! payloads ([`ProviderError`], [`Retryable`], [`Refusal`], [`Truncation`]) so it
//! survives an HTTP boundary. `std::io::Error` and provider protocol types never
//! appear here; the message text carries the context instead.
//!
//! See `docs/design/core-api.md` §6 for the full design.

use serde::{Deserialize, Serialize};

/// The single public error type. `#[non_exhaustive]`.
#[derive(Debug, Clone, Serialize, Deserialize, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    /// The model declined (HTTP 200 refusal, or a guarded content block).
    /// The partial turn is not committed.
    #[error("refused")]
    Refused(Refusal),

    /// Content policy, kept separate from auth.
    #[error("content policy")]
    ContentPolicy(Refusal),

    /// Retryable rate limit.
    #[error("rate limited")]
    RateLimited(Retryable),

    /// The provider is overloaded. Retryable.
    #[error("provider overloaded")]
    Overloaded(Retryable),

    /// Timed out (HTTP 504, or the request/idle timeout). Retryable.
    #[error("timed out")]
    Timeout(Retryable),

    /// Output was cut off (`max_tokens` or `model_context_window_exceeded`).
    #[error("truncated")]
    Truncated(Truncation),

    /// Authentication or permission failure. Not retryable.
    #[error("authentication failed")]
    Auth(ProviderError),

    /// A non-retryable billing or spend-cap error. Not retryable.
    #[error("billing error")]
    Billing(ProviderError),

    /// A malformed request the provider rejected (400, 404, 413, 422).
    #[error("invalid request")]
    InvalidRequest(ProviderError),

    /// Any other provider-side failure. `provider.retryable` says whether the
    /// reliability layer may retry.
    #[error("provider error")]
    Provider(ProviderError),

    /// A tool failed.
    #[error("tool '{name}' failed")]
    Tool {
        /// The tool's model-facing name.
        name: String,
        /// The underlying failure.
        #[source]
        source: Box<Error>,
    },

    /// The provider or model cannot do what was asked.
    #[error("unsupported: {feature}")]
    Unsupported {
        /// The capability that is missing, e.g. `"batch"`, `"embeddings"`.
        feature: String,
        /// The provider's prefix.
        provider: String,
        /// The model, when the limitation is per-model.
        model: Option<String>,
    },

    /// Batch results are not available yet.
    #[error("batch results not ready")]
    NotReady,

    /// A local or concurrent batch cannot be resumed.
    #[error("batch not resumable")]
    BatchNotResumable,

    /// A conversation-store failure.
    #[error("conversation store error")]
    Store(String),

    /// Serialization or schema-conversion failure.
    #[error("serialization error")]
    Serialization(String),

    /// I/O against a file, socket or child process.
    #[error("io error")]
    Io(String),
}

/// A provider-originated error. Every provider error carries the request id,
/// so a failed call can be quoted to the provider's support.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct ProviderError {
    /// The provider's request id. `None` only when the provider sent none.
    pub request_id: Option<RequestId>,
    /// HTTP status, when the error came from a response.
    pub status: Option<u16>,
    /// The provider's error code, e.g. `insufficient_quota`.
    pub code: Option<String>,
    /// The human-readable message from the provider.
    pub message: String,
    /// Whether the reliability layer may retry.
    pub retryable: bool,
}

/// A provider request id. A newtype so it is never confused with other ids.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[non_exhaustive]
pub struct RequestId(pub String);

/// A retryable provider failure, with the delay when the provider gave one.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct Retryable {
    /// The provider error.
    pub error: ProviderError,
    /// Seconds to wait before retrying, when the provider said.
    pub retry_after: Option<f64>,
}

/// A refusal, with the category and explanation when the provider gave them.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct Refusal {
    /// The refusal category, when the provider named one.
    pub category: Option<String>,
    /// The provider's explanation.
    pub explanation: Option<String>,
    /// The provider error this refusal came with.
    pub error: ProviderError,
}

/// A truncated output, with the text produced before truncation.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct Truncation {
    /// Why the output stopped.
    pub reason: crate::event::StopReason,
    /// Text produced before truncation, when the caller asked for it.
    pub partial: String,
    /// The provider error this truncation came with.
    pub error: ProviderError,
}

/// The result type used across the public API.
pub type Result<T, E = Error> = std::result::Result<T, E>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn provider_error_round_trips_through_json() {
        let error = Error::RateLimited(Retryable {
            error: ProviderError {
                request_id: Some(RequestId("req_1".to_string())),
                status: Some(429),
                code: Some("rate_limit_exceeded".to_string()),
                message: "slow down".to_string(),
                retryable: true,
            },
            retry_after: Some(1.5),
        });

        let json = serde_json::to_string(&error).expect("serializes");
        let back: Error = serde_json::from_str(&json).expect("deserializes");
        assert!(matches!(back, Error::RateLimited(_)));
    }
}
