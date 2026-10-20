//! Streaming events, token usage and cost.
//!
//! Streaming is `Stream<Item = Event>`: owned events, no borrowed callbacks.
//! Errors travel as [`Event::Error`], so the stream item is plain [`Event`]
//! rather than a `Result`. Tool calls may arrive whole or as argument deltas, so
//! a complete [`Event::ToolCall`] event is always emitted and deltas are
//! optional. The library never prints; visible text reaches the caller through
//! [`Event::TextDelta`].
//!
//! See `docs/design/core-api.md` §5 for the full design.

use serde::{Deserialize, Serialize};

use crate::error::Error;
use crate::message::ToolCallId;

/// A live-streamed item. Errors travel as [`Event::Error`], so the stream item
/// is plain `Event` rather than `Result`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub enum Event {
    /// Visible text. For live display only; the authoritative text is rebuilt
    /// from parts when the turn ends.
    TextDelta {
        /// The text fragment.
        text: String,
    },
    /// Reasoning/thinking text or summary.
    ReasoningDelta {
        /// The reasoning fragment.
        text: String,
        /// Whether this is a summary rather than the full reasoning.
        summary: bool,
    },
    /// A tool-call argument fragment. Hosted providers only; may never appear.
    ToolCallDelta {
        /// The id of the tool call being streamed.
        id: ToolCallId,
        /// The argument fragment.
        arguments_delta: String,
    },
    /// A complete tool call, once its arguments are whole. Always emitted,
    /// whichever way the provider delivered it.
    ToolCall {
        /// The provider's tool-call id.
        id: ToolCallId,
        /// The tool's model-facing name.
        name: String,
        /// The raw argument JSON, as the model produced it.
        input: Box<serde_json::value::RawValue>,
    },
    /// A provider-specific event worth surfacing; unknown events pass through
    /// here and are logged with `tracing`.
    Provider {
        /// The endpoint dialect that produced the event.
        dialect: String,
        /// A short kind tag for diagnostics.
        kind: String,
        /// The raw event payload.
        raw: serde_json::Value,
    },
    /// Token usage. Present once, near the end, when the provider reports it.
    Usage(Usage),
    /// The model stopped, with the reason. Terminal for the turn.
    Stop(StopReason),
    /// The provider errored. Terminal for the stream.
    Error(Error),
}

/// The owned event stream. No lifetime parameter.
#[cfg(any(feature = "hosted", feature = "lancedb"))]
pub type EventStream = std::pin::Pin<Box<dyn futures::Stream<Item = Event> + Send>>;

/// Why the model stopped.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub enum StopReason {
    /// The model finished its turn.
    EndTurn,
    /// The model asked for one or more tools.
    ToolUse,
    /// The output hit the token limit. Never execute a trailing partial tool
    /// call.
    MaxTokens,
    /// A stop sequence ended the output.
    StopSequence {
        /// The sequence that matched.
        sequence: String,
    },
    /// The output filled the context window; a hint to compact.
    ContextWindowExceeded,
    /// The model declined. The partial turn is not committed.
    Refusal {
        /// The refusal category, when the provider named one.
        category: Option<String>,
        /// The provider's explanation.
        explanation: Option<String>,
    },
    /// A server-side tool loop hit its iteration limit; resending continues.
    PauseTurn,
    /// Unrecognised reason, kept verbatim.
    Other(String),
}

/// Token usage and cost for one turn.
///
/// Field semantics follow the Responses protocol: `output_tokens` includes
/// reasoning tokens and `input_tokens` includes cache-read tokens; cached counts
/// are clamped so a provider that reports `cached > input` never yields a
/// negative.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[non_exhaustive]
pub struct Usage {
    /// Input (prompt) tokens, including cache reads.
    pub input_tokens: u64,
    /// Output (completion) tokens, including reasoning.
    pub output_tokens: u64,
    /// Cache-read tokens.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_read_tokens: Option<u64>,
    /// 5-minute-TTL cache writes.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_write_tokens: Option<u64>,
    /// 1-hour-TTL cache writes, kept separate because they cost more.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_write_1h_tokens: Option<u64>,
    /// Reasoning tokens, when reported separately.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_tokens: Option<u64>,
    /// The total cost, when it could be determined.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cost: Option<Cost>,
    /// Whether this usage came from a batch request (affects the discount).
    #[serde(default)]
    pub batch: bool,
}

/// The cost of one turn and where the figure came from.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct Cost {
    /// The cost in US dollars.
    pub usd: f64,
    /// How the cost was determined.
    pub source: CostSource,
}

/// Where a [`Cost`] figure came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum CostSource {
    /// The provider reported the cost (xAI `cost_in_usd_ticks`).
    Provider,
    /// Computed from the built-in or user-supplied price table.
    PriceTable,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn usage_is_default_zero() {
        let usage = Usage::default();
        assert_eq!(usage.input_tokens, 0);
        assert!(usage.cost.is_none());
    }

    #[test]
    fn event_serializes() {
        let event = Event::TextDelta {
            text: "hi".to_string(),
        };
        let json = serde_json::to_string(&event).expect("serializes");
        assert!(json.contains("TextDelta"));
    }
}
