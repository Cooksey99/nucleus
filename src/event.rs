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

use std::collections::BTreeMap;

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

/// Per-model prices in USD per 1M tokens. Apps override the built-in table.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[non_exhaustive]
pub struct PriceTable {
    entries: BTreeMap<String, ModelPrice>,
}

/// The price of one model, in USD per 1M tokens, with cache multipliers.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[non_exhaustive]
pub struct ModelPrice {
    /// Input (prompt) price in USD per 1M tokens.
    pub input_per_million: f64,
    /// Output (completion) price in USD per 1M tokens.
    pub output_per_million: f64,
    /// Multiplier applied to `input_per_million` for a cache read.
    pub cache_read_multiplier: f64,
    /// Multiplier for a 5-minute cache write.
    pub cache_write_multiplier: f64,
    /// Multiplier for a 1-hour cache write.
    pub cache_write_1h_multiplier: f64,
}

impl PriceTable {
    /// An empty table.
    ///
    /// Nothing is guessed: an unknown model has no price, so [`PriceTable::cost`]
    /// returns `None`. The Gloo catalog (`GET /platform/v2/models`) can seed it,
    /// and apps override entries with [`PriceTable::with`].
    pub fn builtin() -> Self {
        Self {
            entries: BTreeMap::new(),
        }
    }

    /// Return a copy with `model`'s price set, replacing any existing entry.
    pub fn with(mut self, model: &str, price: ModelPrice) -> Self {
        self.entries.insert(model.to_string(), price);
        self
    }

    /// The price for `model`, when the table knows it.
    pub fn get(&self, model: &str) -> Option<ModelPrice> {
        self.entries.get(model).copied()
    }

    /// Compute the cost of `usage` for `model` from the table's prices.
    ///
    /// `input_tokens` includes cache-read tokens (Responses semantics, §5.2);
    /// cache counts are clamped so `cached > input` never yields a negative.
    /// Returns `None` for an unknown model — never guess. The caller applies the
    /// batch discount and platform fee.
    pub fn cost(&self, model: &str, usage: &Usage) -> Option<Cost> {
        let price = self.get(model)?;

        let cache_read = usage.cache_read_tokens.unwrap_or(0).min(usage.input_tokens);
        let base_input = usage.input_tokens - cache_read;
        let cache_write = usage.cache_write_tokens.unwrap_or(0);
        let cache_write_1h = usage.cache_write_1h_tokens.unwrap_or(0);

        let input_units = base_input as f64
            + cache_read as f64 * price.cache_read_multiplier
            + cache_write as f64 * price.cache_write_multiplier
            + cache_write_1h as f64 * price.cache_write_1h_multiplier;
        let usd = (input_units * price.input_per_million
            + usage.output_tokens as f64 * price.output_per_million)
            / 1_000_000.0;

        Some(Cost {
            usd,
            source: CostSource::PriceTable,
        })
    }
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

    fn price() -> ModelPrice {
        ModelPrice {
            input_per_million: 3.0,
            output_per_million: 15.0,
            cache_read_multiplier: 0.1,
            cache_write_multiplier: 1.25,
            cache_write_1h_multiplier: 2.0,
        }
    }

    #[test]
    fn price_table_is_overridable() {
        let table = PriceTable::builtin().with("anthropic/claude-sonnet-5", price());
        assert!(table.get("anthropic/claude-sonnet-5").is_some());
        assert!(table.get("anthropic/unknown").is_none());
    }

    #[test]
    fn cost_uses_cache_read_and_write_tokens() {
        let table = PriceTable::builtin().with("m", price());
        let usage = Usage {
            input_tokens: 1_000_000,
            output_tokens: 100_000,
            cache_read_tokens: Some(200_000),
            cache_write_tokens: Some(100_000),
            cache_write_1h_tokens: Some(0),
            ..Usage::default()
        };

        let cost = table.cost("m", &usage).expect("a known model");
        assert_eq!(cost.source, CostSource::PriceTable);
        // 800k base + 200k*0.1 read + 100k*1.25 write = 945k units at $3/M
        // = $2.835; output 100k at $15/M = $1.5; total $4.335.
        assert!((cost.usd - 4.335).abs() < 1e-9, "got {}", cost.usd);
    }

    #[test]
    fn cost_clamps_cache_read_above_input() {
        let table = PriceTable::builtin().with("m", price());
        let usage = Usage {
            input_tokens: 100,
            cache_read_tokens: Some(500),
            ..Usage::default()
        };
        let cost = table.cost("m", &usage).expect("a known model");
        assert!(cost.usd >= 0.0);
        // All 100 input tokens are cache reads: 100 * 0.1 = 10 units at $3/M.
        assert!((cost.usd - 0.000_03).abs() < 1e-12, "got {}", cost.usd);
    }

    #[test]
    fn cost_is_none_for_an_unknown_model() {
        assert!(PriceTable::builtin().cost("m", &Usage::default()).is_none());
    }

    #[test]
    fn usage_round_trips_with_cost() {
        let usage = Usage {
            input_tokens: 10,
            output_tokens: 20,
            cost: Some(Cost {
                usd: 1.5,
                source: CostSource::Provider,
            }),
            batch: true,
            ..Usage::default()
        };
        let json = serde_json::to_string(&usage).expect("serializes");
        let back: Usage = serde_json::from_str(&json).expect("deserializes");
        assert_eq!(back.input_tokens, 10);
        assert!(back.batch);
        let cost = back.cost.expect("cost survives");
        assert_eq!(cost.source, CostSource::Provider);
    }
}
