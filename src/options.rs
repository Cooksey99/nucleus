//! Per-run and per-agent options: [`RunOptions`], [`Effort`], [`Thinking`].
//!
//! `RunOptions` is passed by value and overrides the agent's defaults for a
//! single call. `None` means "use the agent default" for every field.
//!
//! See `docs/design/core-api.md` §10.1 for the full design.

use serde::{Deserialize, Serialize};

/// Options for one run, overriding the agent's defaults. `None` fields mean
/// "use the agent's value".
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[non_exhaustive]
pub struct RunOptions {
    /// Override the agent's system prompt for this call only.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub system: Option<String>,

    /// Hosted only: ask for a structured output of this JSON Schema.
    /// The tools parameter is therefore seed-independent.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub format: Option<serde_json::Value>,

    /// Hosted only: pass through to native `strict` structured outputs.
    /// Default `false`.
    #[serde(default)]
    pub strict: bool,

    /// Override the model for this call only.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,

    /// Reasoning effort, where the provider supports it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub effort: Option<Effort>,

    /// Thinking configuration. `thinking.is_none()` leaves it to the provider.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thinking: Option<Thinking>,

    /// Per-request timeout in milliseconds. `None` uses the provider default.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub timeout_ms: Option<u64>,

    /// Hosted only: disable the reliability layer for this call. Used by tests
    /// and by callers that retry themselves.
    #[serde(default)]
    pub no_retry: bool,

    /// Hosted only: prompt-cache TTL. `CacheTtl::None` disables caching.
    #[serde(default)]
    pub cache_ttl: CacheTtl,

    /// Price per million tokens, when the caller knows it better than the
    /// built-in table.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub price: Option<serde_json::Value>,
}

/// Reasoning effort for models that expose one.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum Effort {
    /// The lowest reasoning effort the model offers.
    Minimal,
    /// Low reasoning effort.
    Low,
    /// Medium reasoning effort.
    Medium,
    /// High reasoning effort.
    High,
}

/// Thinking configuration. Fields are ignored on models without thinking.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[non_exhaustive]
pub struct Thinking {
    /// Whether thinking is enabled. Default `false`.
    pub enabled: bool,
    /// How thinking text is returned. Default [`ThinkingDisplay::Summarized`].
    pub display: ThinkingDisplay,
}

/// How thinking text is returned.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[non_exhaustive]
pub enum ThinkingDisplay {
    /// Full reasoning is returned.
    Full,
    /// Only a summary is returned.
    #[default]
    Summarized,
    /// Reasoning is omitted.
    Omitted,
}

/// Prompt-cache time-to-live.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[non_exhaustive]
pub enum CacheTtl {
    /// Do not cache.
    #[default]
    None,
    /// The provider's default TTL (OpenAI/Anthropic).
    Default,
    /// A 1-hour write TTL (OpenAI/Anthropic).
    OneHour,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_are_none() {
        let options = RunOptions::default();
        assert!(options.system.is_none());
        assert_eq!(options.cache_ttl, CacheTtl::None);
        assert!(!options.strict);
    }

    #[test]
    fn thinking_defaults_to_summarized() {
        let thinking = Thinking::default();
        assert!(!thinking.enabled);
        assert_eq!(thinking.display, ThinkingDisplay::Summarized);
    }
}
