//! Per-run options: [`RunOptions`], [`Effort`], [`Thinking`].
//!
//! `RunOptions` is passed by value and overrides the agent's defaults for a
//! single call. `effort` is first-class and held constant per agent, because
//! changing it restarts the prompt cache. `temperature` and `top_p` are
//! optional and omitted unless set, since current Claude and OpenAI reasoning
//! models reject them.
//!
//! See `docs/design/core-api.md` §10.1 for the full design.

use std::collections::BTreeMap;
use std::time::Duration;

use serde::{Deserialize, Serialize};

use crate::error::{Error, ProviderError};

/// Options for one run, overriding the agent's defaults.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[non_exhaustive]
pub struct RunOptions {
    /// Reasoning effort. `None` uses the model's default. First-class, held
    /// constant per agent.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub effort: Option<Effort>,

    /// Thinking mode. Mapped per model. Default [`Thinking::Auto`].
    #[serde(default)]
    pub thinking: Thinking,

    /// How much thinking text comes back. Default [`ThinkingDisplay::Omitted`].
    #[serde(default)]
    pub thinking_display: ThinkingDisplay,

    /// Sampling temperature. Sent only when set; omitted by default.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,

    /// Nucleus sampling probability. Sent only when set; omitted by default.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,

    /// The maximum number of output tokens. `None` uses the model's default.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_output_tokens: Option<u32>,

    /// Sequences that stop generation. Empty by default.
    #[serde(default)]
    pub stop_sequences: Vec<String>,

    /// Tool-loop step limit. `None` uses the library default.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_steps: Option<usize>,

    /// Per-request timeout. Covers stream-start and the gap between events (the
    /// idle timeout), not the whole request. `None` uses the provider default.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub timeout: Option<Duration>,

    /// Prompt-cache TTL. 5 minutes is right for an agent loop; 1 hour for a
    /// human-paced chat. Default [`CacheTtl::FiveMinutes`].
    #[serde(default)]
    pub cache_ttl: CacheTtl,

    /// Extra body fields, passed through unchanged.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider_options: Option<serde_json::Value>,

    /// Extra HTTP headers, passed through unchanged.
    #[serde(default)]
    pub provider_headers: BTreeMap<String, String>,
}

impl RunOptions {
    /// Placeholder for the per-model thinking capability check.
    ///
    /// The real table (`Capabilities::thinking`) arrives with the provider
    /// traits; until then the caller passes whether the model can turn thinking
    /// off. On a model that cannot, [`Thinking::Off`] is rejected with
    /// [`Error::InvalidRequest`] before anything is sent.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidRequest`] when thinking is [`Thinking::Off`] and
    /// `can_disable_thinking` is `false`.
    pub fn check_thinking(
        &self,
        provider: &str,
        model: &str,
        can_disable_thinking: bool,
    ) -> Result<(), Error> {
        match self.thinking {
            Thinking::Off if !can_disable_thinking => Err(Error::InvalidRequest(ProviderError {
                request_id: None,
                status: None,
                code: Some("thinking_required".to_string()),
                message: format!("{provider}/{model} cannot turn thinking off"),
                retryable: false,
            })),
            _ => Ok(()),
        }
    }
}

/// Reasoning effort for models that expose one.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum Effort {
    /// Low reasoning effort.
    Low,
    /// Medium reasoning effort.
    Medium,
    /// High reasoning effort.
    High,
    /// Extra-high reasoning effort.
    XHigh,
    /// The model's maximum reasoning effort.
    Max,
}

/// Thinking mode, mapped per model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[non_exhaustive]
pub enum Thinking {
    /// Omit the field; the model's own default applies (adaptive on current
    /// Claude models).
    #[default]
    Auto,
    /// Ask for adaptive thinking explicitly.
    Adaptive,
    /// Turn thinking off, mapped per model. On a model that cannot turn it off
    /// this is [`Error::InvalidRequest`] before sending.
    Off,
}

/// How thinking text is returned.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[non_exhaustive]
pub enum ThinkingDisplay {
    /// The default on current Claude models; thinking text is not streamed.
    #[default]
    Omitted,
    /// Only a summary is returned.
    Summarized,
    /// Beta on Claude; progress notes between tool calls.
    Updates,
}

/// Prompt-cache time-to-live.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[non_exhaustive]
pub enum CacheTtl {
    /// A 5-minute write TTL. The default.
    #[default]
    FiveMinutes,
    /// A 1-hour write TTL (OpenAI/Anthropic).
    OneHour,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_omit_everything_optional() {
        let options = RunOptions::default();
        assert!(options.effort.is_none());
        assert!(options.temperature.is_none());
        assert!(options.top_p.is_none());
        assert!(options.max_output_tokens.is_none());
        assert!(options.max_steps.is_none());
        assert!(options.timeout.is_none());
        assert!(options.provider_options.is_none());
        assert!(options.stop_sequences.is_empty());
        assert!(options.provider_headers.is_empty());
        assert_eq!(options.thinking, Thinking::Auto);
        assert_eq!(options.thinking_display, ThinkingDisplay::Omitted);
        assert_eq!(options.cache_ttl, CacheTtl::FiveMinutes);
    }

    #[test]
    fn default_serialized_form_omits_temperature_and_top_p() {
        let json = serde_json::to_value(RunOptions::default()).expect("serializes");
        let object = json.as_object().expect("an object");
        assert!(!object.contains_key("temperature"));
        assert!(!object.contains_key("top_p"));
        assert!(!object.contains_key("effort"));
        assert!(!object.contains_key("max_output_tokens"));
        assert!(!object.contains_key("timeout"));
        assert_eq!(
            object.get("thinking").and_then(|v| v.as_str()),
            Some("Auto")
        );
    }

    #[test]
    fn temperature_and_top_p_are_present_when_set() {
        let options = RunOptions {
            temperature: Some(0.2),
            top_p: Some(0.9),
            ..RunOptions::default()
        };
        let json = serde_json::to_value(&options).expect("serializes");
        let temperature = json
            .get("temperature")
            .and_then(|v| v.as_f64())
            .expect("temperature is present");
        let top_p = json
            .get("top_p")
            .and_then(|v| v.as_f64())
            .expect("top_p is present");
        assert!((temperature - 0.2).abs() < 1e-6, "got {temperature}");
        assert!((top_p - 0.9).abs() < 1e-6, "got {top_p}");
    }

    #[test]
    fn run_options_round_trips_through_json() {
        let options = RunOptions {
            effort: Some(Effort::XHigh),
            thinking: Thinking::Adaptive,
            thinking_display: ThinkingDisplay::Updates,
            temperature: Some(0.5),
            top_p: Some(0.95),
            max_output_tokens: Some(2048),
            stop_sequences: vec!["STOP".to_string()],
            max_steps: Some(8),
            timeout: Some(Duration::from_secs(30)),
            cache_ttl: CacheTtl::OneHour,
            provider_options: Some(serde_json::json!({"top_k": 5})),
            provider_headers: BTreeMap::from([("x-test".to_string(), "1".to_string())]),
        };
        let json = serde_json::to_string(&options).expect("serializes");
        let back: RunOptions = serde_json::from_str(&json).expect("deserializes");
        assert_eq!(back.effort, Some(Effort::XHigh));
        assert_eq!(back.thinking, Thinking::Adaptive);
        assert_eq!(back.max_steps, Some(8));
        assert_eq!(back.cache_ttl, CacheTtl::OneHour);
        assert_eq!(back.provider_headers.get("x-test"), Some(&"1".to_string()));
    }

    #[test]
    fn thinking_off_is_rejected_on_models_that_cannot_disable_it() {
        let options = RunOptions {
            thinking: Thinking::Off,
            ..RunOptions::default()
        };
        let error = options
            .check_thinking("anthropic", "claude-opus-5.5", false)
            .expect_err("a model that cannot disable thinking rejects Off");
        assert!(matches!(error, Error::InvalidRequest(_)));

        assert!(options
            .check_thinking("anthropic", "claude-sonnet-5", true)
            .is_ok());
    }

    #[test]
    fn auto_thinking_is_always_accepted() {
        RunOptions::default()
            .check_thinking("anthropic", "claude-opus-5.5", false)
            .expect("Auto is always accepted");
    }
}
