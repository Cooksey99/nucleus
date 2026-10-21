//! The model-facing [`Tool`] trait, [`tool_fn`] and the run-scoped
//! [`ToolContext`].
//!
//! A tool is the extension point the model calls. Its schema is the **whole**
//! JSON Schema for the arguments (not just `properties`), and `call` takes the
//! raw JSON the model produced. [`tool_fn`] adapts a function with typed
//! arguments, deriving the schema and deserializing the raw JSON before the call.
//!
//! See `docs/design/core-api.md` §7.1 for the full design.

use std::future::Future;
use std::marker::PhantomData;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use serde_json::value::RawValue;

use crate::conversation::ConversationId;
use crate::error::{Error, Result};
use crate::message::{ToolCallId, ToolOutput};

/// The extension point the model calls: a name, a description, a JSON Schema
/// and an async `call`.
#[async_trait::async_trait]
pub trait Tool: Send + Sync {
    /// The model-facing name. `[A-Za-z0-9_-]`, at most 64 characters.
    fn name(&self) -> &str;

    /// What the tool does, shown to the model.
    fn description(&self) -> &str;

    /// The full JSON Schema for [`Tool::call`]'s arguments.
    fn parameters(&self) -> serde_json::Value;

    /// Untrusted hints (read-only, destructive). Reaches the approval hook as
    /// context and is never used to skip approval.
    fn annotations(&self) -> ToolAnnotations {
        ToolAnnotations::default()
    }

    /// Run the tool. `arguments` is the raw JSON the model produced; `context`
    /// carries cancellation and progress.
    ///
    /// # Errors
    ///
    /// Returns [`Error`] when the arguments cannot be parsed or the tool fails.
    async fn call(&self, arguments: Box<RawValue>, context: ToolContext) -> Result<ToolOutput>;
}

/// Untrusted hints about a tool's behaviour.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[non_exhaustive]
pub struct ToolAnnotations {
    /// A human-readable title for the tool.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    /// The tool does not modify anything.
    #[serde(default)]
    pub read_only_hint: bool,
    /// The tool may make irreversible changes.
    #[serde(default)]
    pub destructive_hint: bool,
    /// Repeating the call has no additional effect.
    #[serde(default)]
    pub idempotent_hint: bool,
    /// The tool interacts with external systems.
    #[serde(default)]
    pub open_world_hint: bool,
}

/// Run-scoped context passed to [`Tool::call`]. Owned; no borrows.
pub struct ToolContext {
    call_id: ToolCallId,
    conversation_id: ConversationId,
    cancelled: Arc<AtomicBool>,
}

impl ToolContext {
    /// A context for `call_id` inside `conversation_id`, not cancelled.
    // Constructed by the agent loop, which lands with the core agent task.
    #[allow(dead_code)]
    pub(crate) fn new(call_id: ToolCallId, conversation_id: ConversationId) -> Self {
        Self {
            call_id,
            conversation_id,
            cancelled: Arc::new(AtomicBool::new(false)),
        }
    }

    /// The id of the tool call being run.
    pub fn call_id(&self) -> &ToolCallId {
        &self.call_id
    }

    /// The conversation the run belongs to.
    pub fn conversation_id(&self) -> &ConversationId {
        &self.conversation_id
    }

    /// Whether the caller has cancelled the run.
    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::Relaxed)
    }
}

/// Wrap a plain async function as a [`Tool`].
///
/// The argument type derives [`schemars::JsonSchema`], whose schema becomes the
/// tool's parameters, and the raw JSON is deserialized into it before the call.
///
/// ```no_run
/// # use nucleus::{tool_fn, ToolOutput};
/// #[derive(serde::Deserialize, schemars::JsonSchema)]
/// struct Weather { city: String }
///
/// let weather = tool_fn("get_weather", "Look up the weather", |a: Weather, _ctx| async move {
///     Ok(ToolOutput::text(format!("Sunny in {}", a.city)))
/// });
/// ```
pub fn tool_fn<A, F, Fut>(name: &str, description: &str, f: F) -> Arc<dyn Tool>
where
    A: DeserializeOwned + schemars::JsonSchema + Send + 'static,
    F: Fn(A, ToolContext) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = Result<ToolOutput>> + Send,
{
    Arc::new(AsyncFnTool {
        name: name.to_string(),
        description: description.to_string(),
        f,
        _marker: PhantomData,
    })
}

/// Sync convenience wrapper, for tools that do no async work.
pub fn tool_fn_sync<A, F>(name: &str, description: &str, f: F) -> Arc<dyn Tool>
where
    A: DeserializeOwned + schemars::JsonSchema + Send + 'static,
    F: Fn(A) -> Result<ToolOutput> + Send + Sync + 'static,
{
    Arc::new(SyncFnTool {
        name: name.to_string(),
        description: description.to_string(),
        f,
        _marker: PhantomData,
    })
}

/// The full JSON Schema for `A` as a JSON value.
fn schema_of<A: schemars::JsonSchema>() -> serde_json::Value {
    serde_json::to_value(schemars::schema_for!(A)).unwrap_or(serde_json::Value::Null)
}

/// Deserialize the raw arguments into the typed handler's argument.
fn parse_arguments<A: DeserializeOwned>(arguments: &RawValue) -> Result<A> {
    serde_json::from_str(arguments.get()).map_err(|e| Error::Serialization(e.to_string()))
}

struct AsyncFnTool<A, F> {
    name: String,
    description: String,
    f: F,
    _marker: PhantomData<fn() -> A>,
}

#[async_trait::async_trait]
impl<A, F, Fut> Tool for AsyncFnTool<A, F>
where
    A: DeserializeOwned + schemars::JsonSchema + Send + 'static,
    F: Fn(A, ToolContext) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = Result<ToolOutput>> + Send,
{
    fn name(&self) -> &str {
        &self.name
    }

    fn description(&self) -> &str {
        &self.description
    }

    fn parameters(&self) -> serde_json::Value {
        schema_of::<A>()
    }

    async fn call(&self, arguments: Box<RawValue>, context: ToolContext) -> Result<ToolOutput> {
        let arguments: A = parse_arguments(&arguments)?;
        (self.f)(arguments, context).await
    }
}

struct SyncFnTool<A, F> {
    name: String,
    description: String,
    f: F,
    _marker: PhantomData<fn() -> A>,
}

#[async_trait::async_trait]
impl<A, F> Tool for SyncFnTool<A, F>
where
    A: DeserializeOwned + schemars::JsonSchema + Send + 'static,
    F: Fn(A) -> Result<ToolOutput> + Send + Sync + 'static,
{
    fn name(&self) -> &str {
        &self.name
    }

    fn description(&self) -> &str {
        &self.description
    }

    fn parameters(&self) -> serde_json::Value {
        schema_of::<A>()
    }

    async fn call(&self, arguments: Box<RawValue>, _context: ToolContext) -> Result<ToolOutput> {
        let arguments: A = parse_arguments(&arguments)?;
        (self.f)(arguments)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Deserialize, schemars::JsonSchema)]
    struct Weather {
        city: String,
    }

    fn context() -> ToolContext {
        ToolContext::new(ToolCallId("call_1".to_string()), ConversationId::new())
    }

    fn raw(json: &str) -> Box<RawValue> {
        RawValue::from_string(json.to_string()).expect("valid raw json")
    }

    #[tokio::test]
    async fn tool_fn_round_trips_typed_arguments() {
        let tool = tool_fn(
            "get_weather",
            "Look up the weather",
            |a: Weather, _ctx| async move {
                Ok(ToolOutput {
                    content: format!("Sunny in {}", a.city),
                    structured: None,
                })
            },
        );

        assert_eq!(tool.name(), "get_weather");
        let output = tool
            .call(raw(r#"{"city":"Paris"}"#), context())
            .await
            .expect("the tool runs");
        assert_eq!(output.content, "Sunny in Paris");
    }

    #[test]
    fn tool_fn_generates_the_full_json_schema() {
        let tool = tool_fn_sync("get_weather", "Look up the weather", |a: Weather| {
            Ok(ToolOutput {
                content: a.city,
                structured: None,
            })
        });

        let schema = tool.parameters();
        assert_eq!(
            schema.get("type").and_then(|v| v.as_str()),
            Some("object"),
            "the whole schema, not just `properties`"
        );
        assert!(schema
            .get("properties")
            .and_then(|p| p.get("city"))
            .is_some());
        let required = schema
            .get("required")
            .and_then(|v| v.as_array())
            .expect("required is present");
        assert!(required.iter().any(|v| v.as_str() == Some("city")));
    }

    #[tokio::test]
    async fn mistyped_arguments_fail_before_the_handler() {
        let tool = tool_fn_sync("get_weather", "Look up the weather", |a: Weather| {
            Ok(ToolOutput {
                content: a.city,
                structured: None,
            })
        });

        let error = tool
            .call(raw(r#"{"city": 7}"#), context())
            .await
            .expect_err("a number is not a city");
        assert!(matches!(error, Error::Serialization(_)));
    }
}
