//! The agent: settings, plus `run` when a provider can be compiled.
//!
//! `Agent` is a concrete type, not a trait. It holds a provider and [`Tool`]s
//! and calls them. It is not serde: a serializable description is `AgentSpec`,
//! which is not built yet.
//!
//! The system prompt lives on the agent and is sent on each request. It is
//! never written into the conversation.
//!
//! See `docs/design/core-api.md` §8 for the full design.

use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::conversation::Conversation;
use crate::error::{Error, ProviderError, Result};
use crate::event::{StopReason, Usage};
use crate::message::{Message, ToolCallId, ToolOutput};
use crate::options::RunOptions;
use crate::provider::{ChatRequest, ModelString};
use crate::store::ConversationStore;
use crate::tool::{Tool, ToolAnnotations};

#[cfg(any(feature = "hosted", feature = "lancedb"))]
use crate::conversation::ConversationId;
#[cfg(any(feature = "hosted", feature = "lancedb"))]
use crate::error::Refusal;
#[cfg(any(feature = "hosted", feature = "lancedb"))]
use crate::event::{Cost, Event};
#[cfg(any(feature = "hosted", feature = "lancedb"))]
use crate::message::{Part, Role};
#[cfg(any(feature = "hosted", feature = "lancedb"))]
use crate::provider::{Provider, ThinkingSupport, ToolChoice, ToolSpec};
#[cfg(any(feature = "hosted", feature = "lancedb"))]
use futures::StreamExt;

/// How many model calls [`Agent::run`] makes when [`RunOptions::max_steps`] is
/// unset.
const DEFAULT_MAX_STEPS: usize = 8;

type ApprovalHook = Arc<dyn Fn(&ToolCall) -> Approval + Send + Sync>;
type RequestHook = Arc<dyn Fn(&ChatRequest) + Send + Sync>;
type ResponseHook = Arc<dyn Fn(&Turn) + Send + Sync>;
type ToolResultHook = Arc<dyn Fn(&ToolCallId, &Result<ToolOutput, Error>) + Send + Sync>;

/// An agent: model, system prompt, tools, and the provider that serves them.
///
/// Cheap to clone. A clone shares the provider, tools, and hooks.
#[derive(Clone)]
pub struct Agent {
    /// Present when `hosted` or `lancedb` is enabled. Those features are what
    /// compile [`Provider`].
    #[cfg(any(feature = "hosted", feature = "lancedb"))]
    provider: Arc<dyn Provider>,
    model: ModelString,
    system: Option<String>,
    tools: Arc<[Arc<dyn Tool>]>,
    options: RunOptions,
    store: Option<Arc<dyn ConversationStore>>,
    on_tool_call: Option<ApprovalHook>,
    on_request: Option<RequestHook>,
    on_response: Option<ResponseHook>,
    on_tool_result: Option<ToolResultHook>,
}

impl std::fmt::Debug for Agent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Agent")
            .field("model", &self.model)
            .field("system", &self.system)
            .field("options", &self.options)
            .finish_non_exhaustive()
    }
}

impl Agent {
    /// Start a builder.
    pub fn builder() -> AgentBuilder {
        AgentBuilder::new()
    }

    /// The model this agent calls.
    pub fn model(&self) -> &ModelString {
        &self.model
    }

    /// A clone that calls `model` instead. The conversation is not touched.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Serialization`] when `model` is not a `provider/id` string.
    pub fn with_model(&self, model: impl Into<String>) -> Result<Self> {
        let model = ModelString::parse(&model.into())?;
        let mut next = self.clone();
        next.model = model;
        Ok(next)
    }

    /// A clone with a different system prompt.
    ///
    /// This does not edit history. On a provider that binds thinking to the
    /// prompt, the next request restarts the prompt cache.
    pub fn with_system(&self, system: impl Into<String>) -> Self {
        let mut next = self.clone();
        next.system = Some(system.into());
        next
    }

    /// A clone that uses `tools` instead of the current set.
    ///
    /// This does not edit history. Changing tools restarts the prompt cache on
    /// providers that bind thinking to the tool list.
    pub fn with_tools(&self, tools: Vec<Arc<dyn Tool>>) -> Self {
        let mut next = self.clone();
        next.tools = Arc::from(tools);
        next
    }

    /// The options sent on each request.
    pub fn options(&self) -> &RunOptions {
        &self.options
    }
}

#[cfg(any(feature = "hosted", feature = "lancedb"))]
impl Agent {
    /// Run one turn: append `input`, call the provider, and run any tool calls.
    ///
    /// Independent tool calls in a step run together. A denial, an unknown
    /// tool, or a tool error is a tool result with `is_error: true`, not an
    /// error from this method, and the loop continues. A provider failure or a
    /// refusal rolls the conversation back to where this call started.
    ///
    /// When [`RunOptions::max_steps`] is unset, the loop stops after eight
    /// model calls. Tool calls still pending at that point get error results,
    /// so the next request is valid.
    ///
    /// The system prompt is sent on the request and is not stored in
    /// `conversation`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidRequest`] when the model cannot turn thinking
    /// off. Returns the provider error when the stream fails, and
    /// [`Error::Refused`] when the model declines. A store error is returned
    /// after the turn has already been appended.
    pub async fn run(
        &self,
        conversation: &mut Conversation,
        input: impl Into<Message>,
    ) -> Result<Turn> {
        self.check_thinking()?;
        let checkpoint = conversation.messages.len();
        conversation.push(input.into());

        let max_steps = self.options.max_steps.unwrap_or(DEFAULT_MAX_STEPS).max(1);
        let mut usage = Usage::default();
        let mut steps = 0usize;

        let (message, stop_reason) = loop {
            steps += 1;
            let step = match self.step(conversation).await {
                Ok(step) => step,
                Err(err) => {
                    conversation.messages.truncate(checkpoint);
                    return Err(err);
                }
            };
            add_usage(&mut usage, &step.usage);

            if let StopReason::Refusal {
                category,
                explanation,
            } = step.stop.clone()
            {
                conversation.messages.truncate(checkpoint);
                return Err(refused(category, explanation));
            }

            let calls = pending_calls(&step.message);
            let stop = step.stop;
            let message = step.message.clone();
            conversation.push(step.message);

            if !should_run_tools(&stop, &calls) {
                break (message, stop);
            }
            if steps >= max_steps {
                self.skip_calls(
                    conversation,
                    &calls,
                    "tool call was not run: step limit reached",
                );
                break (message, StopReason::Other("max_steps".to_string()));
            }

            let id = conversation.id.clone();
            let results = self.execute_tools(&calls, &id).await;
            conversation.push(Message::parts(Role::User, results));
        };

        let turn = Turn {
            message,
            usage,
            stop_reason,
            knowledge_queries: Vec::new(),
        };
        if let Some(hook) = &self.on_response {
            hook(&turn);
        }
        if let Some(store) = &self.store {
            store.save(conversation).await?;
        }
        Ok(turn)
    }

    fn check_thinking(&self) -> Result<()> {
        let caps = self.provider.capabilities(&self.model.id);
        let can_disable = !matches!(caps.thinking, ThinkingSupport::AlwaysOn);
        self.options
            .check_thinking(self.provider.kind().prefix(), &self.model.id, can_disable)
    }

    async fn step(&self, conversation: &Conversation) -> Result<ModelStep> {
        let request = self.chat_request(conversation);
        if let Some(hook) = &self.on_request {
            hook(&request);
        }
        let stream = self.provider.chat_stream(request).await?;
        drain(stream).await
    }

    fn chat_request(&self, conversation: &Conversation) -> ChatRequest {
        let tools: Vec<ToolSpec> = self
            .tools
            .iter()
            .map(|tool| ToolSpec {
                name: tool.name().to_string(),
                description: tool.description().to_string(),
                parameters: tool.parameters(),
                // Strict mode is a provider option. Left off until a caller can ask for it.
                strict: false,
            })
            .collect();
        let tool_choice = if tools.is_empty() {
            ToolChoice::None
        } else {
            ToolChoice::Auto
        };
        ChatRequest {
            model: self.model.clone(),
            messages: conversation.messages.clone(),
            system: self.system.clone(),
            tools,
            tool_choice,
            options: self.options.clone(),
            conversation_id: conversation.id.clone(),
            provider_options: self.options.provider_options.clone(),
        }
    }

    async fn execute_tools(
        &self,
        calls: &[PendingCall],
        conversation_id: &ConversationId,
    ) -> Vec<Part> {
        let futures = calls
            .iter()
            .map(|call| self.execute_one(call, conversation_id));
        futures::future::join_all(futures).await
    }

    async fn execute_one(&self, call: &PendingCall, conversation_id: &ConversationId) -> Part {
        let tool = self.tools.iter().find(|tool| tool.name() == call.name);
        let view = ToolCall {
            id: call.id.clone(),
            name: call.name.clone(),
            input: call.input.clone(),
            annotations: tool.map(|tool| tool.annotations()),
        };
        if let Some(hook) = &self.on_tool_call {
            if hook(&view) == Approval::Deny {
                let outcome = Err(invalid("tool_denied", "tool call denied"));
                self.observe_tool(&call.id, &outcome);
                return error_part(&call.id, "tool call denied");
            }
        }
        let Some(tool) = tool else {
            let outcome = Err(invalid(
                "unknown_tool",
                &format!("unknown tool '{}'", call.name),
            ));
            self.observe_tool(&call.id, &outcome);
            return error_part(&call.id, &format!("unknown tool '{}'", call.name));
        };

        let context = crate::tool::ToolContext::new(call.id.clone(), conversation_id.clone());
        match tool.call(call.input.clone(), context).await {
            Ok(output) => {
                self.observe_tool(&call.id, &Ok(output.clone()));
                Part::ToolResult {
                    call_id: call.id.clone(),
                    output,
                    is_error: false,
                }
            }
            Err(err) => {
                let text = err.to_string();
                self.observe_tool(&call.id, &Err(err));
                error_part(&call.id, &text)
            }
        }
    }

    fn skip_calls(&self, conversation: &mut Conversation, calls: &[PendingCall], reason: &str) {
        let mut parts = Vec::with_capacity(calls.len());
        for call in calls {
            let outcome = Err(invalid("step_limit", reason));
            self.observe_tool(&call.id, &outcome);
            parts.push(error_part(&call.id, reason));
        }
        conversation.push(Message::parts(Role::User, parts));
    }

    fn observe_tool(&self, id: &ToolCallId, outcome: &Result<ToolOutput, Error>) {
        if let Some(hook) = &self.on_tool_result {
            hook(id, outcome);
        }
    }
}

/// Builds an [`Agent`]. Every setter consumes `self` and returns it.
#[derive(Default)]
pub struct AgentBuilder {
    model: Option<String>,
    #[cfg(any(feature = "hosted", feature = "lancedb"))]
    provider: Option<Arc<dyn Provider>>,
    system: Option<String>,
    tools: Vec<Arc<dyn Tool>>,
    options: RunOptions,
    store: Option<Arc<dyn ConversationStore>>,
    on_tool_call: Option<ApprovalHook>,
    on_request: Option<RequestHook>,
    on_response: Option<ResponseHook>,
    on_tool_result: Option<ToolResultHook>,
}

impl AgentBuilder {
    /// An empty builder.
    pub fn new() -> Self {
        Self::default()
    }

    /// The `provider/id` model string. Parsed in [`AgentBuilder::build`].
    ///
    /// `local//absolute/path.gguf` is a file on disk. Build loads it with the
    /// local provider when none was set.
    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }

    /// The provider that serves [`AgentBuilder::model`].
    ///
    /// Required when `hosted` or `lancedb` is enabled, because that is when
    /// [`Provider`] exists. Without one of those features there is no provider
    /// to install and [`Agent::run`] is not compiled.
    #[cfg(any(feature = "hosted", feature = "lancedb"))]
    pub fn provider(mut self, provider: Arc<dyn Provider>) -> Self {
        self.provider = Some(provider);
        self
    }

    /// The system prompt. Sent on each request, never stored in history.
    pub fn system(mut self, system: impl Into<String>) -> Self {
        self.system = Some(system.into());
        self
    }

    /// Add one tool.
    pub fn tool(mut self, tool: Arc<dyn Tool>) -> Self {
        self.tools.push(tool);
        self
    }

    /// Add tools.
    pub fn tools(mut self, tools: impl IntoIterator<Item = Arc<dyn Tool>>) -> Self {
        self.tools.extend(tools);
        self
    }

    /// Replace the run options.
    pub fn options(mut self, options: RunOptions) -> Self {
        self.options = options;
        self
    }

    /// Save each successful turn here.
    pub fn conversation_store(mut self, store: Arc<dyn ConversationStore>) -> Self {
        self.store = Some(store);
        self
    }

    /// Approval gate. A [`Approval::Deny`] becomes an error tool result.
    pub fn on_tool_call(
        mut self,
        hook: impl Fn(&ToolCall) -> Approval + Send + Sync + 'static,
    ) -> Self {
        self.on_tool_call = Some(Arc::new(hook));
        self
    }

    /// Called with each provider request, before it is sent.
    pub fn on_request(mut self, hook: impl Fn(&ChatRequest) + Send + Sync + 'static) -> Self {
        self.on_request = Some(Arc::new(hook));
        self
    }

    /// Called with the committed turn, before it is saved.
    pub fn on_response(mut self, hook: impl Fn(&Turn) + Send + Sync + 'static) -> Self {
        self.on_response = Some(Arc::new(hook));
        self
    }

    /// Called with each tool outcome, including denials and skipped calls.
    pub fn on_tool_result(
        mut self,
        hook: impl Fn(&ToolCallId, &Result<ToolOutput, Error>) + Send + Sync + 'static,
    ) -> Self {
        self.on_tool_result = Some(Arc::new(hook));
        self
    }

    /// Build the agent.
    ///
    /// Async so connecting MCP servers later does not change the signature.
    /// Nothing is awaited yet.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidRequest`] when the model is missing, or when a
    /// provider is required and was not set. Returns [`Error::Serialization`]
    /// when the model string does not parse.
    pub async fn build(self) -> Result<Agent> {
        let model = self
            .model
            .ok_or_else(|| invalid("missing_model", "agent build requires a model"))?;
        let model = ModelString::parse(&model)?;
        #[cfg(any(feature = "hosted", feature = "lancedb"))]
        let provider = match self.provider {
            Some(provider) => provider,
            None => resolve_builtin(&model).await?,
        };
        Ok(Agent {
            #[cfg(any(feature = "hosted", feature = "lancedb"))]
            provider,
            model,
            system: self.system,
            tools: Arc::from(self.tools),
            options: self.options,
            store: self.store,
            on_tool_call: self.on_tool_call,
            on_request: self.on_request,
            on_response: self.on_response,
            on_tool_result: self.on_tool_result,
        })
    }
}

/// The result of one completed turn.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct Turn {
    /// The assistant message as appended to the conversation.
    pub message: Message,
    /// Token usage for the turn.
    pub usage: Usage,
    /// Why the turn ended.
    pub stop_reason: StopReason,
    /// Retrieval queries the agent issued for this turn, in order.
    #[serde(default)]
    pub knowledge_queries: Vec<String>,
}

/// A caller decision about whether to run a tool call.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum Approval {
    /// The tool may run.
    Approve,
    /// Refuse; a denial goes back to the model as a tool result with
    /// `is_error: true`.
    Deny,
}

/// The tool-call information a hook sees before the tool runs.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct ToolCall {
    /// The id of the call.
    pub id: ToolCallId,
    /// The model-facing tool name.
    pub name: String,
    /// Raw arguments JSON, as the model produced it.
    pub input: Box<serde_json::value::RawValue>,
    /// Untrusted annotations from the tool, when it declares any. Reaches the
    /// approval hook as context and is never used to skip approval.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub annotations: Option<ToolAnnotations>,
}

/// How retrieved knowledge is attached to a turn.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[non_exhaustive]
pub enum KnowledgeMode {
    /// Retrieve and inject automatically, prefixed with the system prompt.
    #[default]
    Inject,
    /// Expose retrieval as a tool the model may call.
    Tool,
}

/// The provider named by the model prefix, when the caller did not install one.
#[cfg(any(feature = "hosted", feature = "lancedb"))]
async fn resolve_builtin(
    model: &crate::provider::ModelString,
) -> Result<Arc<dyn crate::provider::Provider>> {
    match model.provider {
        crate::provider::ProviderKind::Local => crate::provider::open_local(&model.id).await,
        _ => Err(invalid(
            "missing_provider",
            "agent build requires a provider",
        )),
    }
}

fn invalid(code: &str, message: &str) -> Error {
    Error::InvalidRequest(ProviderError {
        request_id: None,
        status: None,
        code: Some(code.to_string()),
        message: message.to_string(),
        retryable: false,
    })
}

#[cfg(any(feature = "hosted", feature = "lancedb"))]
struct ModelStep {
    message: Message,
    usage: Usage,
    stop: StopReason,
}

#[cfg(any(feature = "hosted", feature = "lancedb"))]
struct PendingCall {
    id: ToolCallId,
    name: String,
    input: Box<serde_json::value::RawValue>,
}

#[cfg(any(feature = "hosted", feature = "lancedb"))]
async fn drain(mut stream: crate::event::EventStream) -> Result<ModelStep> {
    let mut parts = Vec::new();
    let mut text = String::new();
    let mut usage = Usage::default();
    let mut stop = None;

    while let Some(event) = stream.next().await {
        match event {
            Event::TextDelta { text: delta } => text.push_str(&delta),
            Event::ReasoningDelta {
                text: delta,
                summary,
            } => {
                flush_text(&mut text, &mut parts);
                push_reasoning(&mut parts, delta, summary);
            }
            Event::ToolCall { id, name, input } => {
                flush_text(&mut text, &mut parts);
                parts.push(Part::ToolCall { id, name, input });
            }
            Event::ToolCallDelta { .. } | Event::Provider { .. } => {}
            Event::Usage(next) => usage = next,
            Event::Stop(reason) => stop = Some(reason),
            Event::Error(err) => return Err(err),
        }
    }
    flush_text(&mut text, &mut parts);

    let stop = stop.ok_or_else(|| {
        Error::Provider(ProviderError {
            request_id: None,
            status: None,
            code: Some("stream_ended".to_string()),
            message: "stream ended without a terminal event".to_string(),
            retryable: false,
        })
    })?;
    Ok(ModelStep {
        message: Message::parts(Role::Assistant, parts),
        usage,
        stop,
    })
}

#[cfg(any(feature = "hosted", feature = "lancedb"))]
fn flush_text(text: &mut String, parts: &mut Vec<Part>) {
    if !text.is_empty() {
        parts.push(Part::Text {
            text: std::mem::take(text),
        });
    }
}

#[cfg(any(feature = "hosted", feature = "lancedb"))]
fn push_reasoning(parts: &mut Vec<Part>, delta: String, summary: bool) {
    if let Some(Part::Reasoning {
        text,
        summary: existing,
    }) = parts.last_mut()
    {
        if *existing == summary {
            text.push_str(&delta);
            return;
        }
    }
    parts.push(Part::Reasoning {
        text: delta,
        summary,
    });
}

#[cfg(any(feature = "hosted", feature = "lancedb"))]
fn pending_calls(message: &Message) -> Vec<PendingCall> {
    message
        .parts
        .iter()
        .filter_map(|part| match part {
            Part::ToolCall { id, name, input } => Some(PendingCall {
                id: id.clone(),
                name: name.clone(),
                input: input.clone(),
            }),
            _ => None,
        })
        .collect()
}

#[cfg(any(feature = "hosted", feature = "lancedb"))]
fn should_run_tools(stop: &StopReason, calls: &[PendingCall]) -> bool {
    if calls.is_empty() {
        return false;
    }
    !matches!(
        stop,
        StopReason::MaxTokens
            | StopReason::ContextWindowExceeded
            | StopReason::StopSequence { .. }
            | StopReason::Refusal { .. }
    )
}

#[cfg(any(feature = "hosted", feature = "lancedb"))]
fn error_part(id: &ToolCallId, message: &str) -> Part {
    Part::ToolResult {
        call_id: id.clone(),
        output: ToolOutput::text(message),
        is_error: true,
    }
}

#[cfg(any(feature = "hosted", feature = "lancedb"))]
fn refused(category: Option<String>, explanation: Option<String>) -> Error {
    let message = explanation
        .clone()
        .unwrap_or_else(|| "the model refused".to_string());
    Error::Refused(Refusal {
        category,
        explanation,
        error: ProviderError {
            request_id: None,
            status: Some(200),
            code: Some("refusal".to_string()),
            message,
            retryable: false,
        },
    })
}

#[cfg(any(feature = "hosted", feature = "lancedb"))]
fn add_usage(total: &mut Usage, extra: &Usage) {
    total.input_tokens = total.input_tokens.saturating_add(extra.input_tokens);
    total.output_tokens = total.output_tokens.saturating_add(extra.output_tokens);
    add_opt(&mut total.cache_read_tokens, extra.cache_read_tokens);
    add_opt(&mut total.cache_write_tokens, extra.cache_write_tokens);
    add_opt(
        &mut total.cache_write_1h_tokens,
        extra.cache_write_1h_tokens,
    );
    add_opt(&mut total.reasoning_tokens, extra.reasoning_tokens);
    add_cost(&mut total.cost, extra.cost.clone());
    total.batch |= extra.batch;
}

#[cfg(any(feature = "hosted", feature = "lancedb"))]
fn add_opt(slot: &mut Option<u64>, extra: Option<u64>) {
    if let Some(extra) = extra {
        *slot = Some(slot.unwrap_or(0).saturating_add(extra));
    }
}

#[cfg(any(feature = "hosted", feature = "lancedb"))]
fn add_cost(slot: &mut Option<Cost>, extra: Option<Cost>) {
    match (slot.as_mut(), extra) {
        (Some(total), Some(extra)) => total.usd += extra.usd,
        (None, Some(extra)) => *slot = Some(extra),
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_approval_hook_sees_the_call_and_returns_an_approval() {
        let call = ToolCall {
            id: ToolCallId("call_1".to_string()),
            name: "exec".to_string(),
            input: serde_json::value::RawValue::from_string(r#"{"command":"ls"}"#.to_string())
                .expect("valid raw json"),
            annotations: Some(ToolAnnotations {
                destructive_hint: true,
                ..ToolAnnotations::default()
            }),
        };

        // The approval gate's shape, per §8.1: `Fn(&ToolCall) -> Approval`.
        let gate = |call: &ToolCall| -> Approval {
            if call
                .annotations
                .as_ref()
                .is_some_and(|annotations| annotations.destructive_hint)
            {
                Approval::Deny
            } else {
                Approval::Approve
            }
        };

        assert_eq!(gate(&call), Approval::Deny);
        let json = serde_json::to_string(&call).expect("the call serializes");
        assert!(json.contains("\"exec\""));
    }
}

#[cfg(all(test, any(feature = "hosted", feature = "lancedb")))]
mod run_tests {
    use std::collections::VecDeque;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Mutex;

    use super::*;
    use crate::event::EventStream;
    use crate::message::Part;
    use crate::provider::{BatchSupport, Capabilities, ProviderKind, ThinkingSupport};
    use crate::store::MemoryStore;

    struct Script {
        replies: Mutex<VecDeque<Vec<Event>>>,
        seen: Mutex<Vec<ChatRequest>>,
    }

    impl Script {
        fn new(replies: Vec<Vec<Event>>) -> Arc<Self> {
            Arc::new(Self {
                replies: Mutex::new(VecDeque::from(replies)),
                seen: Mutex::new(Vec::new()),
            })
        }

        fn requests(&self) -> Vec<ChatRequest> {
            self.seen.lock().expect("test").clone()
        }
    }

    #[async_trait::async_trait]
    impl Provider for Script {
        fn kind(&self) -> ProviderKind {
            ProviderKind::Custom
        }

        async fn chat_stream(&self, request: ChatRequest) -> Result<EventStream> {
            self.seen.lock().expect("test").push(request);
            let events = self
                .replies
                .lock()
                .expect("test")
                .pop_front()
                .unwrap_or_default();
            Ok(Box::pin(futures::stream::iter(events)))
        }

        fn capabilities(&self, _model: &str) -> Capabilities {
            caps(ThinkingSupport::NotApplicable, true)
        }
    }

    struct AlwaysOn;

    #[async_trait::async_trait]
    impl Provider for AlwaysOn {
        fn kind(&self) -> ProviderKind {
            ProviderKind::Custom
        }

        async fn chat_stream(&self, _request: ChatRequest) -> Result<EventStream> {
            Err(Error::Unsupported {
                feature: "should-not-send".to_string(),
                provider: "custom".to_string(),
                model: None,
            })
        }

        fn capabilities(&self, _model: &str) -> Capabilities {
            caps(ThinkingSupport::AlwaysOn, false)
        }
    }

    fn caps(thinking: ThinkingSupport, tools: bool) -> Capabilities {
        Capabilities {
            context_window: None,
            max_output_tokens: None,
            streaming: true,
            tools,
            structured_output: false,
            thinking,
            effort_levels: Vec::new(),
            images: false,
            pdf: false,
            batch: BatchSupport::Unsupported,
            cache_min_tokens: None,
        }
    }

    fn text_reply(text: &str) -> Vec<Event> {
        vec![
            Event::TextDelta {
                text: text.to_string(),
            },
            Event::Usage(Usage {
                input_tokens: 3,
                output_tokens: 2,
                ..Usage::default()
            }),
            Event::Stop(StopReason::EndTurn),
        ]
    }

    fn tool_reply(id: &str, name: &str, input: &str) -> Vec<Event> {
        vec![
            Event::ToolCall {
                id: ToolCallId(id.to_string()),
                name: name.to_string(),
                input: serde_json::value::RawValue::from_string(input.to_string())
                    .expect("tool input is json"),
            },
            Event::Stop(StopReason::ToolUse),
        ]
    }

    async fn agent(provider: Arc<dyn Provider>) -> Agent {
        Agent::builder()
            .model("openai/gpt-5.1")
            .provider(provider)
            .build()
            .await
            .expect("builds")
    }

    #[cfg(not(feature = "local"))]
    #[tokio::test]
    async fn a_local_path_without_the_local_feature_is_unsupported() {
        let err = Agent::builder()
            .model("local//tmp/missing.gguf")
            .build()
            .await
            .expect_err("local feature is off");
        assert!(matches!(err, Error::Unsupported { .. }));
    }

    #[tokio::test]
    async fn build_requires_a_model_and_a_provider() {
        let missing_model = Agent::builder().build().await.expect_err("no model");
        assert!(matches!(missing_model, Error::InvalidRequest(_)));

        let missing_provider = Agent::builder()
            .model("openai/gpt-5.1")
            .build()
            .await
            .expect_err("no provider");
        assert!(matches!(missing_provider, Error::InvalidRequest(_)));
    }

    #[tokio::test]
    async fn run_appends_the_turn_and_keeps_the_system_prompt_off_history() {
        let script = Script::new(vec![text_reply("done")]);
        let store = Arc::new(MemoryStore::new());
        let agent = Agent::builder()
            .model("openai/gpt-5.1")
            .provider(script.clone())
            .system("Be brief.")
            .conversation_store(store.clone())
            .build()
            .await
            .expect("builds");

        let mut conversation = Conversation::new();
        let turn = agent
            .run(&mut conversation, Message::user("hello"))
            .await
            .expect("runs");

        assert_eq!(turn.message.text(), "done");
        assert_eq!(conversation.messages().len(), 2);
        assert_eq!(conversation.messages()[0].text(), "hello");
        assert_eq!(conversation.messages()[1].text(), "done");
        assert!(conversation
            .messages()
            .iter()
            .all(|message| message.text() != "Be brief."));
        let requests = script.requests();
        assert_eq!(requests.len(), 1);
        assert_eq!(requests[0].system.as_deref(), Some("Be brief."));
        assert!(matches!(requests[0].tool_choice, ToolChoice::None));

        let saved = store
            .load(&conversation.id)
            .await
            .expect("loads")
            .expect("saved");
        assert_eq!(saved.messages().len(), 2);
    }

    #[tokio::test]
    async fn with_system_and_with_model_clone() {
        let script = Script::new(vec![]);
        let agent = Agent::builder()
            .model("openai/gpt-5.1")
            .system("old")
            .provider(script)
            .build()
            .await
            .expect("builds");

        let swapped = agent.with_system("new");
        let local = agent.with_model("local/Qwen/Qwen3-8B").expect("parses");

        assert_eq!(agent.model().as_str(), "openai/gpt-5.1");
        assert_eq!(swapped.model().as_str(), "openai/gpt-5.1");
        assert_eq!(local.model().id, "Qwen/Qwen3-8B");
        assert!(agent.with_model("no-slash").is_err());
    }

    #[tokio::test]
    async fn a_tool_call_runs_and_the_result_goes_back_in_one_message() {
        let called = Arc::new(AtomicBool::new(false));
        let flag = called.clone();
        let weather = crate::tool::tool_fn_sync(
            "get_weather",
            "Look up the weather",
            move |args: WeatherArgs| {
                flag.store(true, Ordering::SeqCst);
                Ok(ToolOutput::text(format!("Sunny in {}", args.city)))
            },
        );
        let script = Script::new(vec![
            tool_reply("call_1", "get_weather", r#"{"city":"Paris"}"#),
            text_reply("It is sunny."),
        ]);
        let agent = Agent::builder()
            .model("openai/gpt-5.1")
            .provider(script.clone())
            .tool(weather)
            .build()
            .await
            .expect("builds");

        let mut conversation = Conversation::new();
        let turn = agent
            .run(&mut conversation, Message::user("weather?"))
            .await
            .expect("runs");

        assert!(called.load(Ordering::SeqCst));
        assert_eq!(turn.message.text(), "It is sunny.");
        assert_eq!(conversation.messages().len(), 4);
        let result = &conversation.messages()[2];
        assert!(matches!(
            &result.parts[0],
            Part::ToolResult {
                is_error: false,
                ..
            }
        ));
        assert_eq!(script.requests().len(), 2);
        assert!(matches!(script.requests()[0].tool_choice, ToolChoice::Auto));
        assert!(script.requests()[1].messages.len() >= 3);
    }

    #[tokio::test]
    async fn a_denial_is_an_error_result_and_the_tool_does_not_run() {
        let called = Arc::new(AtomicBool::new(false));
        let flag = called.clone();
        let exec = crate::tool::tool_fn_sync("exec", "Run a command", move |_args: WeatherArgs| {
            flag.store(true, Ordering::SeqCst);
            Ok(ToolOutput::text("ran"))
        });
        let script = Script::new(vec![
            tool_reply("call_1", "exec", r#"{"city":"x"}"#),
            text_reply("stopped"),
        ]);
        let agent = Agent::builder()
            .model("openai/gpt-5.1")
            .provider(script)
            .tool(exec)
            .on_tool_call(|_| Approval::Deny)
            .build()
            .await
            .expect("builds");

        let mut conversation = Conversation::new();
        agent
            .run(&mut conversation, Message::user("do it"))
            .await
            .expect("runs");

        assert!(!called.load(Ordering::SeqCst));
        assert!(matches!(
            &conversation.messages()[2].parts[0],
            Part::ToolResult { is_error: true, .. }
        ));
    }

    #[tokio::test]
    async fn the_step_limit_closes_pending_calls_without_running_them() {
        let called = Arc::new(AtomicBool::new(false));
        let flag = called.clone();
        let weather =
            crate::tool::tool_fn_sync("get_weather", "weather", move |_args: WeatherArgs| {
                flag.store(true, Ordering::SeqCst);
                Ok(ToolOutput::text("sunny"))
            });
        let script = Script::new(vec![tool_reply(
            "call_1",
            "get_weather",
            r#"{"city":"Paris"}"#,
        )]);
        let agent = Agent::builder()
            .model("openai/gpt-5.1")
            .provider(script.clone())
            .tool(weather)
            .options(RunOptions {
                max_steps: Some(1),
                ..RunOptions::default()
            })
            .build()
            .await
            .expect("builds");

        let mut conversation = Conversation::new();
        let turn = agent
            .run(&mut conversation, Message::user("weather?"))
            .await
            .expect("runs");

        assert!(!called.load(Ordering::SeqCst));
        assert!(matches!(turn.stop_reason, StopReason::Other(_)));
        assert_eq!(script.requests().len(), 1);
        assert!(matches!(
            &conversation.messages()[2].parts[0],
            Part::ToolResult { is_error: true, .. }
        ));
    }

    #[tokio::test]
    async fn a_refusal_rolls_the_conversation_back() {
        let script = Script::new(vec![vec![Event::Stop(StopReason::Refusal {
            category: Some("safety".to_string()),
            explanation: Some("no".to_string()),
        })]]);
        let agent = agent(script).await;
        let mut conversation = Conversation::new();
        let err = agent
            .run(&mut conversation, Message::user("nope"))
            .await
            .expect_err("refused");

        assert!(matches!(err, Error::Refused(_)));
        assert!(conversation.messages().is_empty());
    }

    #[tokio::test]
    async fn thinking_off_is_rejected_before_a_request_on_an_always_on_model() {
        let agent = Agent::builder()
            .model("anthropic/claude-opus-5-5")
            .provider(Arc::new(AlwaysOn))
            .options(RunOptions {
                thinking: crate::options::Thinking::Off,
                ..RunOptions::default()
            })
            .build()
            .await
            .expect("builds");
        let mut conversation = Conversation::new();
        let err = agent
            .run(&mut conversation, Message::user("hi"))
            .await
            .expect_err("thinking cannot be off");

        assert!(matches!(err, Error::InvalidRequest(_)));
        assert!(conversation.messages().is_empty());
    }

    #[derive(serde::Deserialize, schemars::JsonSchema)]
    struct WeatherArgs {
        city: String,
    }
}
