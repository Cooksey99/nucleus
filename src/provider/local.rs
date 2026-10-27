//! The local provider.
//!
//! A `local//absolute/path.gguf` model string is a file already on disk. This
//! module loads that file. The engine underneath is private.

use std::path::Path;
use std::sync::Arc;

use mistralrs::{GgufModelBuilder, Model, RequestBuilder, TextMessageRole, TextMessages};

use crate::error::{Error, ProviderError, Result};
use crate::event::{Event, EventStream, StopReason, Usage};
use crate::message::Role;
use crate::provider::{
    BatchSupport, Capabilities, ChatRequest, Provider, ProviderKind, ThinkingSupport,
};

/// A model running in-process. Clones share the loaded weights.
pub(crate) struct LocalProvider {
    model: Arc<Model>,
}

impl LocalProvider {
    /// Open the GGUF file at `path`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Io`] when the file is missing, and [`Error::Provider`]
    /// when the file cannot be loaded.
    pub(crate) async fn open(path: &str) -> Result<Self> {
        if !path.starts_with('/') {
            return Err(Error::InvalidRequest(ProviderError {
                request_id: None,
                status: None,
                code: Some("local_path".to_string()),
                message: format!("local// expects an absolute path, got '{path}'"),
                retryable: false,
            }));
        }
        let file = Path::new(path);
        if !file.is_file() {
            return Err(Error::Io(format!("local model file not found: {path}")));
        }
        let dir = file
            .parent()
            .and_then(|dir| dir.to_str())
            .ok_or_else(|| Error::Io(format!("invalid model path: {path}")))?;
        let filename = file
            .file_name()
            .and_then(|name| name.to_str())
            .ok_or_else(|| Error::Io(format!("invalid model filename: {path}")))?;
        let model = GgufModelBuilder::new(dir, vec![filename])
            .build()
            .await
            .map_err(|err| {
                Error::Provider(ProviderError {
                    request_id: None,
                    status: None,
                    code: Some("local_load".to_string()),
                    message: format!("failed to load '{path}': {err}"),
                    retryable: false,
                })
            })?;
        Ok(Self {
            model: Arc::new(model),
        })
    }
}

#[async_trait::async_trait]
impl Provider for LocalProvider {
    fn kind(&self) -> ProviderKind {
        ProviderKind::Local
    }

    async fn chat_stream(&self, request: ChatRequest) -> Result<EventStream> {
        let response = self
            .model
            .send_chat_request(to_request(&request))
            .await
            .map_err(|err| {
                Error::Provider(ProviderError {
                    request_id: None,
                    status: None,
                    code: Some("local_chat".to_string()),
                    message: err.to_string(),
                    retryable: false,
                })
            })?;

        let choice = response.choices.first();
        let text = choice
            .and_then(|choice| choice.message.content.clone())
            .unwrap_or_default();
        let mut events = Vec::new();
        if !text.is_empty() {
            events.push(Event::TextDelta { text });
        }
        events.push(Event::Usage(Usage {
            input_tokens: response.usage.prompt_tokens as u64,
            output_tokens: response.usage.completion_tokens as u64,
            ..Usage::default()
        }));
        events.push(Event::Stop(StopReason::EndTurn));
        Ok(Box::pin(futures::stream::iter(events)))
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

fn to_request(request: &ChatRequest) -> RequestBuilder {
    let mut messages = TextMessages::new();
    if let Some(system) = &request.system {
        messages = messages.add_message(TextMessageRole::System, system);
    }
    for message in &request.messages {
        let text = message.text();
        if text.is_empty() {
            continue;
        }
        let role = match message.role {
            Role::User => TextMessageRole::User,
            Role::Assistant => TextMessageRole::Assistant,
        };
        messages = messages.add_message(role, &text);
    }
    RequestBuilder::from(messages)
}
