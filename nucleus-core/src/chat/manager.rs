//! Chat conversation management with tool-augmented LLM capabilities.
//!
//! This module provides the core conversation orchestration for nucleus,
//! managing multi-turn chats with streaming responses and tool execution.
//!
//! # Architecture
//!
//! The `ChatManager` implements a tool-augmented LLM pattern:
//! - Sends user queries to the LLM with available tool definitions
//! - Detects when the LLM requests tool execution
//! - Executes tools from the plugin registry
//! - Returns tool results to the LLM for final response generation
//!
//! # Tool Calling Flow
//!
//! ```text
//! User Query → LLM → Tool Call?
//!                ↓         ↓
//!             Response   Execute Tool
//!                          ↓
//!                     LLM with Result → Response
//! ```
//!
//! # Streaming Behavior
//!
//! The LLM streams responses in chunks. Tool calls may arrive in early chunks
//! while the final `done=true` chunk contains no tool calls. The manager
//! preserves tool calls from any chunk to ensure they're not lost.

use crate::config::Config;
use crate::models::EmbeddingModel;
use crate::provider::{
    create_provider, ChatRequest, ChatResponse, Message, Provider, ProviderType, StructuredOutput,
    Tool, ToolCall, ToolFunction,
};
use crate::rag::RagEngine;
use anyhow::{Context, Result};
use futures::future::join_all;
use nucleus_plugin::{Permission, PluginRegistry};
use std::path::Path;
use std::sync::Arc;
use tokio::sync::Mutex;
use tracing::{debug, info};

/// Manages multi-turn conversations with tool-augmented LLM capabilities.
///
/// `ChatManager` orchestrates interactions between the user, LLM, and available
/// tools (plugins). It handles streaming responses, tool execution, and conversation
/// state management.
///
/// # Examples
///
/// ```no_run
/// use nucleus_core::{ChatManager, Config};
/// use nucleus_plugin::{PluginRegistry, Permission};
///
/// # async fn example() -> anyhow::Result<()> {
/// let config = Config::load_or_default();
/// let registry = PluginRegistry::new(Permission::READ_ONLY);
/// let manager = ChatManager::new(config, registry).await?;
///
/// let response = manager.query("What files are in the current directory?").await?;
/// println!("AI: {}", response);
/// # Ok(())
/// # }
/// ```
///
/// # Tool Execution
///
/// When the LLM requests a tool, the manager:
/// 1. Adds the assistant message with tool calls to conversation history
/// 2. Executes each tool via the plugin registry
/// 3. Adds tool results as messages
/// 4. Continues the conversation loop for the LLM to synthesize a response
///
/// # Important Notes
///
/// - Tool calls arrive in streaming chunks and must be preserved across chunks
/// - The conversation loop continues until the LLM returns a non-tool response
/// - One manager is one conversation. Follow-up queries continue that history
///   until [`clear_history`](Self::clear_history) or [`set_history`](Self::set_history)
pub struct ChatManager {
    /// Nucleus core configuration
    pub config: Config,
    /// LLM provider for communication
    provider: Arc<dyn Provider>,
    /// Registry for available plugins/tools
    registry: Arc<PluginRegistry>,
    /// RAG manager for knowledge base integration (with persistent storage)
    rag_engine: Option<Arc<RagEngine>>,
    /// Optional JSON schema for forcing a structured JSON output
    pub structured_output: Option<StructuredOutput>,
    /// Stored conversation. Held for the duration of a turn so overlapping
    /// queries cannot interleave messages.
    history: Mutex<Vec<Message>>,
}

impl ChatManager {
    /// Creates a new chat manager with default configuration.
    ///
    /// Creates a non-persistent RAG manager that stores knowledge in memory only.
    /// For custom RAG configuration (including persistence), use [`with_rag`](Self::with_rag).
    ///
    /// # Arguments
    ///
    /// * `config` - Nucleus configuration including LLM settings
    /// * `registry` - Plugin registry containing available tools. The registry is wrapped
    ///   in an `Arc` internally and shared between the manager and provider for tool execution.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use nucleus_core::{ChatManager, Config};
    /// use nucleus_plugin::{PluginRegistry, Permission};
    ///
    /// # async fn example() -> anyhow::Result<()> {
    /// let config = Config::load_or_default();
    /// let registry = PluginRegistry::new(Permission::READ_ONLY);
    /// let manager = ChatManager::new(config, registry).await?;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn new(config: Config, registry: impl Into<Arc<PluginRegistry>>) -> Result<Self> {
        Self::builder()
            .with_config(config)
            .with_registry(registry.into())
            .build()
            .await
    }

    /// Creates a builder for configuring the chat manager.
    ///
    /// The builder allows overriding LLM and embedding models while preserving
    /// other configuration from the provided `Config`.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use nucleus_core::{ChatManager, Config};
    /// use nucleus_plugin::{PluginRegistry, Permission};
    ///
    /// # async fn example() -> anyhow::Result<()> {
    /// let config = Config::load_or_default();
    /// let registry = PluginRegistry::new(Permission::READ_ONLY);
    ///
    /// // Override LLM model
    /// let manager = ChatManager::builder(config.clone(), registry.clone())
    ///     .with_llm_model("Qwen/Qwen3-1.6B-Instruct")
    ///     .build()
    ///     .await?;
    ///
    /// // Override both LLM and embedding models
    /// let manager = ChatManager::builder(config, registry)
    ///     .with_llm_model("Qwen/Qwen3-1.6B-Instruct")
    ///     .with_embedding_model("BAAI/bge-small-en-v1.5")
    ///     .build()
    ///     .await?;
    /// # Ok(())
    /// # }
    /// ```
    fn builder() -> ChatManagerBuilder {
        ChatManagerBuilder::new()
    }

    ///
    /// # Examples
    ///
    /// ```no_run
    /// use nucleus_core::{ChatManager, Config};
    /// use nucleus_core::provider::MistralRsProvider;
    /// use nucleus_plugin::{PluginRegistry, Permission};
    /// use std::sync::Arc;
    ///
    /// # async fn example() -> anyhow::Result<()> {
    /// let config = Config::load_or_default();
    /// let registry = PluginRegistry::new(Permission::READ_ONLY);
    ///
    /// let manager = ChatManager::new(config, registry).await?
    ///     .with_provider(Arc::new(MistralRsProvider::new("qwen3:0.6b"))).await?;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn with_provider(mut self, provider: Arc<dyn Provider>) -> Result<Self> {
        self.rag_engine = Some(Arc::new(RagEngine::new(&self.config, provider.clone()).await?));
        self.provider = provider;
        Ok(self)
    }

    /// Replace the RAG manager.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use nucleus_core::{ChatManager, Config, RagEngine};
    /// # use nucleus_plugin::{PluginRegistry, Permission};
    /// # use std::sync::Arc;
    ///
    /// # async fn example() -> anyhow::Result<()> {
    /// let config = Config::load_or_default();
    /// let registry = PluginRegistry::new(Permission::READ_ONLY);
    /// let provider = Arc::new(/* create provider */);
    ///
    /// let custom_rag = RagEngine::new(&config, provider).await?;
    /// let manager = ChatManager::new(config, registry).await?
    ///     .with_rag(custom_rag);
    /// # Ok(())
    /// # }
    /// ```
    pub fn with_rag(mut self, rag: Arc<RagEngine>) -> Self {
        self.rag_engine = Some(rag);
        self
    }

    /// Loads previously indexed documents from persistent storage.
    ///
    /// Should be called after creating the ChatManager to restore the knowledge base.
    ///
    /// # Returns
    ///
    /// The number of documents loaded from disk.
    ///
    /// # Errors
    ///
    /// Returns an error if loading fails.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use nucleus_core::{ChatManager, Config};
    /// # use nucleus_plugin::{PluginRegistry, Permission};
    /// # async fn example() -> anyhow::Result<()> {
    /// # let config = Config::load_or_default();
    /// # let registry = PluginRegistry::new(Permission::READ_ONLY);
    /// let manager = ChatManager::new(config, registry).await?;
    /// let count = manager.load_knowledge_base().await?;
    /// println!("Loaded {} documents", count);
    /// # Ok(())
    /// # }
    /// ```
    pub async fn knowledge_base_count(&self) -> usize {
        match self.rag_engine.as_ref() {
            Some(engine) => engine.count().await,
            None => 0
        }
    }

    /// Indexes a directory into the knowledge base.
    ///
    /// # Arguments
    ///
    /// * `dir_path` - Path to the directory to index
    ///
    /// # Returns
    ///
    /// The number of files successfully indexed.
    ///
    /// # Errors
    ///
    /// Returns an error if indexing fails.
    pub async fn index_directory(&self, dir_path: &Path) -> Result<usize> {
        match self.rag_engine.as_ref() {
            Some(engine) => engine.index_directory(dir_path).await.context("Failed to index directory"),
            None => Err(anyhow::anyhow!("RAG Engine not configured"))
        }
    }

    pub async fn index_text(&self, content: &str, source: &str) -> Result<()> {
        match self.rag_engine.as_ref() {
            Some(engine) => engine.add_knowledge(content, source).await.context("Failed to index text"),
            None => Err(anyhow::anyhow!("RAG Engine not configured"))
        }
    }

    pub async fn clear_knowledge_base(&self) -> Result<()> {
        match self.rag_engine.as_ref() {
            Some(engine) => engine.clear().await.context("Unable to clear RAG knowledge base"),
            None => Err(anyhow::anyhow!("RAG Engine not configured"))
        }
    }

    /// Returns whether RAG is configured on this manager.
    pub fn is_rag_enabled(&self) -> bool {
        self.rag_engine.is_some()
    }

    /// Retrieves the RAG context that would be used to augment a query.
    ///
    /// Returns an empty string when RAG is disabled, the knowledge base is empty,
    /// or no relevant context is found.
    pub async fn preview_rag_context(&self, query: &str) -> Result<String> {
        match self.rag_engine.as_ref() {
            Some(engine) => {
                if engine.count().await == 0 {
                    Ok(String::new())
                } else {
                    engine
                        .retrieve_context(query)
                        .await
                        .context("Failed to retrieve RAG context")
                }
            }
            None => Ok(String::new()),
        }
    }

    /// Sets the structured output for the `ChatManager`.
    pub fn set_structured_output(&mut self, schema: serde_json::Value) {
        self.structured_output = Some(StructuredOutput::new(schema));
    }

    /// Removed any previously set `structured_output` JSON schema from `ChatManager`
    pub fn clear_structured_output(&mut self) {
        self.structured_output = None;
    }

    /// Snapshot of the stored conversation, including system, user, assistant, and tool turns.
    pub async fn history(&self) -> Vec<Message> {
        self.history.lock().await.clone()
    }

    /// Replace the stored conversation.
    ///
    /// The caller owns the system prompt and prior turns. The next [`query`](Self::query)
    /// appends to this history instead of seeding a new one, unless `messages` is empty.
    pub async fn set_history(&self, messages: Vec<Message>) {
        *self.history.lock().await = messages;
    }

    /// Drop stored turns. The next query starts clean and re-seeds the system prompt.
    pub async fn clear_history(&self) {
        self.history.lock().await.clear();
    }

    /// Run a slash command, if `input` starts with `/`.
    ///
    /// Returns [`CommandEffect::NotACommand`] for ordinary text, including the word
    /// `reset`. Callers that want `/reset` should use this before [`query`](Self::query).
    pub async fn handle_command(&self, input: &str) -> crate::chat::CommandEffect {
        use crate::chat::commands::{self, Action};
        use crate::chat::CommandEffect;

        match commands::action(input) {
            Action::NotACommand => CommandEffect::NotACommand,
            Action::ClearHistory => {
                self.clear_history().await;
                CommandEffect::Handled {
                    message: "History cleared.".to_string(),
                }
            }
            Action::Reply(message) => CommandEffect::Handled { message },
            Action::Exit => CommandEffect::Exit {
                message: "Exiting.".to_string(),
            },
        }
    }

    /// Sends a query and returns the final response.
    ///
    /// The manager keeps the conversation. This call appends `user_message`, runs any
    /// tool loop, records the assistant reply, and later calls continue from there.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use nucleus_core::{ChatManager, Config};
    /// # use nucleus_plugin::PluginRegistry;
    /// # use std::sync::Arc;
    /// # async fn example() -> anyhow::Result<()> {
    /// # let config = Config::load_or_default();
    /// # let registry = Arc::new(PluginRegistry::new(nucleus_plugin::Permission::READ_ONLY));
    /// # let manager = ChatManager::new(config, registry).await?;
    /// let response = manager.query("Summarize the README file").await?;
    /// let follow_up = manager.query("What does the license section say?").await?;
    /// println!("{response}\n{follow_up}");
    /// manager.clear_history().await;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn query(&self, user_message: &str) -> Result<String> {
        self.query_stream(user_message, |_| {}).await
    }

    /// Streaming version of [`query`](Self::query).
    ///
    /// Appends the user turn to stored history, streams the reply, and records the
    /// assistant turn (plus any tool turns) before returning.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use nucleus_core::{ChatManager, Config};
    /// # use nucleus_plugin::PluginRegistry;
    /// # use std::sync::Arc;
    /// # use std::io::{self, Write};
    /// # async fn example() -> anyhow::Result<()> {
    /// # let config = Config::load_or_default();
    /// # let registry = Arc::new(PluginRegistry::new(nucleus_plugin::Permission::READ_ONLY));
    /// # let manager = ChatManager::new(config, registry).await?;
    /// let response = manager.query_stream("Tell me a story", |chunk| {
    ///     print!("{}", chunk);
    ///     io::stdout().flush().unwrap();
    /// }).await?;
    /// println!("Final response: {}", response);
    /// # Ok(())
    /// # }
    /// ```
    pub async fn query_stream<F>(&self, user_message: &str, on_chunk: F) -> Result<String>
    where
        F: FnMut(&str) + Send,
    {
        let mut history = self.history.lock().await;
        self.run_turn(&mut history, user_message, on_chunk).await
    }

    /// Run one turn against a supplied history without reading or writing stored history.
    ///
    /// Use this for a trial, a branch, or a caller that owns the message vec.
    /// The supplied messages are not modified; only the returned text is produced.
    pub async fn query_with(&self, messages: &[Message], user_message: &str) -> Result<String> {
        self.query_with_stream(messages, user_message, |_| {}).await
    }

    /// Streaming version of [`query_with`](Self::query_with).
    pub async fn query_with_stream<F>(
        &self,
        messages: &[Message],
        user_message: &str,
        on_chunk: F,
    ) -> Result<String>
    where
        F: FnMut(&str) + Send,
    {
        let mut messages = messages.to_vec();
        self.run_turn(&mut messages, user_message, on_chunk).await
    }

    /// Converts registered plugins into tool definitions.
    ///
    /// Transforms plugins from the registry into the JSON schema format
    /// Each plugin becomes a tool with its name, description, and parameter schema.
    ///
    /// # Note
    ///
    /// This method is called once at the start of each query. Tools are
    /// included in every LLM request throughout the conversation loop.
    async fn build_tools(&self) -> Vec<Tool> {
        join_all(self.registry.all().iter().map(async move |plugin| {
            let plugin = plugin.lock().await;
            let spec = plugin.parameter_schema();
            Tool {
                tool_type: "function".to_string(),
                function: ToolFunction {
                    name: plugin.name().to_string(),
                    description: plugin.description().to_string(),
                    parameters: spec,
                },
            }
        }))
        .await
    }

    /// Run one user turn, appending user, tool, and assistant messages to `messages`.
    ///
    /// On failure, messages added during this turn are removed so a retry starts clean.
    async fn run_turn<F>(
        &self,
        messages: &mut Vec<Message>,
        user_message: &str,
        mut on_chunk: F,
    ) -> Result<String>
    where
        F: FnMut(&str) + Send,
    {
        let checkpoint = messages.len();
        match self
            .run_turn_inner(messages, user_message, &mut on_chunk)
            .await
        {
            Ok(content) => Ok(content),
            Err(err) => {
                messages.truncate(checkpoint);
                Err(err)
            }
        }
    }

    async fn run_turn_inner<F>(
        &self,
        messages: &mut Vec<Message>,
        user_message: &str,
        mut on_chunk: F,
    ) -> Result<String>
    where
        F: FnMut(&str) + Send,
    {
        self.seed_system_prompt(messages);
        let context = self.retrieve_context(user_message).await;
        messages.push(Self::user_message(&context, user_message));

        let tools = self.build_tools().await;

        loop {
            let mut request = ChatRequest::new(&self.config.llm.model, messages.clone())
                .with_temperature(self.config.llm.temperature);

            if !tools.is_empty() {
                request.tools = Some(tools.clone());
            }

            if let Some(structured_output) = &self.structured_output {
                request = request.with_structured_output(structured_output.clone());
            }

            let assistant_message = self.process_response_stream(request, &mut on_chunk).await?;

            if let Some(tool_calls) = assistant_message
                .tool_calls
                .filter(|calls| !calls.is_empty())
            {
                messages.push(Message {
                    role: "assistant".to_string(),
                    context: Some(context.clone()),
                    content: assistant_message.content.clone(),
                    images: None,
                    tool_calls: Some(tool_calls.clone()),
                });

                for tool_call in tool_calls {
                    let tool_name = &tool_call.function.name;
                    info!(tool_name = %tool_name, "Executing tool");

                    let result = self
                        .registry
                        .execute(tool_name, tool_call.function.arguments.clone())
                        .await
                        .with_context(|| format!("Failed to execute tool: {tool_name}"))?;

                    messages.push(Message {
                        role: "tool".to_string(),
                        context: Some(context.clone()),
                        content: result.content,
                        images: None,
                        tool_calls: None,
                    });
                }

                continue;
            }

            let content = assistant_message.content;
            messages.push(Message {
                role: "assistant".to_string(),
                context: Some(context.clone()),
                content: content.clone(),
                images: None,
                tool_calls: None,
            });
            return Ok(content);
        }
    }

    fn seed_system_prompt(&self, messages: &mut Vec<Message>) {
        if messages.is_empty() && !self.config.system_prompt.is_empty() {
            messages.push(Message::system(None, self.config.system_prompt.clone()));
        }
    }

    fn user_message(context: &str, user_message: &str) -> Message {
        let enhanced_message = if context.is_empty() {
            debug!("No RAG context available, using original message");
            user_message.to_string()
        } else {
            debug!(
                "Enhanced message with {} characters of RAG context",
                context.len()
            );
            format!("{context}{user_message}")
        };

        Message::user(Some(context.to_string()), enhanced_message)
    }

    /// Retrieve RAG context for this user turn only. Older turns are left unchanged.
    async fn retrieve_context(&self, user_message: &str) -> String {
        match self.rag_engine.as_ref() {
            Some(engine) => {
                let count = engine.count().await;
                debug!("RAG knowledge base has {} documents", count);

                if count == 0 {
                    debug!("RAG knowledge base is empty, skipping context retrieval");
                    return String::new();
                }

                debug!("Retrieving RAG context for query: {}", user_message);
                engine.retrieve_context(user_message).await.unwrap_or_else(|e| {
                    debug!("Could not retrieve RAG context: {}", e);
                    String::new()
                })
            }
            None => {
                debug!("RAG engine not configured, skipping context retrieval");
                String::new()
            }
        }
    }

    /// Process LLM response stream and accumulate content.
    ///
    /// Handles streaming response chunks, accumulates content, and preserves
    /// tool calls from any chunk in the stream.
    ///
    /// # Arguments
    ///
    /// * `request` - The chat request to send to the LLM
    /// * `on_chunk` - Callback for streaming content chunks
    ///
    /// # Returns
    ///
    /// The complete assistant message with accumulated content and preserved tool calls.
    async fn process_response_stream<F>(
        &self,
        request: ChatRequest,
        mut on_chunk: F,
    ) -> Result<Message>
    where
        F: FnMut(&str) + Send,
    {
        let mut accumulated_content = String::new();
        let mut final_response: Option<ChatResponse> = None;
        let mut tool_calls: Option<Vec<ToolCall>> = None;

        self.provider
            .chat(
                request,
                Box::new(|response| {
                    if !response.done && !response.content.is_empty() {
                        on_chunk(&response.content);
                        accumulated_content.push_str(&response.content);
                    }

                    if let Some(ref calls) = response.message.tool_calls {
                        tool_calls = Some(calls.clone());
                    }

                    final_response = Some(response);
                }),
            )
            .await
            .context("Failed to get LLM response")?;

        let mut response = final_response.context("No response from LLM")?;
        response.message.content = accumulated_content;
        response.message.tool_calls = tool_calls;

        Ok(response.message)
    }
}

/// Builder for configuring and creating a `ChatManager`.
///
/// This builder provides a fluent API for customizing LLM and embedding models
/// while maintaining sensible defaults from the config.
///
/// # Examples
///
/// ```no_run
/// use nucleus_core::{ChatManager, Config};
/// use nucleus_plugin::{PluginRegistry, Permission};
///
/// # async fn example() -> anyhow::Result<()> {
/// let config = Config::load_or_default();
/// let registry = PluginRegistry::new(Permission::READ_ONLY);
///
/// // Use defaults from config
/// let manager = ChatManager::builder(config.clone(), registry.clone())
///     .build()
///     .await?;
///
/// // Override LLM model
/// let manager = ChatManager::builder(config.clone(), registry.clone())
///     .with_llm_model("Qwen/Qwen3-1.6B-Instruct")
///     .build()
///     .await?;
///
/// // Override embedding model
/// let manager = ChatManager::builder(config.clone(), registry.clone())
///     .with_embedding_model("Qwen/Qwen3-Embedding-0.6B")
///     .build()
///     .await?;
///
/// // Override both
/// let manager = ChatManager::builder(config, registry)
///     .with_llm_model("Qwen/Qwen3-1.6B-Instruct")
///     .with_embedding_model("Qwen/Qwen3-Embedding-0.6B")
///     .build()
///     .await?;
/// # Ok(())
/// # }
/// ```
pub struct ChatManagerBuilder {
    config: Config,
    registry: Arc<PluginRegistry>,
    llm_model_override: Option<String>,
    embedding_model_override: Option<EmbeddingModel>,
    provider_type_override: Option<ProviderType>,
    structured_output: Option<StructuredOutput>,
}

impl ChatManagerBuilder {
    /// Creates a new builder with the given config and registry.
    pub fn new() -> Self {
        let config = Config::default();
        let registry = Arc::new(PluginRegistry::new(Permission::NONE));
        Self {
            config,
            registry,
            llm_model_override: None,
            embedding_model_override: None,
            provider_type_override: None,
            structured_output: None,
        }
    }

    pub fn with_config(mut self, config: Config) -> Self {
        self.config = config;
        self
    }

    pub fn with_registry(mut self, registry: impl Into<Arc<PluginRegistry>>) -> Self {
        self.registry = registry.into();
        self
    }
    /// Override the default LLM model from the configuration.
    ///
    /// Accepts a model identifier, which may be:
    /// - A Hugging Face repo ID: `"Qwen/Qwen3-1.6B-Instruct"`
    /// - A local GGUF path: `"/path/to/model.gguf"`
    /// - A quantized model name: `"TheBloke/Llama-2-7B-Chat-GGUF"`
    ///
    /// # Examples
    ///
    /// **Hugging Face model**
    /// ```
    /// # use nucleus_core::{ChatManager, Config};
    /// # use nucleus_plugin::{PluginRegistry, Permission};
    /// # async fn example() -> anyhow::Result<()> {
    /// let manager = ChatManager::builder(Config::load_or_default(), PluginRegistry::new(Permission::READ_ONLY))
    ///     .with_llm_model("Qwen/Qwen3-1.6B-Instruct")
    ///     .build()
    ///     .await?;
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// **Local GGUF model**
    /// ```
    /// # use nucleus_core::{ChatManager, Config};
    /// # use nucleus_plugin::{PluginRegistry, Permission};
    /// # async fn example() -> anyhow::Result<()> {
    /// let manager = ChatManager::builder(Config::load_or_default(), PluginRegistry::new(Permission::READ_ONLY))
    ///     .with_llm_model("/Users/alice/models/mistral-7b-instruct-v0.2.Q4_K_M.gguf")
    ///     .build()
    ///     .await?;
    /// # Ok(())
    /// # }
    /// ````
    ///
    /// **Local GGUF Blob (Ollama) — NOT CURRENTLY SUPPORTED**
    /// ```
    /// let manager = ChatManager::builder(config, registry)
    ///     .with_llm_model("~/.ollama/models/blobs/sha256-0d003f6662faee786ed5da3e31b29c978de5ae5d275c8794c606a7f3c01aa8f5")  // Q4_K_M
    ///     .build()
    ///     .await?;
    /// ```
    pub fn with_llm_model(mut self, model: impl Into<String>) -> Self {
        self.llm_model_override = Some(model.into());
        self
    }

    /// Override the embedding model from config.
    ///
    /// # Arguments
    ///
    /// * `model` - Embedding model identifier (HuggingFace repo, GGUF path, etc.)
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use nucleus_core::{ChatManager, Config};
    /// # use nucleus_plugin::{PluginRegistry, Permission};
    /// # async fn example() -> anyhow::Result<()> {
    /// # let config = Config::load_or_default();
    /// # let registry = PluginRegistry::new(Permission::READ_ONLY);
    /// let manager = ChatManager::builder(config, registry)
    ///     .with_embedding_model("Qwen/Qwen3-Embedding-0.6B")
    ///     .build()
    ///     .await?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn with_embedding_model(mut self, model: impl Into<EmbeddingModel>) -> Self {
        self.embedding_model_override = Some(model.into());
        self
    }

    /// Override the provider type.
    ///
    /// This allows you to specify which LLM provider to use (Ollama, MistralRs, or CoreML).
    /// The provider will be constructed automatically using the config and registry.
    ///
    /// # Arguments
    ///
    /// * `provider_type` - The type of provider to use
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use nucleus_core::{ChatManager, Config};
    /// # use nucleus_core::provider::ProviderType;
    /// # use nucleus_plugin::{PluginRegistry, Permission};
    /// # async fn example() -> anyhow::Result<()> {
    /// # let config = Config::load_or_default();
    /// # let registry = PluginRegistry::new(Permission::READ_ONLY);
    /// let manager = ChatManager::builder()
    ///     .with_config(config)
    ///     .with_registry(registry)
    ///     .with_provider(ProviderType::CoreML)
    ///     .build()
    ///     .await?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn with_provider(mut self, provider_type: ProviderType) -> Self {
        self.provider_type_override = Some(provider_type);
        self
    }

    /// Builds the `ChatManager` with the configured settings.
    ///
    /// This initializes the provider with the (possibly overridden) LLM model,
    /// and the RAG system with the (possibly overridden) embedding model.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The provider fails to initialize
    /// - The RAG system fails to initialize
    pub async fn build(self) -> Result<ChatManager> {
        let mut config = self.config.clone();

        if let Some(llm_model) = self.llm_model_override {
            config.llm.model = llm_model;
        }

        if let Some(provider_type) = self.provider_type_override {
            config.llm.provider = provider_type.as_str().to_string();
        }

        let provider = create_provider(&config, Arc::clone(&self.registry)).await?;
        let mut rag_engine = None;

        if config.rag.is_some() {
            if let Some(embedding_model) = self.embedding_model_override {
                if let Some(rag_config) = config.rag.as_mut() {
                    rag_config.embedding_model = embedding_model;
                }
            }

            rag_engine = Some(Arc::new(RagEngine::new(&config, provider.clone()).await?));
        }


        Ok(ChatManager {
            config,
            provider,
            registry: self.registry,
            rag_engine,
            structured_output: self.structured_output,
            history: Mutex::new(Vec::new()),
        })
    }
}
