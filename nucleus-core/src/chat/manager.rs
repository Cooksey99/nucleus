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
use crate::provider::{ChatRequest, ChatResponse, Message, MistralRsProvider, Provider, Tool, ToolCall, ToolFunction};
use crate::rag::Rag;
use nucleus_plugin::PluginRegistry;
use anyhow::{Context, Result};
use std::path::Path;
use std::sync::Arc;
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
/// - All conversation history is maintained for context
pub struct ChatManager {
    /// Nucleus core configuration
    config: Config,
    /// LLM provider for communication
    provider: Arc<dyn Provider>,
    /// Registry for available plugins/tools
    registry: Arc<PluginRegistry>,
    /// RAG manager for knowledge base integration (with persistent storage)
    rag: Rag,
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
    pub async fn new(config: Config, registry: PluginRegistry) -> Result<Self> {
        Self::builder(config, registry).build().await
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
    pub fn builder(config: Config, registry: PluginRegistry) -> ChatManagerBuilder {
        ChatManagerBuilder::new(config, registry)
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
        self.rag = Rag::new(&self.config, provider.clone()).await?;
        self.provider = provider;
        Ok(self)
    }
    
    /// Replace the RAG manager.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use nucleus_core::{ChatManager, Config, Rag};
    /// # use nucleus_plugin::{PluginRegistry, Permission};
    /// # use std::sync::Arc;
    ///
    /// # async fn example() -> anyhow::Result<()> {
    /// let config = Config::load_or_default();
    /// let registry = PluginRegistry::new(Permission::READ_ONLY);
    /// let provider = Arc::new(/* create provider */);
    /// 
    /// let custom_rag = Rag::new(&config, provider).await?;
    /// let manager = ChatManager::new(config, registry).await?
    ///     .with_rag(custom_rag);
    /// # Ok(())
    /// # }
    /// ```
    pub fn with_rag(mut self, rag: Rag) -> Self {
        self.rag = rag;
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
        self.rag.count().await
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
        self.rag.index_directory(dir_path).await
            .context("Failed to index directory")
    }

    /// Sends a query to the LLM and returns the final response.
    ///
    /// This method handles the complete conversation flow including:
    /// - Sending the initial user message
    /// - Processing tool calls requested by the LLM
    /// - Executing tools via the plugin registry
    /// - Continuing the conversation until a final response is generated
    ///
    /// # Arguments
    ///
    /// * `user_message` - The user's question or prompt
    ///
    /// # Returns
    ///
    /// The LLM's final response after any necessary tool executions.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The LLM request fails
    /// - A requested tool fails to execute
    /// - The response cannot be parsed
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
    /// # let manager = ChatManager::new(config, registry);
    /// let response = manager.query("Summarize the README file").await?;
    /// println!("Response: {}", response);
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Implementation Details
    ///
    /// This method runs in a loop:
    /// 1. Send current messages to LLM with available tools
    /// 2. Stream the response, preserving tool calls from any chunk
    /// 3. If tool calls present:
    ///    - Execute each tool
    ///    - Add tool results to conversation
    ///    - Continue loop
    /// 4. If no tool calls, return the response
    ///
    /// The loop ensures the LLM can chain multiple tool calls if needed.
    pub async fn query(&self, user_message: &str) -> Result<String> {
        self.query_stream(user_message, |_| {}).await
    }
    
    /// Send a query to the LLM and stream the response through a callback.
    ///
    /// This is the streaming version of [`query`](Self::query). It allows you to
    /// process response chunks as they arrive instead of waiting for the complete
    /// response.
    ///
    /// # Arguments
    ///
    /// * `user_message` - The user's question or prompt
    /// * `on_chunk` - Callback invoked for each chunk of streaming content.
    ///   Receives the incremental content (not accumulated).
    ///
    /// # Returns
    ///
    /// The LLM's final complete response after any necessary tool executions.
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
    /// // Print response as it streams
    /// let response = manager.query_stream("Tell me a story", |chunk| {
    ///     print!("{}", chunk);
    ///     io::stdout().flush().unwrap();
    /// }).await?;
    /// println!("\n\nFinal response: {}", response);
    /// # Ok(())
    /// # }
    /// ```
    pub async fn query_stream<F>(&self, user_message: &str, mut on_chunk: F) -> Result<String>
    where
        F: FnMut(&str) + Send,
    {
        // Retrieve relevant context from knowledge base if available
        let rag_count = self.rag.count().await;
        debug!("RAG knowledge base has {} documents", rag_count);
        
        // Context retrieved from RAG
        let context = if rag_count > 0 {
            debug!("Retrieving RAG context for query: {}", user_message);
            self.rag.retrieve_context(user_message).await
                .unwrap_or_else(|e| {
                    debug!("Could not retrieve RAG context: {}", e);
                    String::new()
                })
        } else {
            debug!("RAG knowledge base is empty, skipping context retrieval");
            String::new()
        };
        
        // Construct user message with context if available
        let enhanced_message = if !context.is_empty() {
            debug!("Enhanced message with {} characters of RAG context", context.len());
            format!("{}{}", context, user_message)
        } else {
            debug!("No RAG context available, using original message");
            user_message.to_string()
        };
        
        let mut messages = vec![Message::user(Some(context.clone()), &enhanced_message)];

        let tools = self.build_tools();

        loop {
            let mut request = ChatRequest::new(&self.config.llm.model, messages.clone())
                .with_temperature(self.config.llm.temperature);

            if !tools.is_empty() {
                request.tools = Some(tools.clone());
            }

            // Stream the LLM response, accumulating content and preserving tool calls.
            // Important: Tool calls may arrive in early chunks while content streams,
            // so we must preserve them separately from the final chunk.
            let mut accumulated_content = String::new();
            let mut current_response: Option<ChatResponse> = None;
            let mut tool_calls: Option<Vec<ToolCall>> = None;
            self.provider
                .chat(request, Box::new(|response| {
                    // Call user's streaming callback with incremental content
                    if !response.done && !response.content.is_empty() {
                        on_chunk(&response.content);
                    }
                    
                    // Accumulate incremental content (response.content), not full message
                    accumulated_content.push_str(&response.content);
                    
                    // Preserve tool calls from any chunk - they typically arrive early
                    // in the stream and may be absent from the final done=true chunk
                    if let Some(ref tool_calls_ref) = response.message.tool_calls {
                        tool_calls = Some(tool_calls_ref.clone());
                    }
                    
                    current_response = Some(response);
                }))
                .await
                .context("Failed to get LLM response")?;

            let mut response = current_response
                .context("No response from LLM")?;

            // Reconstruct the complete message with accumulated content and preserved tool calls
            response.message.content = accumulated_content;
            response.message.tool_calls = tool_calls;
            let assistant_message = response.message;

            // Handle tool calls: execute each tool and add results to conversation
            if let Some(tool_calls) = &assistant_message.tool_calls {
                // Add the assistant's message with tool calls to conversation history
                messages.push(Message {
                    role: "assistant".to_string(),
                    context: Some(context.to_string()),
                    content: assistant_message.content.clone(),
                    images: None,
                    tool_calls: Some(tool_calls.clone()),
                });

                // Execute each requested tool and add results
                for tool_call in tool_calls {
                    let tool_name = &tool_call.function.name;
                    let tool_args = &tool_call.function.arguments;
                    info!(tool_name = %tool_name, "Executing tool");

                    let result = self
                        .registry
                        .execute(tool_name, tool_args.clone())
                        .await
                        .with_context(|| format!("Failed to execute tool: {}", tool_name))?;

                    // Add tool result as a message for the LLM to synthesize
                    messages.push(Message {
                        role: "tool".to_string(),
                        context: Some(context.to_string()),
                        content: result.content,
                        images: None,
                        tool_calls: None,
                    });
                }
                // Continue loop to get LLM's response using the tool results
            } else {
                // No tool calls - this is the final response
                return Ok(assistant_message.content);
            }
        }
    }

    /// Converts registered plugins into Ollama tool definitions.
    ///
    /// Transforms plugins from the registry into the JSON schema format
    /// expected by Ollama's tool calling API. Each plugin becomes a tool
    /// with its name, description, and parameter schema.
    ///
    /// # Returns
    ///
    /// A vector of tool definitions to send with LLM requests.
    ///
    /// # Note
    ///
    /// This method is called once at the start of each query. Tools are
    /// included in every LLM request throughout the conversation loop.
    fn build_tools(&self) -> Vec<Tool> {
        self.registry
            .all()
            .iter()
            .map(|plugin| {
                let spec = plugin.parameter_schema();
                Tool {
                    tool_type: "function".to_string(),
                    function: ToolFunction {
                        name: plugin.name().to_string(),
                        description: plugin.description().to_string(),
                        parameters: spec,
                    },
                }
            })
            .collect()
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
    registry: PluginRegistry,
    llm_model_override: Option<String>,
    embedding_model_override: Option<String>,
}

impl ChatManagerBuilder {
    /// Creates a new builder with the given config and registry.
    pub fn new(config: Config, registry: PluginRegistry) -> Self {
        Self {
            config,
            registry,
            llm_model_override: None,
            embedding_model_override: None,
        }
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
    pub fn with_embedding_model(mut self, model: impl Into<String>) -> Self {
        self.embedding_model_override = Some(model.into());
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
        let mut config = self.config;

        if let Some(llm_model) = self.llm_model_override {
            config.llm.model = llm_model;
        }
        if let Some(embedding_model) = self.embedding_model_override {
            config.rag.embedding_model = embedding_model;
        }

        let registry = Arc::new(self.registry);
        let provider: Arc<dyn Provider> = Arc::new(
            MistralRsProvider::new(&config, Arc::clone(&registry)).await?
        );
        let rag = Rag::new(&config, provider.clone()).await?;

        Ok(ChatManager {
            config,
            provider,
            registry,
            rag,
        })
    }
}
