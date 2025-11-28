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
use crate::ollama::{self, Client, ChatRequest, Message, Tool, ToolFunction};
use nucleus_plugin::PluginRegistry;
use anyhow::{Context, Result};
use std::sync::Arc;

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
/// use nucleus_plugin::PluginRegistry;
/// use std::sync::Arc;
///
/// # async fn example() -> anyhow::Result<()> {
/// let config = Config::load_default()?;
/// let registry = Arc::new(PluginRegistry::new(nucleus_plugin::Permission::READ_ONLY));
/// let manager = ChatManager::new(config, registry);
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
    /// Ollama client for LLM communication
    ollama: Client,
    /// Registry for available plugins/tools
    registry: Arc<PluginRegistry>,
}

impl ChatManager {
    /// Creates a new chat manager with the given configuration and plugin registry.
    ///
    /// # Arguments
    ///
    /// * `config` - Nucleus configuration including LLM settings
    /// * `registry` - Plugin registry containing available tools
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use nucleus_core::{ChatManager, Config};
    /// use nucleus_plugin::{PluginRegistry, Permission};
    /// use std::sync::Arc;
    ///
    /// # fn example() -> anyhow::Result<()> {
    /// let config = Config::load_default()?;
    /// let registry = Arc::new(PluginRegistry::new(Permission::READ_ONLY));
    /// let manager = ChatManager::new(config, registry);
    /// # Ok(())
    /// # }
    /// ```
    pub fn new(config: Config, registry: Arc<PluginRegistry>) -> Self {
        let ollama = Client::new(&config.llm.base_url);
        Self {
            config,
            ollama,
            registry,
        }
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
    /// # let config = Config::load_default()?;
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
        let mut messages = vec![Message::user(user_message)];

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
            let mut current_response: Option<ollama::ChatResponse> = None;
            let mut tool_calls: Option<Vec<ollama::ToolCall>> = None;

            self.ollama
                .chat(request, |response| {
                    accumulated_content.push_str(&response.message.content);
                    
                    // Preserve tool calls from any chunk - they typically arrive early
                    // in the stream and may be absent from the final done=true chunk
                    if response.message.tool_calls.is_some() {
                        tool_calls = response.message.tool_calls.clone();
                    }
                    
                    current_response = Some(response);
                })
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
                    content: assistant_message.content.clone(),
                    images: None,
                    tool_calls: Some(tool_calls.clone()),
                });

                // Execute each requested tool and add results
                for tool_call in tool_calls {
                    let tool_name = &tool_call.function.name;
                    let tool_args = &tool_call.function.arguments;

                    let result = self
                        .registry
                        .execute(tool_name, tool_args.clone())
                        .await
                        .with_context(|| format!("Failed to execute tool: {}", tool_name))?;

                    // Add tool result as a message for the LLM to synthesize
                    messages.push(Message {
                        role: "tool".to_string(),
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
