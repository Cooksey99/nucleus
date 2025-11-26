use crate::config::Config;
use crate::ollama::{self, Client, ChatRequest, Message, Tool, ToolFunction};
use nucleus_plugin::PluginRegistry;
use anyhow::{Context, Result};
use std::sync::Arc;

pub struct ChatManager {
    /// Nucleus core configuration
    config: Config,
    /// Ollama client
    ollama: Client,
    /// Registry for available plugins
    registry: Arc<PluginRegistry>,
}

impl ChatManager {
    pub fn new(config: Config, registry: Arc<PluginRegistry>) -> Self {
        let ollama = Client::new(&config.llm.base_url);
        Self {
            config,
            ollama,
            registry,
        }
    }

    pub async fn query(&self, user_message: &str) -> Result<String> {
        let mut messages = vec![Message::user(user_message)];

        let tools = self.build_tools();
        eprintln!("[DEBUG] Tools available: {}", serde_json::to_string_pretty(&tools).unwrap());

        loop {
            let mut request = ChatRequest::new(&self.config.llm.model, messages.clone())
                .with_temperature(self.config.llm.temperature);

            if !tools.is_empty() {
                request.tools = Some(tools.clone());
            }

            let mut accumulated_content = String::new();
            let mut current_response: Option<ollama::ChatResponse> = None;

            self.ollama
                .chat(request, |response| {
                    accumulated_content.push_str(&response.message.content);
                    current_response = Some(response);
                })
                .await
                .context("Failed to get LLM response")?;

            let mut response = current_response
                .context("No response from LLM")?;

            response.message.content = accumulated_content;
            let assistant_message = response.message;

            if let Some(tool_calls) = &assistant_message.tool_calls {
                messages.push(Message {
                    role: "assistant".to_string(),
                    content: assistant_message.content.clone(),
                    images: None,
                    tool_calls: Some(tool_calls.clone()),
                });

                for tool_call in tool_calls {
                    let tool_name = &tool_call.function.name;
                    let tool_args = &tool_call.function.arguments;

                    let result = self
                        .registry
                        .execute(tool_name, tool_args.clone())
                        .await
                        .with_context(|| format!("Failed to execute tool: {}", tool_name))?;

                    messages.push(Message {
                        role: "tool".to_string(),
                        content: result.content,
                        images: None,
                        tool_calls: None,
                    });
                }
            } else {
                return Ok(assistant_message.content);
            }
        }
    }

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
