//! mistral.rs provider implementation.
//!
//! This module provides an in-process LLM provider using mistral.rs.
//! Supports both local GGUF files and automatic HuggingFace downloads.

use crate::Config;

use super::types::*;
use async_trait::async_trait;
use mistralrs::{
    CalledFunction, Function, GgufModelBuilder, IsqType, Model, RequestBuilder, TextMessageRole, TextMessages, TextModelBuilder, Tool as MistralTool, ToolCallback, ToolChoice, ToolType
};
use nucleus_plugin::{Plugin, PluginRegistry};
use tracing::{debug, info, warn};

use std::path::Path;
use std::sync::Arc;

/// mistral.rs in-process provider.
///
/// Automatically detects if model is:
/// 1. A local GGUF file path (loads directly)
/// 2. A HuggingFace model ID (downloads if needed)
///
/// Note: Use async `new()` - model loading requires async operations.
pub struct MistralRsProvider {
    model: Arc<Model>,
    model_name: String,
    registry: Arc<PluginRegistry>,
}

impl MistralRsProvider {
    /// Creates a new mistral.rs provider.
    ///
    /// Downloads and loads the model. This may take time on first use.
    ///
    /// # Model Resolution
    ///
    /// - `"repo:file.gguf"` - HuggingFace GGUF (pre-quantized, fastest)
    /// - `"/path/file.gguf"` - Local GGUF file
    /// - `"Repo/Model-ID"` - HuggingFace model (quantizes on load)
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use nucleus_core::provider::MistralRsProvider;
    /// # async fn example() -> anyhow::Result<()> {
    /// // Pre-quantized GGUF from HuggingFace (recommended, fastest)
    /// let provider = MistralRsProvider::new("Qwen/Qwen3-0.6B-Instruct-GGUF:qwen3-0_6b-instruct-q4_k_m.gguf").await?;
    ///
    /// // Local GGUF file
    /// let provider = MistralRsProvider::new("./models/qwen3-0.6b.gguf").await?;
    ///
    /// // HuggingFace model (slow, quantizes on load)
    /// let provider = MistralRsProvider::new("Qwen/Qwen3-0.6B-Instruct").await?;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn new(config: Config, registry: Arc<PluginRegistry>) -> Result<Self> {
        let model_name = config.llm.model.clone();
        let model = Self::build_model(config.clone(), Arc::clone(&registry)).await?;

        Ok(Self {
            model: Arc::new(model),
            model_name,
            registry,
        })
    }

    async fn build_model(config: Config, registry: Arc<PluginRegistry>) -> Result<Model> {
        let model_name = config.llm.model;

        // Detect model type
        let is_local_gguf = model_name.ends_with(".gguf") && Path::new(&model_name).exists();
        let is_hf_gguf = model_name.contains(':') && model_name.ends_with(".gguf");
        
        let model = if is_hf_gguf {
            // Format: "Repo/Model-GGUF:filename.gguf"
            let parts: Vec<&str> = model_name.split(':').collect();
            if parts.len() != 2 {
                return Err(ProviderError::Other(
                    format!("Invalid GGUF format. Expected 'Repo/Model-GGUF:file.gguf', got '{}'" , model_name)
                ));
            }
            
            // Note: PagedAttention is intentionally omitted here.
            // .with_paged_attn() can cause indefinite hangs on macOS with Metal/GPU
            // due to initialization issues in mistral.rs (as of Dec 2024).
            // The library will automatically disable it when needed anyway.
            GgufModelBuilder::new(parts[0], vec![parts[1]])
                .with_logging()
                .build()
                .await
                .map_err(|e| ProviderError::Other(
                    format!("Failed to load GGUF '{}' from HuggingFace: {:?}", model_name, e)
                ))?
        } else if is_local_gguf {
            // Extract path and filename to load modal
            let path = Path::new(&model_name);
            let dir = path.parent()
                .ok_or_else(|| ProviderError::Other("Invalid GGUF file path".to_string()))?
                .to_str()
                .ok_or_else(|| ProviderError::Other("Invalid UTF-8 in path".to_string()))?;
            let filename = path.file_name()
                .ok_or_else(|| ProviderError::Other("Invalid GGUF filename".to_string()))?
                .to_str()
                .ok_or_else(|| ProviderError::Other("Invalid UTF-8 in filename".to_string()))?;

            GgufModelBuilder::new(dir, vec![filename])
                .with_logging()
                .build()
                .await
                .map_err(|e| ProviderError::Other(format!("Failed to load local GGUF '{}': {:?}", model_name, e)))?
        } else {
            // Download from HuggingFace if not cached  
            let mut builder = TextModelBuilder::new(&model_name)
                .with_isq(IsqType::Q4K) // 4-bit quantization
                .with_logging();

            for plugin in registry.all().into_iter() {
                builder = builder.with_tool_callback(plugin.name(), plugin_to_callback(plugin));
            }
            
            builder.build()
                .await
                .map_err(|e| ProviderError::Other(
                    format!("Failed to load model '{}'. Make sure it exists on HuggingFace or is a valid local .gguf file: {:?}", 
                        model_name, e)
                ))?
        };

        Ok(model)
    }

}

/// Convert the nucleus plugin structure to the mistralrs tool structure
fn plugin_to_callback(plugin: &Arc<dyn Plugin>) -> Arc<ToolCallback> {
    let plugin = Arc::clone(plugin);

    Arc::new(move |called_fn: &CalledFunction| {
        // Get arguments from the called function
        let args: serde_json::Value = serde_json::from_str(&called_fn.arguments)
            .map_err(|e| ProviderError::Other(format!("Failed to parse tool arguments: {}", e)))?;

        let handle = tokio::runtime::Handle::try_current()
            .map_err(|e| ProviderError::Other(format!("No tokio runtime available: {}", e)))?;

        let result = handle.block_on(async {
            plugin.execute(args).await
        })
        .map_err(|e| ProviderError::Other(format!("Plugin execution failed: {}", e)))?;

        Ok(result.content)
    })
}


#[async_trait]
impl Provider for MistralRsProvider {
    async fn chat<'a>(
        &'a self,
        request: ChatRequest,
        mut callback: Box<dyn FnMut(ChatResponse) + Send + 'a>,
    ) -> Result<()> {
        // Build messages using TextMessages builder
        let mut messages = TextMessages::new();
        
        for msg in &request.messages {
            let role = match msg.role.as_str() {
                "system" => TextMessageRole::System,
                "user" => TextMessageRole::User,
                "assistant" => TextMessageRole::Assistant,
                "tool" => TextMessageRole::Tool,
                _ => TextMessageRole::User,
            };
            
            messages = messages.add_message(role, &msg.content);
        }

        // Convert to RequestBuilder
        let mut builder = RequestBuilder::from(messages);

        // Convert plugins to mistral.rs tool definitions
        if self.registry.get_count() > 0 {
            let plugins = self.registry.all();
            info!(plugin_count = plugins.len(), "Converting plugins to mistral.rs tools");
            
            let mistral_tools: Vec<MistralTool> = plugins
                .iter()
                .map(|plugin| {
                    let schema = plugin.parameter_schema();
                    debug!(
                        tool_name = %plugin.name(),
                        description = %plugin.description(),
                        "Processing plugin"
                    );
                    debug!(parameters = ?schema, "Plugin parameter schema");
                    
                    // Extract properties from JSON Schema format
                    // Input: {"type": "object", "properties": {"path": {...}}, "required": [...]}
                    // Output: HashMap<String, Value> of just the properties
                    let parameters = if let Some(props) = schema.get("properties") {
                        if let Some(obj) = props.as_object() {
                            let extracted = obj.clone().into_iter().collect();
                            debug!(
                                properties = ?obj.keys().collect::<Vec<_>>(),
                                "Extracted tool properties"
                            );
                            Some(extracted)
                        } else {
                            warn!("Plugin properties field is not an object");
                            None
                        }
                    } else {
                        debug!("No properties field in schema, using as-is");
                        serde_json::from_value(schema).ok()
                    };
                    
                    MistralTool {
                        tp: ToolType::Function,
                        function: Function {
                            name: plugin.name().to_string(),
                            description: Some(plugin.description().to_string()),
                            parameters,
                        },
                    }
                })
                .collect();
            
            info!(tool_count = mistral_tools.len(), "Setting tools with ToolChoice::Auto");
            builder = builder.set_tools(mistral_tools).set_tool_choice(ToolChoice::Auto);
        }

        // Send request and get response with timeout to prevent hangs
        debug!("Sending chat request to mistral.rs");
        let timeout_duration = std::time::Duration::from_secs(60);
        let response = tokio::time::timeout(
            timeout_duration,
            self.model.send_chat_request(builder)
        )
        .await
        .map_err(|_| {
            warn!(timeout_secs = timeout_duration.as_secs(), "Chat request timed out");
            ProviderError::Other(
                format!("Chat request timed out after {} seconds. This may indicate a hang in mistral.rs with tool calling.", 
                    timeout_duration.as_secs())
            )
        })?
        .map_err(|e| ProviderError::Other(format!("Chat request failed: {:?}", e)))?;
        debug!("Received response from mistral.rs");

        let choice = response.choices.first()
            .ok_or_else(|| ProviderError::Other("No response choices".to_string()))?;
        
        let content = choice.message.content.as_ref()
            .map(|s| s.to_string())
            .unwrap_or_default();

        // Convert tool calls back to our format
        let tool_calls = choice.message.tool_calls.as_ref().map(|tcs| {
            tcs.iter().map(|tc| super::types::ToolCall {
                function: super::types::ToolCallFunction {
                    name: tc.function.name.clone(),
                    arguments: serde_json::from_str(&tc.function.arguments)
                        .unwrap_or(serde_json::json!({})),
                },
            }).collect()
        });

        // Send complete response through callback
        callback(ChatResponse {
            model: self.model_name.clone(),
            content: content.clone(),
            done: true,
            message: Message {
                role: "assistant".to_string(),
                content,
                images: None,
                tool_calls,
            },
        });

        Ok(())
    }

    async fn embed(&self, _text: &str, _model: &str) -> Result<Vec<f32>> {
        Err(ProviderError::Other(
            "Embeddings not yet supported for mistral.rs provider".to_string(),
        ))
    }
}
