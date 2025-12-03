//! mistral.rs provider implementation.
//!
//! This module provides an in-process LLM provider using mistral.rs.
//! Supports both local GGUF files and automatic HuggingFace downloads.

use super::types::*;
use async_trait::async_trait;
use mistralrs::{
    Function, GgufModelBuilder, IsqType, Model, RequestBuilder,
    TextMessageRole, TextMessages, TextModelBuilder, Tool as MistralTool,
    ToolChoice, ToolType,
};

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
    pub async fn new(model_name: impl Into<String>) -> Result<Self> {
        let model_name = model_name.into();
        let model = Self::build_model(&model_name).await?;

        Ok(Self {
            model: Arc::new(model),
            model_name,
        })
    }

    async fn build_model(model_name: &str) -> Result<Model> {
        // Detect model type
        let is_local_gguf = model_name.ends_with(".gguf") && Path::new(model_name).exists();
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
            TextModelBuilder::new(&model_name)
                .with_isq(IsqType::Q4K) // 4-bit quantization
                .with_logging()
                .build()
                .await
                .map_err(|e| ProviderError::Other(
                    format!("Failed to load model '{}'. Make sure it exists on HuggingFace or is a valid local .gguf file: {:?}", 
                        model_name, e)
                ))?
        };

        Ok(model)
    }

    
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

        // Add tools if provided
        if let Some(tools) = &request.tools {
            let mistral_tools: Vec<MistralTool> = tools
                .iter()
                .map(|t| MistralTool {
                    tp: ToolType::Function,
                    function: Function {
                        name: t.function.name.clone(),
                        description: Some(t.function.description.clone()),
                        parameters: Some(
                            serde_json::from_value(t.function.parameters.clone())
                                .unwrap_or_default()
                        ),
                    },
                })
                .collect();
            builder = builder.set_tools(mistral_tools).set_tool_choice(ToolChoice::Auto);
        }

        // Send request and get response
        let response = self.model
            .send_chat_request(builder)
            .await
            .map_err(|e| ProviderError::Other(format!("Chat request failed: {:?}", e)))?;

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
