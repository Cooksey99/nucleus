//! Ollama provider implementation.
//!
//! This module provides an Ollama HTTP API client that implements the Provider trait.

use super::types::*;
use async_trait::async_trait;
use futures::StreamExt;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Ollama HTTP API provider.
#[derive(Debug, Clone)]
pub struct OllamaProvider {
    base_url: String,
    http_client: reqwest::Client,
}

impl OllamaProvider {
    /// Creates a new Ollama provider with the specified base URL.
    pub fn new(base_url: impl Into<String>) -> Self {
        Self {
            base_url: base_url.into(),
            http_client: reqwest::Client::new(),
        }
    }
}

#[async_trait]
impl Provider for OllamaProvider {
    async fn chat(
        &self,
        request: ChatRequest,
        mut callback: impl FnMut(ChatResponse) + Send,
    ) -> Result<()> {
        let url = format!("{}/api/chat", self.base_url);
        
        // Convert to Ollama-specific request format
        let ollama_request = OllamaChatRequest {
            model: request.model.clone(),
            messages: request.messages.iter().map(|m| OllamaMessage {
                role: m.role.clone(),
                content: m.content.clone(),
                images: m.images.clone(),
                tool_calls: m.tool_calls.as_ref().map(|tcs| {
                    tcs.iter().map(|tc| OllamaToolCall {
                        function: OllamaToolCallFunction {
                            name: tc.function.name.clone(),
                            arguments: tc.function.arguments.clone(),
                        },
                    }).collect()
                }),
            }).collect(),
            options: {
                let mut opts = HashMap::new();
                opts.insert("temperature".to_string(), serde_json::json!(request.temperature));
                Some(opts)
            },
            stream: true,
            tools: request.tools.as_ref().map(|tools| {
                tools.iter().map(|t| OllamaTool {
                    tool_type: t.tool_type.clone(),
                    function: OllamaToolFunction {
                        name: t.function.name.clone(),
                        description: t.function.description.clone(),
                        parameters: t.function.parameters.clone(),
                    },
                }).collect()
            }),
        };
        
        let response = self.http_client
            .post(&url)
            .json(&ollama_request)
            .send()
            .await?;
        
        if !response.status().is_success() {
            let error_text = response.text().await?;
            return Err(ProviderError::Api(error_text));
        }
        
        let mut stream = response.bytes_stream();
        let mut buffer = Vec::new();
        
        while let Some(chunk_result) = stream.next().await {
            let chunk = chunk_result?;
            buffer.extend_from_slice(&chunk);
            
            while let Some(newline_pos) = buffer.iter().position(|&b| b == b'\n') {
                let line = buffer.drain(..=newline_pos).collect::<Vec<_>>();
                
                if line.len() <= 1 {
                    continue;
                }
                
                let line_str = String::from_utf8_lossy(&line[..line.len()-1]);
                
                if let Ok(ollama_response) = serde_json::from_str::<OllamaChatResponse>(&line_str) {
                    // Convert to common ChatResponse
                    callback(ChatResponse {
                        model: ollama_response.model.clone(),
                        content: ollama_response.message.content.clone(),
                        done: ollama_response.done,
                        message: Message {
                            role: ollama_response.message.role.clone(),
                            content: ollama_response.message.content.clone(),
                            images: ollama_response.message.images.clone(),
                            tool_calls: ollama_response.message.tool_calls.as_ref().map(|tcs| {
                                tcs.iter().map(|tc| ToolCall {
                                    function: ToolCallFunction {
                                        name: tc.function.name.clone(),
                                        arguments: tc.function.arguments.clone(),
                                    },
                                }).collect()
                            }),
                        },
                    });
                }
            }
        }
        
        Ok(())
    }
    
    async fn embed(&self, text: &str, model: &str) -> Result<Vec<f32>> {
        let url = format!("{}/api/embed", self.base_url);
        
        let embed_request = EmbedRequest {
            model: model.to_string(),
            input: text.to_string(),
        };
        
        let response = self.http_client
            .post(&url)
            .json(&embed_request)
            .send()
            .await?;
        
        if !response.status().is_success() {
            let error_text = response.text().await?;
            return Err(ProviderError::Api(error_text));
        }
        
        let embed_response = response.json::<EmbedResponse>().await?;
        
        embed_response.embeddings
            .into_iter()
            .next()
            .ok_or_else(|| ProviderError::Other("No embeddings returned".to_string()))
    }
}

// Ollama-specific request/response types (internal)

#[derive(Debug, Clone, Serialize, Deserialize)]
struct OllamaChatRequest {
    model: String,
    messages: Vec<OllamaMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    options: Option<HashMap<String, serde_json::Value>>,
    #[serde(default = "default_stream")]
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<OllamaTool>>,
}

fn default_stream() -> bool {
    true
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct OllamaMessage {
    role: String,
    content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    images: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<OllamaToolCall>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct OllamaChatResponse {
    model: String,
    #[serde(default)]
    created_at: String,
    message: OllamaMessage,
    #[serde(default)]
    done: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    done_reason: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct OllamaTool {
    #[serde(rename = "type")]
    tool_type: String,
    function: OllamaToolFunction,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct OllamaToolFunction {
    name: String,
    description: String,
    parameters: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct OllamaToolCall {
    function: OllamaToolCallFunction,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct OllamaToolCallFunction {
    name: String,
    arguments: serde_json::Value,
}
