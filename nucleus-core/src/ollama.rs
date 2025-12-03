//! Ollama HTTP API client for chat completions.
//!
//! This module provides a Rust interface to the Ollama API, supporting
//! streaming chat completions with the local or remote Ollama server.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use thiserror::Error;
use futures::StreamExt;

/// Errors that can occur when interacting with the Ollama API.
#[derive(Debug, Error)]
pub enum OllamaError {
    #[error("HTTP request failed: {0}")]
    Request(#[from] reqwest::Error),
    
    #[error("JSON parsing failed: {0}")]
    Json(#[from] serde_json::Error),
    
    #[error("API error: {0}")]
    Api(String),
}

pub type Result<T> = std::result::Result<T, OllamaError>;

/// HTTP client for communicating with an Ollama server.
#[derive(Debug, Clone)]
pub struct Client {
    /// The base URL of the Ollama server (e.g., "http://localhost:11434")
    base_url: String,
    http_client: reqwest::Client,
}

impl Client {
    /// Creates a new Ollama client with the specified base URL.
    pub fn new(base_url: impl Into<String>) -> Self {
        Self {
            base_url: base_url.into(),
            http_client: reqwest::Client::new(),
        }
    }
    
    /// Sends a chat request to Ollama and streams the response.
    ///
    /// The response is streamed in chunks, with each chunk being passed to the callback
    /// function as it arrives. This allows for real-time display of the LLM's output.
    ///
    /// # Arguments
    ///
    /// * `request` - The chat request containing the model and messages
    /// * `callback` - Function called for each response chunk received
    ///
    /// # Errors
    ///
    /// Returns an error if the HTTP request fails, the response cannot be parsed,
    /// or the Ollama API returns an error.
    ///
    pub async fn chat(
        &self,
        request: ChatRequest,
        mut callback: impl FnMut(ChatResponse),
    ) -> Result<()> {
        let url = format!("{}/api/chat", self.base_url);
        
        let response = self.http_client
            .post(&url)
            .json(&request)
            .send()
            .await?;
        
        if !response.status().is_success() {
            let error_text = response.text().await?;
            return Err(OllamaError::Api(error_text));
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
                
                if let Ok(chat_response) = serde_json::from_str::<ChatResponse>(&line_str) {
                    callback(chat_response);
                }
            }
        }
        
        Ok(())
    }
    
    /// Generates embeddings for the given text.
    pub async fn embed(&self, request: EmbedRequest) -> Result<EmbedResponse> {
        let url = format!("{}/api/embed", self.base_url);
        
        let response = self.http_client
            .post(&url)
            .json(&request)
            .send()
            .await?;
        
        if !response.status().is_success() {
            let error_text = response.text().await?;
            return Err(OllamaError::Api(error_text));
        }
        
        let embed_response = response.json::<EmbedResponse>().await?;
        
        // Debug: log if embeddings are empty
        if embed_response.embeddings.is_empty() {
            eprintln!("WARNING: Ollama returned empty embeddings for model: {}", request.model);
            eprintln!("         Input length: {}", request.input.len());
        }
        
        Ok(embed_response)
    }
}

/// A single message in a chat conversation.
///
/// Messages have a role (system, user, or assistant) and content.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: String,
    pub content: String,
    
    #[serde(skip_serializing_if = "Option::is_none")]
    pub images: Option<Vec<String>>,
    
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
}

impl Message {
    /// Creates a system message.
    ///
    /// System messages set the behavior and context for the AI assistant.
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: "system".to_string(),
            content: content.into(),
            images: None,
            tool_calls: None,
        }
    }
    
    /// Creates a user message.
    ///
    /// User messages represent input from the person interacting with the AI.
    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: "user".to_string(),
            content: content.into(),
            images: None,
            tool_calls: None,
        }
    }
}

/// Request payload for the Ollama chat API.
///
/// Specifies the model to use, conversation history, and generation options.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatRequest {
    pub model: String,
    pub messages: Vec<Message>,
    
    #[serde(skip_serializing_if = "Option::is_none")]
    pub options: Option<HashMap<String, serde_json::Value>>,
    
    #[serde(default = "default_stream")]
    pub stream: bool,
    
    /// Tools available for the model to call
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<Tool>>,
}

fn default_stream() -> bool {
    true
}

impl ChatRequest {
    /// Creates a new chat request.
    ///
    /// # Arguments
    ///
    /// * `model` - The model identifier (e.g., "llama2", "mistral")
    /// * `messages` - Conversation history including system, user, and assistant messages
    pub fn new(model: impl Into<String>, messages: Vec<Message>) -> Self {
        Self {
            model: model.into(),
            messages,
            options: None,
            stream: true,
            tools: None,
        }
    }
    
    /// Sets the temperature parameter for response generation.
    ///
    /// Temperature controls randomness in the output. Higher values (e.g., 0.8)
    /// produce more creative responses, while lower values (e.g., 0.2) are more
    /// deterministic and focused.
    pub fn with_temperature(mut self, temperature: f64) -> Self {
        let mut options = self.options.unwrap_or_default();
        options.insert("temperature".to_string(), serde_json::json!(temperature));
        self.options = Some(options);
        self
    }
}

/// Response from the Ollama chat API.
///
/// Responses are streamed, with multiple response objects sent for a single request.
/// The `done` field indicates when the response is complete.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatResponse {
    pub model: String,
    
    #[serde(default)]
    pub created_at: String,
    
    pub message: Message,
    
    #[serde(default)]
    pub done: bool,
    
    #[serde(skip_serializing_if = "Option::is_none")]
    pub done_reason: Option<String>,
}

/// Request for generating embeddings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbedRequest {
    pub model: String,
    pub input: String,
}

/// Response containing embeddings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbedResponse {
    pub model: String,
    
    #[serde(default)]
    pub embeddings: Vec<Vec<f32>>,
}

/// Tool specification for Ollama native tool calling.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tool {
    #[serde(rename = "type")]
    pub tool_type: String,
    pub function: ToolFunction,
}

/// Function definition within a tool.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolFunction {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value,
}

/// Tool call requested by the LLM.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub function: ToolCallFunction,
}

/// Function call details.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallFunction {
    pub name: String,
    pub arguments: serde_json::Value,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_message_constructors() {
        let sys = Message::system("test");
        assert_eq!(sys.role, "system");
        assert_eq!(sys.content, "test");
        
        let user = Message::user("hello");
        assert_eq!(user.role, "user");
    }
    
    #[test]
    fn test_chat_request_builder() {
        let req = ChatRequest::new("llama2", vec![Message::user("test")])
            .with_temperature(0.7);
        
        assert_eq!(req.model, "llama2");
        assert!(req.options.is_some());
    }
}
