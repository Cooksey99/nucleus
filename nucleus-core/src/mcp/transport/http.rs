//! HTTP transport for MCP
//!
//! Handles communication over HTTP using JSON-RPC messages sent via POST requests.
//! Used for remote MCP servers accessible over HTTP.

use crate::mcp::types::{JsonRpcMessage, JsonRpcRequest, JsonRpcResponse};
use anyhow::{Context, Result};
use reqwest::Client;
use serde_json::Value;

/// HTTP transport for MCP communication
pub struct HttpTransport {
    client: Client,
    server_url: String,
    next_id: u64,
    session_id: Option<String>,
}

impl HttpTransport {
    /// Create a new HTTP transport with a server URL
    pub fn new(server_url: impl Into<String>) -> Self {
        Self {
            client: Client::new(),
            server_url: server_url.into(),
            next_id: 1,
            session_id: None,
        }
    }

    /// Create a new HTTP transport with a custom HTTP client
    pub fn with_client(client: Client, server_url: impl Into<String>) -> Self {
        Self {
            client,
            server_url: server_url.into(),
            next_id: 1,
            session_id: None,
        }
    }

    /// Send a JSON-RPC message and receive a response
    ///
    /// Note: HTTP transport only supports request/response pattern.
    /// Notifications are sent but no response is expected.
    pub async fn send(&mut self, message: &JsonRpcMessage) -> Result<Option<JsonRpcResponse>> {
        // Serialize the message
        let json_body = serde_json::to_value(message)
            .context("Failed to serialize JSON-RPC message")?;

        // Send POST request
        // MCP servers typically require Accept header for both application/json and text/event-stream
        let mut request_builder = self
            .client
            .post(&self.server_url)
            .header("Accept", "application/json, text/event-stream")
            .header("Content-Type", "application/json");
        
        // Add session ID if we have one
        if let Some(session_id) = &self.session_id {
            request_builder = request_builder.header("Mcp-Session-Id", session_id);
        }
        
        let response = request_builder
            .json(&json_body)
            .send()
            .await
            .context("Failed to send HTTP request")?;

        // Check if request was successful
        if !response.status().is_success() {
            let status = response.status();
            let text = response.text().await.unwrap_or_default();
            anyhow::bail!(
                "HTTP request failed with status {}: {}",
                status,
                text
            );
        }

        // Handle notifications (no response expected)
        match message {
            JsonRpcMessage::Request(JsonRpcRequest::Notification { .. }) => {
                // Notifications don't expect a response
                return Ok(None);
            }
            JsonRpcMessage::Request(JsonRpcRequest::Request { .. }) => {
                // Requests expect a response
            }
            JsonRpcMessage::Response(_) => {
                // Responses shouldn't be sent via HTTP (they're received)
                anyhow::bail!("Cannot send a response via HTTP transport");
            }
        }

        // Extract session ID from response headers if present
        if let Some(session_id_header) = response.headers().get("mcp-session-id") {
            if let Ok(session_id) = session_id_header.to_str() {
                self.session_id = Some(session_id.to_string());
            }
        }
        
        // Check content type to determine how to parse the response
        let content_type = response
            .headers()
            .get("content-type")
            .and_then(|h| h.to_str().ok())
            .unwrap_or("")
            .to_string();

        // Handle streaming responses (SSE or newline-delimited JSON)
        if content_type.contains("text/event-stream") || content_type.contains("application/x-ndjson") {
            // For streaming, read the response as text and parse the first JSON-RPC message
            let text = response.text().await.context("Failed to read response text")?;
            
            // Note: For debugging, you can uncomment the following lines:
            // eprintln!("DEBUG: Content-Type: {}", content_type);
            // eprintln!("DEBUG: Response preview (first 500 chars): {}", 
            //          if text.len() > 500 { &text[..500] } else { &text });
            
            // For SSE, extract data lines (format: "data: {...}\n")
            // For newline-delimited JSON, each line is a JSON object
            let json_text = if text.contains("data:") {
                // SSE format - extract JSON from "data: {...}" lines
                text.lines()
                    .find_map(|line| {
                        let trimmed = line.trim();
                        if trimmed.starts_with("data:") {
                            let data = trimmed.strip_prefix("data:").unwrap_or("").trim();
                            if !data.is_empty() && data != "[DONE]" {
                                Some(data)
                            } else {
                                None
                            }
                        } else {
                            None
                        }
                    })
                    .unwrap_or("")
            } else {
                // Newline-delimited JSON - take the first non-empty line
                text.lines()
                    .find(|line| !line.trim().is_empty())
                    .unwrap_or("")
            };

            if json_text.is_empty() {
                anyhow::bail!("No JSON data found in streaming response. Raw response: {}", 
                             if text.len() > 200 { &text[..200] } else { &text });
            }

            let response_json: Value = serde_json::from_str(json_text)
                .with_context(|| format!("Failed to parse JSON from streaming response. JSON text: {}", json_text))?;

            let jsonrpc_response: JsonRpcResponse = serde_json::from_value(response_json)
                .context("Failed to deserialize JSON-RPC response")?;

            Ok(Some(jsonrpc_response))
        } else {
            // Standard JSON response
            let response_json: Value = response
                .json()
                .await
                .context("Failed to parse HTTP response as JSON")?;

            let jsonrpc_response: JsonRpcResponse = serde_json::from_value(response_json)
                .context("Failed to deserialize JSON-RPC response")?;

            Ok(Some(jsonrpc_response))
        }
    }

    /// Send a request and wait for a response
    ///
    /// This is a convenience method that wraps send() and extracts the result.
    pub async fn request(
        &mut self,
        method: impl Into<String>,
        params: Option<Value>,
    ) -> Result<Value> {
        let id = Value::Number(serde_json::Number::from(self.next_id));
        self.next_id += 1;
        
        let request = JsonRpcRequest::new(id.clone(), method, params);
        let message = JsonRpcMessage::Request(request);

        let response = self.send(&message).await?;

        match response {
            Some(resp) => {
                // Verify the response ID matches
                if resp.id != id {
                    anyhow::bail!(
                        "Response ID mismatch: expected {:?}, got {:?}",
                        id,
                        resp.id
                    );
                }
                
                match resp.result_or_error {
                    crate::mcp::types::ResultOrError::Success { result } => Ok(result),
                    crate::mcp::types::ResultOrError::Error { error } => {
                        anyhow::bail!("JSON-RPC error: {} (code: {})", error.message, error.code)
                    }
                }
            }
            None => {
                anyhow::bail!("Expected response but received none")
            }
        }
    }

    /// Send a notification (no response expected)
    pub async fn notify(&mut self, method: impl Into<String>, params: Option<Value>) -> Result<()> {
        let notification = JsonRpcRequest::notification(method, params);
        let message = JsonRpcMessage::Request(notification);
        self.send(&message).await?;
        Ok(())
    }

    /// Get the server URL
    pub fn server_url(&self) -> &str {
        &self.server_url
    }
}

