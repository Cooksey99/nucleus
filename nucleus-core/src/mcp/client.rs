//! MCP Client
//!
//! A client for communicating with MCP servers using various transports.

use crate::mcp::transport::stdio::StdioTransport;
use crate::mcp::types::{JsonRpcMessage, JsonRpcRequest};
use anyhow::{Context, Result};
use serde_json::Value;

/// MCP Client
pub struct McpClient {
    transport: StdioTransport,
    next_id: u64,
}

impl McpClient {
    /// Create a new MCP client with stdio transport
    pub fn new_stdio(command: &str, args: &[&str]) -> Result<Self> {
        let transport = StdioTransport::spawn(command, args)
            .context("Failed to create stdio transport")?;

        Ok(Self {
            transport,
            next_id: 1,
        })
    }

    /// Send a request and wait for a response
    pub async fn request(
        &mut self,
        method: impl Into<String>,
        params: Option<Value>,
    ) -> Result<Value> {
        let id = Value::Number(serde_json::Number::from(self.next_id));
        self.next_id += 1;

        let request = JsonRpcRequest::new(id.clone(), method, params);
        let message = JsonRpcMessage::Request(request);

        // Send the request
        self.transport.send(&message).await?;

        // Wait for the response, matching by ID
        loop {
            let response = self.transport.receive().await?;

            match response {
                JsonRpcMessage::Response(resp) => {
                    // Check if this is the response we're waiting for
                    if resp.id == id {
                        match resp.result_or_error {
                            crate::mcp::types::ResultOrError::Success { result } => return Ok(result),
                            crate::mcp::types::ResultOrError::Error { error } => {
                                anyhow::bail!("JSON-RPC error: {} (code: {})", error.message, error.code)
                            }
                        }
                    } else {
                        // This is a response for a different request
                        // In a full implementation, we'd route it to the correct handler
                        // For now, we'll continue waiting
                        continue;
                    }
                }
                JsonRpcMessage::Request(_) => {
                    // Received a request from the server (server-initiated)
                    // In a full implementation, we'd handle this
                    // For now, we'll continue waiting for our response
                    continue;
                }
            }
        }
    }

    /// Send a notification (no response expected)
    pub async fn notify(&mut self, method: impl Into<String>, params: Option<Value>) -> Result<()> {
        let notification = JsonRpcRequest::notification(method, params);
        let message = JsonRpcMessage::Request(notification);
        self.transport.send(&message).await?;
        Ok(())
    }

    /// Receive a message (for handling server-initiated requests/notifications)
    pub async fn receive(&mut self) -> Result<JsonRpcMessage> {
        self.transport.receive().await
    }

    /// Check if the transport is still alive
    pub fn is_alive(&mut self) -> bool {
        self.transport.is_alive()
    }
}

