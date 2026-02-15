//! MCP (Model Context Protocol) client implementation
//!
//! This module provides a client for communicating with MCP servers
//! using various transports (stdio, HTTP, WebSocket).

pub mod client;
pub mod transport;
pub mod types;

pub use client::McpClient;
pub use transport::{http::HttpTransport, stdio::StdioTransport};
pub use types::{JsonRpcError, JsonRpcMessage, JsonRpcRequest, JsonRpcResponse};

