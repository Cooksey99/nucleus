//! Model Context Protocol (MCP) server implementation.
//!
//! MCP is a protocol for AI assistants to interact with external tools
//! and data sources. This module provides transport implementations for
//! communicating over different channels.

pub mod transports;

pub use transports::stdio::{
    JsonRpcError, JsonRpcRequest, JsonRpcResponse, StdioTransport, StdioTransportError,
};

