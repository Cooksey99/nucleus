//! MCP (Model Context Protocol) types
//!
//! This module contains types for JSON-RPC 2.0 messages and MCP-specific types.

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// JSON-RPC 2.0 request
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum JsonRpcRequest {
    /// Request with ID (expects response)
    Request {
        jsonrpc: String,
        id: Value,
        method: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        params: Option<Value>,
    },
    /// Notification without ID (no response expected)
    Notification {
        jsonrpc: String,
        method: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        params: Option<Value>,
    },
}

/// JSON-RPC 2.0 response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonRpcResponse {
    pub jsonrpc: String,
    pub id: Value,
    #[serde(flatten)]
    pub result_or_error: ResultOrError,
}

/// JSON-RPC 2.0 result or error
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ResultOrError {
    Success {
        result: Value,
    },
    Error {
        error: JsonRpcError,
    },
}

/// JSON-RPC 2.0 error object
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonRpcError {
    pub code: i32,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<Value>,
}

/// JSON-RPC 2.0 message (can be request, notification, or response)
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum JsonRpcMessage {
    Request(JsonRpcRequest),
    Response(JsonRpcResponse),
}

impl JsonRpcRequest {
    /// Create a new request
    pub fn new(id: Value, method: impl Into<String>, params: Option<Value>) -> Self {
        JsonRpcRequest::Request {
            jsonrpc: "2.0".to_string(),
            id,
            method: method.into(),
            params,
        }
    }

    /// Create a new notification
    pub fn notification(method: impl Into<String>, params: Option<Value>) -> Self {
        JsonRpcRequest::Notification {
            jsonrpc: "2.0".to_string(),
            method: method.into(),
            params,
        }
    }

    /// Get the method name
    pub fn method(&self) -> &str {
        match self {
            JsonRpcRequest::Request { method, .. } => method,
            JsonRpcRequest::Notification { method, .. } => method,
        }
    }
}

impl JsonRpcResponse {
    /// Create a success response
    pub fn success(id: Value, result: Value) -> Self {
        Self {
            jsonrpc: "2.0".to_string(),
            id,
            result_or_error: ResultOrError::Success { result },
        }
    }

    /// Create an error response
    pub fn error(id: Value, error: JsonRpcError) -> Self {
        Self {
            jsonrpc: "2.0".to_string(),
            id,
            result_or_error: ResultOrError::Error { error },
        }
    }
}

impl JsonRpcError {
    /// Parse error
    pub fn parse_error(data: Option<Value>) -> Self {
        Self {
            code: -32700,
            message: "Parse error".to_string(),
            data,
        }
    }

    /// Invalid request error
    pub fn invalid_request(data: Option<Value>) -> Self {
        Self {
            code: -32600,
            message: "Invalid Request".to_string(),
            data,
        }
    }

    /// Method not found error
    pub fn method_not_found(data: Option<Value>) -> Self {
        Self {
            code: -32601,
            message: "Method not found".to_string(),
            data,
        }
    }

    /// Invalid params error
    pub fn invalid_params(data: Option<Value>) -> Self {
        Self {
            code: -32602,
            message: "Invalid params".to_string(),
            data,
        }
    }

    /// Internal error
    pub fn internal_error(data: Option<Value>) -> Self {
        Self {
            code: -32603,
            message: "Internal error".to_string(),
            data,
        }
    }
}

