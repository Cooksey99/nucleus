//! MCP transport implementations.
//!
//! This module contains different transport mechanisms for MCP communication:
//! - `stdio`: Standard input/output transport (JSON-RPC over stdio)
//! - `http`: HTTP transport (for future implementation)

pub mod stdio;

#[cfg(feature = "http")]
pub mod http;

