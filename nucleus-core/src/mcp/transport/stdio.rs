//! Stdio transport for MCP
//!
//! Handles communication over stdin/stdout using newline-delimited JSON-RPC messages.
//! used for local MCP servers

use crate::mcp::types::JsonRpcMessage;
use anyhow::{Context, Result};
use serde_json;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader as TokioBufReader};
use tokio::process::{Child, Command};

/// Stdio transport for MCP communication
pub struct StdioTransport {
    stdin: tokio::process::ChildStdin,
    stdout: TokioBufReader<tokio::process::ChildStdout>,
    child: Child,
}

impl StdioTransport {
    /// Create a new stdio transport by spawning an MCP server process
    pub fn spawn(command: &str, args: &[&str]) -> Result<Self> {
        let mut child = Command::new(command)
            .args(args)
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .kill_on_drop(true)
            .spawn()
            .context("Failed to spawn MCP server process")?;

        let stdin = child
            .stdin
            .take()
            .context("Failed to get stdin handle")?;
        let stdout = child
            .stdout
            .take()
            .context("Failed to get stdout handle")?;

        Ok(Self {
            stdin,
            stdout: TokioBufReader::new(stdout),
            child,
        })
    }

    /// Send a JSON-RPC message
    pub async fn send(&mut self, message: &JsonRpcMessage) -> Result<()> {
        let json = serde_json::to_string(message)
            .context("Failed to serialize JSON-RPC message")?;
        
        self.stdin
            .write_all(json.as_bytes())
            .await
            .context("Failed to write to stdin")?;
        self.stdin
            .write_all(b"\n")
            .await
            .context("Failed to write newline")?;
        self.stdin
            .flush()
            .await
            .context("Failed to flush stdin")?;

        Ok(())
    }

    /// Receive a JSON-RPC message
    pub async fn receive(&mut self) -> Result<JsonRpcMessage> {
        let mut line = String::new();
        self.stdout
            .read_line(&mut line)
            .await
            .context("Failed to read from stdout")?;

        if line.is_empty() {
            anyhow::bail!("EOF reached");
        }

        // Trim newline
        let line = line.trim_end();

        let message: JsonRpcMessage = serde_json::from_str(line)
            .context("Failed to parse JSON-RPC message")?;

        Ok(message)
    }

    /// Check if the child process is still running
    pub fn is_alive(&mut self) -> bool {
        self.child.try_wait().map(|s| s.is_none()).unwrap_or(false)
    }

    /// Wait for the child process to exit
    pub async fn wait(&mut self) -> Result<std::process::ExitStatus> {
        self.child
            .wait()
            .await
            .context("Failed to wait for child process")
    }
}

impl Drop for StdioTransport {
    fn drop(&mut self) {
        let _ = self.child.wait();
    }
}

