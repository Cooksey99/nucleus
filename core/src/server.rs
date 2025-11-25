//! Unix socket server for handling AI requests.
//!
//! Listens on a Unix socket and handles various request types including
//! chat, RAG operations, and system commands.

use crate::config::Config;
use crate::ollama;
use serde::{Deserialize, Serialize};
use std::path::Path;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::{UnixListener, UnixStream};
use tokio::signal;
use thiserror::Error;

const SOCKET_PATH: &str = "/tmp/llm-workspace.sock";

#[derive(Debug, Error)]
pub enum ServerError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
    
    #[error("Ollama error: {0}")]
    Ollama(#[from] ollama::OllamaError),
}

pub type Result<T> = std::result::Result<T, ServerError>;

/// Request from client to server.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Request {
    #[serde(rename = "type")]
    pub request_type: String,
    pub content: String,
    
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pwd: Option<String>,
    
    #[serde(skip_serializing_if = "Option::is_none")]
    pub history: Option<Vec<ollama::Message>>,
}

/// Streaming response chunk.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamChunk {
    #[serde(rename = "type")]
    pub chunk_type: String,
    pub content: String,
    
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

impl StreamChunk {
    pub fn chunk(content: impl Into<String>) -> Self {
        Self {
            chunk_type: "chunk".to_string(),
            content: content.into(),
            error: None,
        }
    }
    
    pub fn done(content: impl Into<String>) -> Self {
        Self {
            chunk_type: "done".to_string(),
            content: content.into(),
            error: None,
        }
    }
    
    pub fn error(error: impl Into<String>) -> Self {
        Self {
            chunk_type: "error".to_string(),
            content: String::new(),
            error: Some(error.into()),
        }
    }
}

/// Main server managing Unix socket connections.
pub struct Server {
    config: Config,
    ollama_client: ollama::Client,
}

impl Server {
    /// Creates a new server instance.
    pub fn new(config: Config) -> Self {
        let ollama_client = ollama::Client::new(&config.llm.base_url);
        
        Self {
            config,
            ollama_client,
        }
    }
    
    /// Starts the server and listens for connections.
    pub async fn start(&self) -> Result<()> {
        if Path::new(SOCKET_PATH).exists() {
            std::fs::remove_file(SOCKET_PATH)?;
        }
        
        let listener = UnixListener::bind(SOCKET_PATH)?;
        
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let perms = std::fs::Permissions::from_mode(0o600);
            std::fs::set_permissions(SOCKET_PATH, perms)?;
        }
        
        println!("AI Server listening on {}", SOCKET_PATH);
        println!("Model: {}", self.config.llm.model);
        
        let shutdown = signal::ctrl_c();
        tokio::pin!(shutdown);
        
        loop {
            tokio::select! {
                Ok((stream, _)) = listener.accept() => {
                    tokio::spawn(Self::handle_connection(
                        stream,
                        self.config.clone(),
                        self.ollama_client.clone(),
                    ));
                }
                _ = &mut shutdown => {
                    println!("\nShutting down...");
                    drop(listener);
                    let _ = std::fs::remove_file(SOCKET_PATH);
                    break;
                }
            }
        }
        
        Ok(())
    }
    
    /// Handles a single client connection.
    async fn handle_connection(
        stream: UnixStream,
        config: Config,
        ollama_client: ollama::Client,
    ) {
        if let Err(e) = Self::handle_connection_impl(stream, config, ollama_client).await {
            eprintln!("Connection error: {}", e);
        }
    }
    
    async fn handle_connection_impl(
        mut stream: UnixStream,
        config: Config,
        ollama_client: ollama::Client,
    ) -> Result<()> {
        let (reader, mut writer) = stream.split();
        let mut reader = BufReader::new(reader);
        let mut line = String::new();
        
        reader.read_line(&mut line).await?;
        let request: Request = serde_json::from_str(&line)?;
        
        match request.request_type.as_str() {
            "chat" | "edit" => {
                Self::handle_chat(&mut writer, request, config, ollama_client).await?;
            }
            "add" => {
                let chunk = StreamChunk::done("Added to knowledge base (RAG not implemented yet)");
                let json = serde_json::to_string(&chunk)?;
                writer.write_all(json.as_bytes()).await?;
                writer.write_all(b"\n").await?;
            }
            "index" => {
                let chunk = StreamChunk::done(format!(
                    "Indexed directory: {} (RAG not implemented yet)",
                    request.content
                ));
                let json = serde_json::to_string(&chunk)?;
                writer.write_all(json.as_bytes()).await?;
                writer.write_all(b"\n").await?;
            }
            "stats" => {
                let chunk = StreamChunk::done("Knowledge base: 0 documents (RAG not implemented yet)");
                let json = serde_json::to_string(&chunk)?;
                writer.write_all(json.as_bytes()).await?;
                writer.write_all(b"\n").await?;
            }
            _ => {
                let chunk = StreamChunk::error("Unknown request type");
                let json = serde_json::to_string(&chunk)?;
                writer.write_all(json.as_bytes()).await?;
                writer.write_all(b"\n").await?;
            }
        }
        
        Ok(())
    }
    
    async fn handle_chat(
        writer: &mut tokio::net::unix::WriteHalf<'_>,
        request: Request,
        config: Config,
        ollama_client: ollama::Client,
    ) -> Result<()> {
        let mut messages = vec![
            ollama::Message::system(&config.system_prompt),
        ];
        
        if let Some(history) = request.history {
            for msg in history {
                messages.push(ollama::Message {
                    role: msg.role,
                    content: msg.content,
                });
            }
        }
        
        messages.push(ollama::Message::user(&request.content));
        
        let chat_request = ollama::ChatRequest::new(&config.llm.model, messages)
            .with_temperature(config.llm.temperature);
        
        let mut full_response = String::new();
        
        let result = ollama_client.chat(chat_request, |response| {
            if !response.message.content.is_empty() {
                full_response.push_str(&response.message.content);
                
                let chunk = StreamChunk::chunk(&response.message.content);
                if let Ok(json) = serde_json::to_string(&chunk) {
                    let _ = futures::executor::block_on(async {
                        writer.write_all(json.as_bytes()).await?;
                        writer.write_all(b"\n").await?;
                        writer.flush().await
                    });
                }
            }
        }).await;
        
        match result {
            Ok(_) => {
                let chunk = StreamChunk::done(&full_response);
                let json = serde_json::to_string(&chunk)?;
                writer.write_all(json.as_bytes()).await?;
                writer.write_all(b"\n").await?;
            }
            Err(e) => {
                let chunk = StreamChunk::error(e.to_string());
                let json = serde_json::to_string(&chunk)?;
                writer.write_all(json.as_bytes()).await?;
                writer.write_all(b"\n").await?;
            }
        }
        
        Ok(())
    }
}
