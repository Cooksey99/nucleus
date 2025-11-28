//! Unix socket server for handling AI requests.
//!
//! The server is organized into separate concerns:
//! - `types`: Protocol types for requests and responses
//! - `handler`: Business logic for processing requests
//! - `transport`: Unix socket communication layer

mod handler;
mod transport;
mod types;

// Re-export types for external use
#[allow(unused)]
pub use types::{ChunkType, Message, Request, RequestType, StreamChunk};

use crate::{config::Config, detection, ollama};
use std::sync::Arc;
use tokio::signal;
use tokio::sync::mpsc;

const SOCKET_PATH: &str = "/tmp/llm-workspace.sock";

/// Main server coordinating transport and request handling.
pub struct Server {
    handler: Arc<handler::RequestHandler>,
    transport: transport::UnixSocketTransport,
}

impl Server {
    /// Creates a new server instance.
    /// 
    /// This will check if Ollama is installed and running.
    /// If not, helpful installation/startup instructions will be printed.
    pub fn new(config: Config) -> Result<Self, detection::DetectionError> {
        detection::detect_ollama()?;
        
        let ollama_client = ollama::Client::new(&config.llm.base_url);
        let handler = Arc::new(handler::RequestHandler::new(config, ollama_client));
        let transport = transport::UnixSocketTransport::new(SOCKET_PATH);
        
        Ok(Self { handler, transport })
    }
    
    /// Starts the server and listens for connections.
    pub async fn start(&self) -> Result<(), Box<dyn std::error::Error>> {
        let listener = self.transport.bind().await?;
        
        println!("AI Server listening on {}", SOCKET_PATH);
        
        let shutdown = signal::ctrl_c();
        tokio::pin!(shutdown);
        
        loop {
            tokio::select! {
                Ok((stream, _)) = listener.accept() => {
                    let handler = Arc::clone(&self.handler);
                    tokio::spawn(async move {
                        if let Err(e) = handle_connection(stream, handler).await {
                            eprintln!("Connection error: {}", e);
                        }
                    });
                }
                _ = &mut shutdown => {
                    println!("\nShutting down...");
                    self.transport.cleanup();
                    break;
                }
            }
        }
        
        Ok(())
    }
}

/// Handles a single client connection.
async fn handle_connection(
    mut stream: tokio::net::UnixStream,
    handler: Arc<handler::RequestHandler>,
) -> Result<(), Box<dyn std::error::Error>> {
    let request = transport::read_request(&mut stream).await?;
    
    let (sender, receiver) = mpsc::unbounded_channel();
    
    let handle_task = tokio::spawn(async move {
        handler.handle(request, sender).await;
    });
    
    let write_task = tokio::spawn(async move {
        transport::write_chunks(&mut stream, receiver).await
    });
    
    let _ = tokio::try_join!(handle_task, write_task)?;
    
    Ok(())
}
