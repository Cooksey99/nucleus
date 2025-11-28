use super::types::{Request, StreamChunk};
use std::path::Path;
use thiserror::Error;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::{UnixListener, UnixStream};
use tokio::sync::mpsc;

#[derive(Debug, Error)]
pub enum TransportError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
}

pub type Result<T> = std::result::Result<T, TransportError>;

/// Unix socket transport for IPC communication.
pub struct UnixSocketTransport {
    socket_path: String,
}

impl UnixSocketTransport {
    pub fn new(socket_path: impl Into<String>) -> Self {
        Self {
            socket_path: socket_path.into(),
        }
    }
    
    /// Binds to the Unix socket and returns a listener.
    pub async fn bind(&self) -> Result<UnixListener> {
        if Path::new(&self.socket_path).exists() {
            std::fs::remove_file(&self.socket_path)?;
        }
        
        let listener = UnixListener::bind(&self.socket_path)?;
        
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let perms = std::fs::Permissions::from_mode(0o600);
            std::fs::set_permissions(&self.socket_path, perms)?;
        }
        
        Ok(listener)
    }
    
    /// Cleans up the socket file.
    pub fn cleanup(&self) {
        let _ = std::fs::remove_file(&self.socket_path);
    }
}

/// Reads a request from the stream.
pub async fn read_request(stream: &mut UnixStream) -> Result<Request> {
    let (reader, _) = stream.split();
    let mut reader = BufReader::new(reader);
    let mut line = String::new();
    
    reader.read_line(&mut line).await?;
    let request = serde_json::from_str(&line)?;
    
    Ok(request)
}

/// Writes stream chunks to the client.
pub async fn write_chunks(
    stream: &mut UnixStream,
    mut receiver: mpsc::UnboundedReceiver<StreamChunk>,
) -> Result<()> {
    while let Some(chunk) = receiver.recv().await {
        let json = serde_json::to_string(&chunk)?;
        stream.write_all(json.as_bytes()).await?;
        stream.write_all(b"\n").await?;
        stream.flush().await?;
    }
    
    Ok(())
}
