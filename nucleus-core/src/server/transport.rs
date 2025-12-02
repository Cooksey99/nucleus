use super::types::{Request, StreamChunk};
use thiserror::Error;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::sync::mpsc;

#[cfg(unix)]
use tokio::net::{UnixListener, UnixStream};
#[cfg(unix)]
use std::path::Path;

#[cfg(windows)]
use tokio::net::windows::named_pipe::{NamedPipeServer, ServerOptions};

#[derive(Debug, Error)]
pub enum TransportError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
}

pub type Result<T> = std::result::Result<T, TransportError>;

// Type aliases for platform-specific types
#[cfg(unix)]
pub type IpcStream = UnixStream;

#[cfg(windows)]
pub type IpcStream = NamedPipeServer;

#[cfg(unix)]
pub type IpcListener = UnixListener;

#[cfg(windows)]
pub struct WindowsPipeListener {
    path: String,
}

#[cfg(windows)]
impl WindowsPipeListener {
    pub async fn accept(&self) -> Result<(NamedPipeServer, ())> {
        let server = ServerOptions::new()
            .first_pipe_instance(false)
            .create(&self.path)?;
        
        server.connect().await?;
        
        Ok((server, ()))
    }
}

#[cfg(windows)]
pub type IpcListener = WindowsPipeListener;

/// IPC transport for communication (Unix sockets on Unix, Named Pipes on Windows).
pub struct IpcTransport {
    socket_path: String,
}

impl IpcTransport {
    pub fn new(socket_path: impl Into<String>) -> Self {
        Self {
            socket_path: socket_path.into(),
        }
    }
    
    /// Binds to the IPC endpoint and returns a listener.
    #[cfg(unix)]
    pub async fn bind(&self) -> Result<IpcListener> {
        if Path::new(&self.socket_path).exists() {
            std::fs::remove_file(&self.socket_path)?;
        }
        
        let listener = UnixListener::bind(&self.socket_path)?;
        
        use std::os::unix::fs::PermissionsExt;
        let perms = std::fs::Permissions::from_mode(0o600);
        std::fs::set_permissions(&self.socket_path, perms)?;
        
        Ok(listener)
    }
    
    #[cfg(windows)]
    pub async fn bind(&self) -> Result<IpcListener> {
        // Create the first instance of the named pipe
        let server = ServerOptions::new()
            .first_pipe_instance(true)
            .create(&self.socket_path)?;
        
        Ok(WindowsPipeListener {
            path: self.socket_path.clone(),
        })
    }
    
    /// Cleans up the IPC endpoint.
    #[cfg(unix)]
    pub fn cleanup(&self) {
        let _ = std::fs::remove_file(&self.socket_path);
    }
    
    #[cfg(windows)]
    pub fn cleanup(&self) {
        // Named pipes are automatically cleaned up on Windows
    }
}

/// Reads a request from the stream.
#[cfg(unix)]
pub async fn read_request(stream: &mut IpcStream) -> Result<Request> {
    let (reader, _) = stream.split();
    let mut reader = BufReader::new(reader);
    let mut line = String::new();
    
    reader.read_line(&mut line).await?;
    let request = serde_json::from_str(&line)?;
    
    Ok(request)
}

#[cfg(windows)]
pub async fn read_request(stream: &mut IpcStream) -> Result<Request> {
    let mut reader = BufReader::new(stream);
    let mut line = String::new();
    
    reader.read_line(&mut line).await?;
    let request = serde_json::from_str(&line)?;
    
    Ok(request)
}

/// Writes stream chunks to the client.
pub async fn write_chunks(
    stream: &mut IpcStream,
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
