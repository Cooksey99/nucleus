//! Helper utilities for managing Qdrant lifecycle during development.
//!
//! This module provides utilities to check if Qdrant is running and provide
//! helpful error messages if it's not available.

use std::process::Command;
use anyhow::{Context, Result};

/// Checks if Qdrant is running at the given URL.
///
/// Returns `Ok(())` if Qdrant is accessible, otherwise returns an error
/// with helpful setup instructions.
///
/// Note: This checks the HTTP REST API on port 6333, not the gRPC port.
pub async fn ensure_qdrant_running(url: &str) -> Result<()> {
    // Qdrant gRPC is on 6334, but we check HTTP REST API on 6333
    let http_url = url.replace(":6334", ":6333");
    
    // Try to connect to Qdrant
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(2))
        .build()?;
    
    match client.get(&http_url).send().await {
        Ok(_) => Ok(()),
        Err(_) => {
            Err(anyhow::anyhow!(
                r#"
❌ Cannot connect to Qdrant at {}

Qdrant is required for RAG (vector database).

To start Qdrant:

1. Download Qdrant for Windows:
   https://github.com/qdrant/qdrant/releases/latest/download/qdrant-x86_64-pc-windows-msvc.zip

2. Extract and run:
   qdrant.exe

3. Or use Docker:
   docker run -p 6334:6334 qdrant/qdrant

Once Qdrant is running, try again.
"#,
                url
            ))
        }
    }
}

/// Attempts to start Qdrant if a known executable path is provided.
///
/// This is useful for development but not recommended for production.
/// Returns `Ok(())` if Qdrant was started or is already running.
#[cfg(target_os = "windows")]
pub fn try_start_qdrant(qdrant_path: &str) -> Result<()> {
    use std::path::Path;
    
    if !Path::new(qdrant_path).exists() {
        return Err(anyhow::anyhow!(
            "Qdrant executable not found at: {}\nDownload from: https://github.com/qdrant/qdrant/releases/latest",
            qdrant_path
        ));
    }
    
    println!("🚀 Starting Qdrant...");
    
    Command::new(qdrant_path)
        .current_dir(Path::new(qdrant_path).parent().unwrap())
        .spawn()
        .context("Failed to start Qdrant")?;
    
    println!("✓ Qdrant started in background");
    
    // Give it a moment to start up
    std::thread::sleep(std::time::Duration::from_secs(2));
    
    Ok(())
}

#[cfg(not(target_os = "windows"))]
pub fn try_start_qdrant(_qdrant_path: &str) -> Result<()> {
    Err(anyhow::anyhow!("Auto-start only supported on Windows. Please start Qdrant manually."))
}
