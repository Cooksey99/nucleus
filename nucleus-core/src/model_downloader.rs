

//! Auto-download functionality for default embedding models.
//!
//! This module handles automatic downloading and caching of the default
//! embedding model when it's not already present locally.

use std::path::{Path, PathBuf};
use std::fs;
use std::io::Write;
use thiserror::Error;
use tracing::info;

#[derive(Debug, Error)]
pub enum DownloadError {
    #[error("HTTP request failed: {0}")]
    Request(#[from] reqwest::Error),
    
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    
    #[error("Checksum mismatch: expected {expected}, got {actual}")]
    ChecksumMismatch { expected: String, actual: String },
}

pub type Result<T> = std::result::Result<T, DownloadError>;

// Default model metadata - synced with config.rs default
const DEFAULT_HF_MODEL: &str = "Qwen/Qwen3-Embedding-0.6B-GGUF";
const DEFAULT_GGUF_FILE: &str = "Qwen3-Embedding-0.6B-Q8_0.gguf";
const DEFAULT_MODEL_DIR: &str = "qwen3-embedding-0.6b-gguf";
const RELEASE_TAG: &str = "models-v1.0.0";
const GITHUB_REPO: &str = "Cooksey99/nucleus";
const MODEL_ARCHIVE: &str = "qwen3-embedding-0.6b-gguf.tar.gz";
const EXPECTED_CHECKSUM: &str = "40ada098b5b7a2159b0f08b5a513b396a75eac1584fb0adc0a8dd3944d061e39";

/// Returns the models directory (project-local).
pub fn get_models_dir() -> PathBuf {
    PathBuf::from("models")
}

/// Returns the path to the default embedding model.
pub fn get_default_model_path() -> PathBuf {
    get_models_dir().join(DEFAULT_MODEL_DIR)
}

/// Checks if the default model is already installed.
pub fn is_default_model_installed() -> bool {
    let model_path = get_default_model_path();
    model_path.exists() && model_path.join(DEFAULT_GGUF_FILE).exists()
}

/// Resolves a model path, handling auto-download for the default model.
///
/// This function:
/// 1. Detects if the model string matches the default HuggingFace model
/// 2. Checks if it exists in models/ directory, downloads if needed
/// 3. Otherwise returns the path as-is (for local files or other HF models)
pub async fn resolve_model_path(model_name: &str) -> Result<String> {
    // Check if this is the default model in HuggingFace format
    let default_hf_format = format!("{}:{}", DEFAULT_HF_MODEL, DEFAULT_GGUF_FILE);
    
    if model_name == default_hf_format {
        // Use models/ directory (auto-downloads if not present)
        let path = ensure_default_model().await?;
        info!("Using default model from: {}", path.display());
        Ok(format!("{}:{}", path.display(), DEFAULT_GGUF_FILE))
    } else if Path::new(model_name).exists() {
        // Use the path as-is if it exists locally
        Ok(model_name.to_string())
    } else {
        // Return as-is for other HuggingFace models or formats
        Ok(model_name.to_string())
    }
}

/// Downloads and installs the default embedding model if not already present.
///
/// This function will:
/// 1. Check if the model is already installed
/// 2. Download the model archive from GitHub releases
/// 3. Verify the checksum
/// 4. Extract to the cache directory
///
/// Returns the path to the installed model.
pub async fn ensure_default_model() -> Result<PathBuf> {
    let model_path = get_default_model_path();
    
    if is_default_model_installed() {
        info!("Default embedding model already installed at: {}", model_path.display());
        return Ok(model_path);
    }
    
    info!("Default embedding model not found. Downloading...");
    download_and_install_model().await?;
    
    Ok(model_path)
}

async fn download_and_install_model() -> Result<()> {
    let models_dir = get_models_dir();
    fs::create_dir_all(&models_dir)?;
    
    let download_url = format!(
        "https://github.com/{}/releases/download/{}/{}",
        GITHUB_REPO, RELEASE_TAG, MODEL_ARCHIVE
    );
    
    info!("Downloading model from: {}", download_url);
    info!("This may take a few minutes (582MB download)...");
    
    let client = reqwest::Client::new();
    let response = client.get(&download_url).send().await?;
    
    if !response.status().is_success() {
        return Err(DownloadError::Request(
            reqwest::Error::from(response.error_for_status().unwrap_err())
        ));
    }
    
    let bytes = response.bytes().await?;
    
    info!("Download complete. Verifying checksum...");
    let actual_checksum = sha256_digest(&bytes);
    if actual_checksum != EXPECTED_CHECKSUM {
        return Err(DownloadError::ChecksumMismatch {
            expected: EXPECTED_CHECKSUM.to_string(),
            actual: actual_checksum,
        });
    }
    
    info!("Checksum verified. Extracting model...");
    let archive_path = models_dir.join(MODEL_ARCHIVE);
    let mut file = fs::File::create(&archive_path)?;
    file.write_all(&bytes)?;
    
    extract_tarball(&archive_path, &models_dir.join(DEFAULT_MODEL_DIR))?;
    
    fs::remove_file(&archive_path)?;
    
    info!("Model installed successfully at: {}", models_dir.join(DEFAULT_MODEL_DIR).display());
    
    Ok(())
}

fn extract_tarball(archive_path: &Path, destination: &Path) -> Result<()> {
    use std::process::Command;
    
    fs::create_dir_all(destination)?;
    
    let status = Command::new("tar")
        .arg("-xzf")
        .arg(archive_path)
        .arg("-C")
        .arg(destination)
        .status()?;
    
    if !status.success() {
        return Err(DownloadError::Io(std::io::Error::new(
            std::io::ErrorKind::Other,
            "Failed to extract tarball"
        )));
    }
    
    Ok(())
}

fn sha256_digest(data: &[u8]) -> String {
    use std::process::{Command, Stdio};
    use std::io::Write as _;
    
    let mut child = Command::new("shasum")
        .arg("-a")
        .arg("256")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .expect("Failed to spawn shasum");
    
    if let Some(mut stdin) = child.stdin.take() {
        stdin.write_all(data).expect("Failed to write to stdin");
    }
    
    let output = child.wait_with_output().expect("Failed to read shasum output");
    let output_str = String::from_utf8_lossy(&output.stdout);
    
    output_str.split_whitespace().next().unwrap_or("").to_string()
}
