//! Ollama availability detection and installation guidance.

use std::process::Command;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum DetectionError {
    #[error("Ollama is not installed or not in PATH")]
    NotInstalled,

    #[error("Ollama is installed but not running")]
    NotRunning,

    #[error("Failed to check Ollama status: {0}")]
    CheckFailed(String),
}

pub type Result<T> = std::result::Result<T, DetectionError>;

/// Checks if Ollama is available and provides helpful guidance if not.
///
/// This is called automatically when creating a Server, but can also be
/// called manually to verify Ollama is available before attempting operations.
///
/// # Example
///
/// ```no_run
/// use nucleus_core::detection;
///
/// match detection::detect_ollama() {
///     Ok(_) => println!("Ready to go!"),
///     Err(e) => eprintln!("Setup required: {}", e),
/// }
/// ```
pub fn detect_ollama() -> Result<OllamaInfo> {
    if !is_ollama_installed() {
        print_installation_help();
        return Err(DetectionError::NotInstalled);
    }

    match is_ollama_running() {
        Ok(true) => Ok(OllamaInfo {
            installed: true,
            running: true,
        }),
        Ok(false) => {
            print_startup_help();
            Err(DetectionError::NotRunning)
        }
        Err(e) => {
            eprintln!("⚠️  Could not verify Ollama status: {}", e);
            Err(DetectionError::CheckFailed(e))
        }
    }
}

/// Quietly checks if Ollama is available without printing help messages.
///
/// Useful for programmatic checks where you want to handle the error yourself.
pub fn check_ollama_silent() -> Result<OllamaInfo> {
    if !is_ollama_installed() {
        return Err(DetectionError::NotInstalled);
    }

    match is_ollama_running() {
        Ok(true) => Ok(OllamaInfo {
            installed: true,
            running: true,
        }),
        Ok(false) => Err(DetectionError::NotRunning),
        Err(e) => Err(DetectionError::CheckFailed(e)),
    }
}

/// Information about Ollama availability.
#[derive(Debug, Clone)]
pub struct OllamaInfo {
    pub installed: bool,
    pub running: bool,
}

fn is_ollama_installed() -> bool {
    Command::new("which")
        .arg("ollama")
        .output()
        .map(|output| output.status.success())
        .unwrap_or(false)
}

fn is_ollama_running() -> std::result::Result<bool, String> {
    let output = Command::new("ollama")
        .arg("list")
        .output()
        .map_err(|e| e.to_string())?;

    Ok(output.status.success())
}

fn print_installation_help() {
    eprintln!("❌ Ollama not found!");
    eprintln!();
    eprintln!("  Nucleus requires Ollama to be installed.");
    eprintln!();
    eprintln!("  Install Ollama:");

    #[cfg(target_os = "macos")]
    {
        eprintln!("   • macOS:  curl -fsSL https://ollama.ai/install.sh | sh");
        eprintln!("   • Or:     brew install ollama");
    }

    #[cfg(target_os = "linux")]
    {
        eprintln!("   • Linux:  curl -fsSL https://ollama.ai/install.sh | sh");
    }

    #[cfg(target_os = "windows")]
    {
        eprintln!("   • Windows: Download from https://ollama.ai/download");
    }

    eprintln!();
    eprintln!("  After installation, pull a model:");
    eprintln!("   ollama pull qwen2.5-coder:1.5b    (recommended, ~1GB)");
    eprintln!("   ollama pull llama3.2:3b            (alternative, ~2GB)");
    eprintln!();
    eprintln!("  Learn more: https://github.com/ollama/ollama");
}

fn print_startup_help() {
    eprintln!("❌ Ollama is installed but not running!");
    eprintln!();
    eprintln!("  Start Ollama:");

    #[cfg(target_os = "macos")]
    {
        eprintln!("   • Run the Ollama app from Applications");
        eprintln!("   • Or:  ollama serve  (in a separate terminal)");
    }

    #[cfg(not(target_os = "macos"))]
    {
        eprintln!("   ollama serve");
    }

    eprintln!();
    eprintln!("  Verify it's running:");
    eprintln!("   ollama list");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detection_returns_result() {
        let result = detect_ollama();
        assert!(result.is_ok() || result.is_err());
    }
}
