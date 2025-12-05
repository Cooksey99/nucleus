//! Helper utilities for embedded Qdrant setup.
//!
//! This module provides utilities to initialize embedded Qdrant with zero external setup.

use anyhow::Result;
use std::path::Path;

/// Ensures the Qdrant storage directory exists.
///
/// Creates the directory if it doesn't exist. This is all that's needed
/// for embedded mode - no server required.
pub fn ensure_storage_dir(path: &str) -> Result<()> {
    let path = Path::new(path);
    if !path.exists() {
        std::fs::create_dir_all(path)?;
    }
    Ok(())
}
