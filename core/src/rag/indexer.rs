//! File indexing and text chunking for RAG.
//!
//! This module provides functionality to:
//! - Recursively collect code files from directories
//! - Split large text into overlapping chunks
//! - Filter files by extension

use std::path::{Path, PathBuf};
use tokio::fs;
use thiserror::Error;

/// Errors that can occur during file indexing.
#[derive(Debug, Error)]
pub enum IndexerError {
    /// An I/O error occurred while reading files or directories.
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

/// Result type for indexing operations.
pub type Result<T> = std::result::Result<T, IndexerError>;

/// Splits text into overlapping chunks for better context preservation.
///
/// Text chunking is essential for RAG because:
/// - LLMs have context limits, so long documents must be split
/// - Overlapping chunks preserve context across boundaries
/// - Smaller chunks produce more focused embeddings
///
/// # Algorithm
///
/// 1. If text fits in chunk_size, return as single chunk
/// 2. Otherwise, slide a window of size `chunk_size` across the text
/// 3. Move window by `(chunk_size - overlap)` each step
/// 4. Last chunk may be smaller than chunk_size
///
/// # Arguments
///
/// * `text` - The text to split into chunks
/// * `chunk_size` - Maximum size of each chunk in bytes
/// * `overlap` - Number of bytes to overlap between consecutive chunks
///
/// # Returns
///
/// A vector of text chunks. Each chunk (except possibly the last) will be
/// exactly `chunk_size` bytes. Adjacent chunks will share `overlap` bytes.
///
/// # Example
///
/// ```
/// # use core::rag::indexer::chunk_text;
/// let text = "0123456789ABCDEF";
/// let chunks = chunk_text(text, 10, 2);
///
/// assert_eq!(chunks.len(), 2);
/// assert_eq!(chunks[0], "0123456789");
/// assert_eq!(chunks[1], "89ABCDEF");  // "89" is the overlap
/// ```
///
/// # Note on Byte vs Character Boundaries
///
/// This function works with byte indices, not character boundaries. For UTF-8
/// text with multi-byte characters, chunks may not align perfectly with
/// character boundaries. Consider using a smarter chunking strategy for
/// production use (e.g., splitting on sentence or paragraph boundaries).
pub fn chunk_text(text: &str, chunk_size: usize, overlap: usize) -> Vec<String> {
    if text.len() <= chunk_size {
        return vec![text.to_string()];
    }
    
    let mut chunks = Vec::new();
    let mut start = 0;
    
    while start < text.len() {
        let end = (start + chunk_size).min(text.len());
        chunks.push(text[start..end].to_string());
        
        if end == text.len() {
            break;
        }
        
        start += chunk_size - overlap;
    }
    
    chunks
}

/// A file that has been collected and read for indexing.
///
/// Represents a single file with its path and text content, ready to be
/// chunked and embedded.
#[derive(Debug, Clone)]
pub struct IndexedFile {
    /// Absolute or relative path to the source file.
    pub path: PathBuf,
    
    /// The complete text content of the file.
    ///
    /// Binary files and files that cannot be read as UTF-8 are skipped.
    pub content: String,
}

/// Recursively collects all indexable files from a directory.
///
/// Walks the directory tree starting from `dir_path`, filtering files by
/// extension and reading their content. Binary files and unreadable files
/// are silently skipped.
///
/// # Supported Extensions
///
/// Currently indexes files with these extensions:
/// - Code: `.rs`, `.go`, `.py`, `.js`, `.ts`, `.tsx`, `.jsx`
/// - Documentation: `.md`, `.txt`
///
/// # Arguments
///
/// * `dir_path` - Root directory to start collecting files from
///
/// # Returns
///
/// A vector of all successfully read indexable files found in the directory
/// tree. Files are returned in arbitrary order (depends on filesystem).
///
/// # Errors
///
/// Returns an error if:
/// - The directory doesn't exist or isn't accessible
/// - There's a permission error reading the directory structure
///
/// Individual file read errors are logged but don't fail the entire operation.
///
/// # Example
///
/// ```no_run
/// # use core::rag::indexer::collect_files;
/// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
/// let files = collect_files("./src").await?;
/// println!("Found {} indexable files", files.len());
///
/// for file in files {
///     println!("  {}: {} bytes", file.path.display(), file.content.len());
/// }
/// # Ok(())
/// # }
/// ```
pub async fn collect_files(dir_path: impl AsRef<Path>) -> Result<Vec<IndexedFile>> {
    let mut files = Vec::new();
    collect_files_recursive(dir_path.as_ref(), &mut files).await?;
    Ok(files)
}

fn collect_files_recursive<'a>(dir: &'a Path, files: &'a mut Vec<IndexedFile>) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<()>> + Send + 'a>> {
    Box::pin(async move {
    let mut entries = fs::read_dir(dir).await?;
    
    while let Some(entry) = entries.next_entry().await? {
        let path = entry.path();
        
        if path.is_dir() {
            collect_files_recursive(&path, files).await?;
        } else if is_indexable(&path) {
            if let Ok(content) = fs::read_to_string(&path).await {
                files.push(IndexedFile {
                    path: path.clone(),
                    content,
                });
            }
        }
    }
    
    Ok(())
    })
}

/// Checks if a file should be indexed based on its extension.
///
/// Returns `true` if the file has one of the supported extensions:
/// - Rust: `.rs`
/// - Go: `.go`
/// - Python: `.py`
/// - JavaScript/TypeScript: `.js`, `.ts`, `.tsx`, `.jsx`
/// - Documentation: `.md`, `.txt`
///
/// Files without extensions or with unsupported extensions return `false`.
///
/// # Arguments
///
/// * `path` - Path to check
///
/// # Example
///
/// ```
/// # use std::path::Path;
/// # use core::rag::indexer::is_indexable;
/// assert!(is_indexable(Path::new("main.rs")));
/// assert!(is_indexable(Path::new("README.md")));
/// assert!(!is_indexable(Path::new("binary.exe")));
/// assert!(!is_indexable(Path::new("no_extension")));
/// ```
fn is_indexable(path: &Path) -> bool {
    if let Some(ext) = path.extension() {
        matches!(
            ext.to_str(),
            Some("rs" | "go" | "py" | "js" | "ts" | "tsx" | "jsx" | "md" | "txt")
        )
    } else {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_chunk_text_small() {
        let text = "Hello";
        let chunks = chunk_text(text, 10, 2);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0], "Hello");
    }
    
    #[test]
    fn test_chunk_text_with_overlap() {
        let text = "0123456789ABCDEF";
        let chunks = chunk_text(text, 10, 2);
        
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0], "0123456789");
        assert_eq!(chunks[1], "89ABCDEF");
    }
    
    #[test]
    fn test_is_indexable() {
        assert!(is_indexable(Path::new("test.rs")));
        assert!(is_indexable(Path::new("test.md")));
        assert!(!is_indexable(Path::new("test.exe")));
        assert!(!is_indexable(Path::new("test")));
    }
}
