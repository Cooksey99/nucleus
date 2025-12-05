//! File indexing and text chunking for RAG.
//!
//! This module provides functionality to:
//! - Recursively collect code files from directories
//! - Split large text into overlapping chunks
//! - Filter files by extension and exclude patterns

use crate::config::IndexerConfig;
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
/// This function is internal to the RAG system.
///
/// # UTF-8 Safety
///
/// This function respects UTF-8 character boundaries by finding the nearest
/// valid boundary when chunk sizes would split multi-byte characters.
pub(crate) fn chunk_text(text: &str, chunk_size: usize, overlap: usize) -> Vec<String> {
    if text.is_empty() {
        eprintln!("WARNING: chunk_text called with empty text");
        return vec![];
    }
    
    if text.len() <= chunk_size {
        return vec![text.to_string()];
    }
    
    let mut chunks = Vec::new();
    let mut start = 0;
    
    while start < text.len() {
        let mut end = (start + chunk_size).min(text.len());
        
        // Find the nearest character boundary at or before 'end'
        while end > start && !text.is_char_boundary(end) {
            end -= 1;
        }
        
        let chunk = text[start..end].to_string();
        if chunk.is_empty() {
            eprintln!("WARNING: Empty chunk created at start={}, end={}", start, end);
        } else {
            chunks.push(chunk);
        }
        
        if end == text.len() {
            break;
        }
        
        // Calculate next start position, ensuring it's on a char boundary
        let step = chunk_size.saturating_sub(overlap);
        start += step;
        
        // Adjust start to nearest char boundary
        while start < text.len() && !text.is_char_boundary(start) {
            start += 1;
        }
    }
    
    chunks
}

/// A file that has been collected and read for indexing.
#[derive(Debug, Clone)]
pub struct IndexedFile {
    pub path: PathBuf,
    pub content: String,
}

/// Recursively collects all indexable files from a directory.
///
/// Walks the directory tree starting from `dir_path`, filtering files based on
/// the provided configuration. Binary files and unreadable files are silently skipped.
///
/// # Filtering
///
/// Files are filtered based on:
/// - **Extensions**: Only files with extensions in `config.extensions` are indexed.
///   If empty, all readable text files are indexed.
/// - **Exclude patterns**: Directories or files matching patterns in `config.exclude_patterns`
///   are skipped (e.g., "node_modules", ".git").
///
/// This function is internal to the RAG system. Use [`Rag::index_directory`](crate::rag::Rag::index_directory)
/// for public-facing directory indexing.
pub(crate) async fn collect_files(dir_path: impl AsRef<Path>, config: &IndexerConfig) -> Result<Vec<IndexedFile>> {
    let mut files = Vec::new();
    collect_files_recursive(dir_path.as_ref(), &mut files, config).await?;
    Ok(files)
}

fn collect_files_recursive<'a>(
    dir: &'a Path, 
    files: &'a mut Vec<IndexedFile>,
    config: &'a IndexerConfig,
) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<()>> + Send + 'a>> {
    Box::pin(async move {
        let mut entries = fs::read_dir(dir).await?;
        
        while let Some(entry) = entries.next_entry().await? {
            let path = entry.path();
            
            if should_exclude(&path, &config.exclude_patterns) {
                continue;
            }
            
            if path.is_dir() {
                collect_files_recursive(&path, files, config).await?;
            } else if is_indexable(&path, &config.extensions) {
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
/// If `extensions` is empty, all files are considered indexable (useful for
/// catching files without extensions like Dockerfile, Makefile, etc.).
fn is_indexable(path: &Path, extensions: &[String]) -> bool {
    if extensions.is_empty() {
        return true;
    }
    
    if let Some(ext) = path.extension() {
        if let Some(ext_str) = ext.to_str() {
            return extensions.iter().any(|e| e == ext_str);
        }
    }
    
    false
}

/// Checks if a path should be excluded based on exclude patterns.
///
/// A path is excluded if any component of its path matches an exclude pattern.
fn should_exclude(path: &Path, patterns: &[String]) -> bool {
    path.components().any(|component| {
        if let Some(name) = component.as_os_str().to_str() {
            patterns.iter().any(|pattern| name.contains(pattern))
        } else {
            false
        }
    })
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
        let extensions = vec!["rs".to_string(), "md".to_string()];
        
        assert!(is_indexable(Path::new("test.rs"), &extensions));
        assert!(is_indexable(Path::new("test.md"), &extensions));
        assert!(!is_indexable(Path::new("test.exe"), &extensions));
        assert!(!is_indexable(Path::new("test"), &extensions));
    }
    
    #[test]
    fn test_is_indexable_empty_extensions() {
        let empty_extensions: Vec<String> = vec![];
        
        assert!(is_indexable(Path::new("Dockerfile"), &empty_extensions));
        assert!(is_indexable(Path::new("Makefile"), &empty_extensions));
        assert!(is_indexable(Path::new("test.rs"), &empty_extensions));
    }
    
    #[test]
    fn test_should_exclude() {
        let patterns = vec!["node_modules".to_string(), ".git".to_string(), "target".to_string()];
        
        assert!(should_exclude(Path::new("src/node_modules/file.js"), &patterns));
        assert!(should_exclude(Path::new(".git/config"), &patterns));
        assert!(should_exclude(Path::new("target/debug/main"), &patterns));
        assert!(!should_exclude(Path::new("src/main.rs"), &patterns));
    }
}
