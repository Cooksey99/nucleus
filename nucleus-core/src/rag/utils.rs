//! Utility functions for RAG indexing operations.
//!
//! This module provides helper functions for common indexing patterns,
//! such as finding and indexing project directories.

use std::path::{Path, PathBuf};
use tokio::fs;

/// Finds all subdirectories within a parent directory that match certain criteria.
///
/// This is useful for indexing multiple related projects or modules in a workspace.
///
/// # Arguments
///
/// * `parent_dir` - The parent directory to search
/// * `max_depth` - Maximum depth to search (1 = immediate children only)
///
/// # Returns
///
/// A vector of directory paths found.
///
/// # Errors
///
/// Returns an error if the directory cannot be read.
///
pub async fn find_subdirectories(
    parent_dir: impl AsRef<Path>,
    max_depth: usize,
) -> std::io::Result<Vec<PathBuf>> {
    let mut dirs = Vec::new();
    find_subdirectories_recursive(parent_dir.as_ref(), max_depth, 0, &mut dirs).await?;
    Ok(dirs)
}

fn find_subdirectories_recursive<'a>(
    dir: &'a Path,
    max_depth: usize,
    current_depth: usize,
    results: &'a mut Vec<PathBuf>,
) -> std::pin::Pin<Box<dyn std::future::Future<Output = std::io::Result<()>> + Send + 'a>> {
    Box::pin(async move {
        if current_depth >= max_depth {
            return Ok(());
        }

        let mut entries = fs::read_dir(dir).await?;

        while let Some(entry) = entries.next_entry().await? {
            let path = entry.path();

            if path.is_dir() {
                results.push(path.clone());
                find_subdirectories_recursive(&path, max_depth, current_depth + 1, results).await?;
            }
        }

        Ok(())
    })
}

/// Checks if a directory contains code files that should be indexed.
///
/// This helps filter out empty directories or directories with only
/// binary/non-indexable files.
///
/// # Arguments
///
/// * `dir_path` - The directory to check
/// * `extensions` - File extensions to look for (e.g., ["rs", "py", "js"])
///
/// # Returns
///
/// `true` if the directory contains at least one file with a matching extension.
///
pub async fn contains_indexable_files(
    dir_path: impl AsRef<Path>,
    extensions: &[String],
) -> bool {
    let mut entries = match fs::read_dir(dir_path).await {
        Ok(e) => e,
        Err(_) => return false,
    };

    while let Ok(Some(entry)) = entries.next_entry().await {
        let path = entry.path();

        if path.is_file() {
            if extensions.is_empty() {
                return true;
            }

            if let Some(ext) = path.extension() {
                if let Some(ext_str) = ext.to_str() {
                    if extensions.iter().any(|e| e == ext_str) {
                        return true;
                    }
                }
            }
        }
    }

    false
}

/// Gets the relative path from a base directory.
///
/// Useful for displaying cleaner paths in logs and metadata.
///
pub fn get_relative_path(base: impl AsRef<Path>, full: impl AsRef<Path>) -> PathBuf {
    let base = base.as_ref();
    let full = full.as_ref();

    full.strip_prefix(base).unwrap_or(full).to_path_buf()
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;
    use tokio::fs;

    #[tokio::test]
    async fn test_find_subdirectories() {
        let temp = tempdir().unwrap();
        let base = temp.path();

        // Create some nested directories
        fs::create_dir_all(base.join("dir1/subdir1")).await.unwrap();
        fs::create_dir_all(base.join("dir2")).await.unwrap();

        let dirs = find_subdirectories(base, 2).await.unwrap();

        assert!(dirs.len() >= 2);
    }

    #[tokio::test]
    async fn test_contains_indexable_files() {
        let temp = tempdir().unwrap();
        let base = temp.path();

        // Create a test file
        fs::write(base.join("test.rs"), "fn main() {}").await.unwrap();

        let extensions = vec!["rs".to_string()];
        assert!(contains_indexable_files(base, &extensions).await);

        let wrong_extensions = vec!["py".to_string()];
        assert!(!contains_indexable_files(base, &wrong_extensions).await);
    }

    #[test]
    fn test_get_relative_path() {
        let base = PathBuf::from("/home/user/project");
        let full = PathBuf::from("/home/user/project/src/main.rs");

        let relative = get_relative_path(&base, &full);
        assert_eq!(relative, PathBuf::from("src/main.rs"));
    }
}
