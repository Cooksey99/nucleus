/// Common file patterns used across nucleus for filtering files during indexing and search.

/// Default patterns to exclude from indexing and search.
/// Returns patterns that should be skipped in file operations.
pub fn default_exclude_patterns() -> Vec<String> {
    vec![
        // Version control
        ".git".to_string(),
        ".svn".to_string(),
        ".hg".to_string(),
        
        // Build outputs
        "target".to_string(),
        "dist".to_string(),
        "build".to_string(),
        "out".to_string(),
        ".next".to_string(),
        
        // Package managers
        "node_modules".to_string(),
        "vendor".to_string(),
        ".pnpm-store".to_string(),
        
        // Python
        "__pycache__".to_string(),
        ".venv".to_string(),
        "venv".to_string(),
        ".pytest_cache".to_string(),
        "*.egg-info".to_string(),
        
        // IDEs
        ".vscode".to_string(),
        ".idea".to_string(),
        
        // OS
        ".DS_Store".to_string(),
        "Thumbs.db".to_string(),
        
        // Temp/cache
        "tmp".to_string(),
        "temp".to_string(),
        "cache".to_string(),
        ".cache".to_string(),
        
        // Database/Storage
        "storage".to_string(),
        "qdrant_storage".to_string(),
        ".qdrant".to_string(),
        "data".to_string(),
        "db".to_string(),
        ".db".to_string(),
    ]
}

/// Binary file extensions that should be skipped.
pub fn binary_extensions() -> Vec<&'static str> {
    vec![
        // Images
        "png", "jpg", "jpeg", "gif", "bmp", "ico", "svg", "webp",
        // Documents
        "pdf", "doc", "docx", "xls", "xlsx", "ppt", "pptx",
        // Archives
        "zip", "tar", "gz", "bz2", "7z", "rar",
        // Executables/Libraries
        "exe", "dll", "so", "dylib", "a", "lib",
        // Media
        "mp3", "mp4", "avi", "mov", "mkv", "wav", "flac",
        // Binary data
        "wasm", "bin", "dat", "db", "sqlite", "sqlite3",
        // Lock files (binary)
        "lock",
    ]
}

/// Check if a path should be skipped based on exclude patterns.
pub fn should_exclude(path: &std::path::Path, exclude_patterns: &[String]) -> bool {
    let path_str = path.to_string_lossy();
    
    // Check exclude patterns
    for pattern in exclude_patterns {
        if path_str.contains(pattern) {
            return true;
        }
    }
    
    // Check binary extensions
    if let Some(ext) = path.extension() {
        let ext = ext.to_string_lossy().to_lowercase();
        if binary_extensions().contains(&ext.as_str()) {
            return true;
        }
    }
    
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    
    #[test]
    fn test_should_exclude_git() {
        let path = PathBuf::from("/home/user/project/.git/config");
        let patterns = default_exclude_patterns();
        assert!(should_exclude(&path, &patterns));
    }
    
    #[test]
    fn test_should_exclude_node_modules() {
        let path = PathBuf::from("/home/user/project/node_modules/package/index.js");
        let patterns = default_exclude_patterns();
        assert!(should_exclude(&path, &patterns));
    }
    
    #[test]
    fn test_should_not_exclude_source() {
        let path = PathBuf::from("/home/user/project/src/main.rs");
        let patterns = default_exclude_patterns();
        assert!(!should_exclude(&path, &patterns));
    }
    
    #[test]
    fn test_should_exclude_binary() {
        let path = PathBuf::from("/home/user/project/image.png");
        let patterns = default_exclude_patterns();
        assert!(should_exclude(&path, &patterns));
    }
}
