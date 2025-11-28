//! Standard library plugins for nucleus
//!
//! The standard library is a collection of built-in plugins that are typical in most use-cases.
//! Provides essential plugins that work out of the box:
//! - File operations (read, write, list)
//! - Search (text and code search)
//! - Execution (safe command execution)

mod files;

pub use files::{ReadFilePlugin, WriteFilePlugin};
// TODO: Implement ListDirectoryPlugin
// TODO: Implement search functionality
// TODO: Implement command execution
