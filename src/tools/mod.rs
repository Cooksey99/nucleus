//! Built-in tools: file read and write, text search, and command execution.

mod commands;
mod files;
mod search;

pub use commands::ExecPlugin;
pub use files::{ReadFilePlugin, WriteFilePlugin};
pub use search::SearchPlugin;
