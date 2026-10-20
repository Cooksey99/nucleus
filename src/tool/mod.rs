//! Tool plugins and the plugin registry.
//!
//! The new `Tool` trait lands here with the core implementation task; until
//! then this module holds the legacy [`Plugin`] trait and its registry.

mod plugin;
mod registry;

pub use plugin::{Plugin, PluginError, PluginOutput, Result};
pub use registry::PluginRegistry;
