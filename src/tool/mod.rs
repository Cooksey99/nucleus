//! Tools: the model-facing [`Tool`] trait, [`tool_fn`], and the legacy
//! [`Plugin`] trait with its registry.
//!
//! [`Tool`] is the new extension point. The legacy [`Plugin`] trait and its
//! [`PluginRegistry`] keep working for the hosted and local implementations
//! until Phase 3.

mod api;
mod plugin;
mod registry;

pub use api::{tool_fn, tool_fn_sync, Tool, ToolAnnotations, ToolContext};
pub use plugin::{Plugin, PluginError, PluginOutput, Result};
pub use registry::PluginRegistry;
