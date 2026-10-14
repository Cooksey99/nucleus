#![recursion_limit = "256"]

//! nucleus - Privacy-first, modular AI engine
//!
//! This is the convenience wrapper crate that re-exports nucleus components
//! with optional feature flags for easy usage.
//!
//! # Quick Start
//!
//! ```toml
//! [dependencies]
//! nucleus = "0.1"  # Includes core + std plugins by default
//! ```
//!
//! # Features
//!
//! - `std` (default): Include standard library plugins

// Re-export core
pub use nucleus_core::*;
pub use nucleus_plugin;

// Re-export std plugins if feature is enabled
#[cfg(feature = "std")]
pub use nucleus_std;

/// Prelude module for convenient imports
pub mod prelude {
    pub use nucleus_core::*;
    pub use nucleus_plugin::{Permission, Plugin, PluginRegistry};

    #[cfg(feature = "std")]
    pub use nucleus_std;
}
