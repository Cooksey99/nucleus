//! Structured extraction: `extract::<T>()`.
//!
//! The extract surface is not defined yet. It depends on the `Agent` driver and
//! the provider traits, which arrive with the core implementation task. It will
//! build against `schemars` (workspace `schemars = "1.2"`) and appear here as a
//! method on `Agent` plus the private schema transform and local validator
//! described in the design.
//!
//! See `docs/design/core-api.md` §11.2 for the full design.
