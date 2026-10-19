//! The built-in mock provider.
//!
//! The mock provider is not defined yet. It is the offline test provider: a
//! scripted queue of responses on the new `Provider` trait, plus a
//! `MockHandle` for assertions. It arrives with the trait, which depends on the
//! event and message model and on the core implementation task.
//!
//! See `docs/design/core-api.md` §12 for the full design.
