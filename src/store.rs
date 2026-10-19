//! The [`ConversationStore`] trait: pluggable conversation persistence.
//!
//! The agent never consults a store on its own; only [`Conversation::load`] and
//! [`Conversation::save`] call into it.
//!
//! [`Conversation::load`]: crate::conversation::Conversation::load
//! [`Conversation::save`]: crate::conversation::Conversation::save
//!
//! See `docs/design/core-api.md` §7.3 for the full design.

use async_trait::async_trait;

use crate::conversation::{Conversation, ConversationId};
use crate::error::Result;

/// Persistence for [`Conversation`]s. A trait object, so any backend fits.
#[async_trait]
pub trait ConversationStore: Send + Sync {
    /// Save a conversation, replacing any existing one with the same id.
    ///
    /// # Errors
    ///
    /// Returns a store error when the write fails.
    async fn save(&self, conversation: &Conversation) -> Result<()>;

    /// Load a conversation by id. `Ok(None)` when it does not exist.
    ///
    /// # Errors
    ///
    /// Returns a store error when the read fails.
    async fn load(&self, id: &ConversationId) -> Result<Option<Conversation>>;

    /// Delete a conversation. Deleting a missing id is not an error.
    ///
    /// # Errors
    ///
    /// Returns a store error when the delete fails.
    async fn delete(&self, id: &ConversationId) -> Result<()>;

    /// List known conversation ids, newest first.
    ///
    /// # Errors
    ///
    /// Returns a store error when the listing fails.
    async fn list(&self) -> Result<Vec<ConversationId>>;
}
