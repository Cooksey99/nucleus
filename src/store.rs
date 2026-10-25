//! The [`ConversationStore`] trait: pluggable conversation persistence.
//!
//! A configured agent saves the conversation at the end of a successful run.
//! [`Conversation::load`] and [`Conversation::save`] remain the direct API.
//!
//! [`Conversation::load`]: crate::conversation::Conversation::load
//! [`Conversation::save`]: crate::conversation::Conversation::save
//!
//! See `docs/design/core-api.md` §7.3 for the full design.

use std::sync::{Mutex, MutexGuard, PoisonError};

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

/// An in-memory [`ConversationStore`].
///
/// Conversations live for as long as the store does and are lost when it is
/// dropped. Always compiled, without a feature gate.
#[derive(Debug, Default)]
pub struct MemoryStore {
    conversations: Mutex<Vec<Conversation>>,
}

impl MemoryStore {
    /// A new, empty store.
    pub fn new() -> Self {
        Self {
            conversations: Mutex::new(Vec::new()),
        }
    }

    /// Lock the inner list, recovering from a poisoned mutex instead of
    /// panicking: the stored conversations are plain data.
    fn lock(&self) -> MutexGuard<'_, Vec<Conversation>> {
        self.conversations
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
    }
}

#[async_trait]
impl ConversationStore for MemoryStore {
    async fn save(&self, conversation: &Conversation) -> Result<()> {
        let mut conversations = self.lock();
        // Re-saving an id replaces it and moves it to the front of "newest
        // first", rather than leaving a stale order.
        conversations.retain(|c| c.id != conversation.id);
        conversations.push(conversation.clone());
        Ok(())
    }

    async fn load(&self, id: &ConversationId) -> Result<Option<Conversation>> {
        Ok(self.lock().iter().find(|c| c.id == *id).cloned())
    }

    async fn delete(&self, id: &ConversationId) -> Result<()> {
        self.lock().retain(|c| c.id != *id);
        Ok(())
    }

    async fn list(&self) -> Result<Vec<ConversationId>> {
        Ok(self.lock().iter().rev().map(|c| c.id.clone()).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn save_load_resume_continues_the_same_history() {
        let store = MemoryStore::new();
        let mut conversation = Conversation::new();
        conversation.push(crate::message::Message::user("hi"));
        conversation.push(crate::message::Message::assistant_text("hello"));
        conversation.save(&store).await.expect("saves");

        let mut resumed = Conversation::load(&store, &conversation.id)
            .await
            .expect("loads")
            .expect("exists");
        assert_eq!(resumed.id, conversation.id);
        assert_eq!(resumed.messages().len(), 2);

        // Resume: continue the same history and save it back.
        resumed.push(crate::message::Message::user("next"));
        resumed.save(&store).await.expect("saves");

        let again = Conversation::load(&store, &resumed.id)
            .await
            .expect("loads")
            .expect("exists");
        assert_eq!(again.messages().len(), 3);
        assert_eq!(again.messages()[0].text(), "hi");
        assert_eq!(again.messages()[2].text(), "next");
    }

    #[tokio::test]
    async fn unknown_id_loads_as_none() {
        let store = MemoryStore::new();
        let missing = ConversationId::new();
        assert!(store.load(&missing).await.expect("loads").is_none());
    }

    #[tokio::test]
    async fn delete_removes_and_list_is_newest_first() {
        let store = MemoryStore::new();
        let first = Conversation::new();
        let second = Conversation::new();
        first.save(&store).await.expect("saves");
        second.save(&store).await.expect("saves");

        assert_eq!(
            store.list().await.expect("lists"),
            vec![second.id.clone(), first.id.clone()]
        );

        store.delete(&second.id).await.expect("deletes");
        assert_eq!(store.list().await.expect("lists"), vec![first.id.clone()]);
        // Deleting a missing id is not an error.
        store.delete(&second.id).await.expect("deletes again");
    }

    #[tokio::test]
    async fn save_replaces_and_moves_to_front() {
        let store = MemoryStore::new();
        let first = Conversation::new();
        let second = Conversation::new();
        first.save(&store).await.expect("saves");
        second.save(&store).await.expect("saves");

        let mut updated = first.clone();
        updated.push(crate::message::Message::user("more"));
        updated.save(&store).await.expect("saves");

        assert_eq!(
            store.list().await.expect("lists"),
            vec![first.id.clone(), second.id.clone()]
        );
        let reloaded = store.load(&first.id).await.expect("loads").expect("exists");
        assert_eq!(reloaded.messages().len(), 1);
    }
}
