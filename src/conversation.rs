//! [`Conversation`]: plain, serializable conversation state with a stable id.
//!
//! A conversation holds no provider handles and no agent state. History is
//! append-only; the agent only ever appends to it.
//!
//! See `docs/design/core-api.md` §4.3 for the full design.

use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};
use crate::message::Message;
use crate::store::ConversationStore;

/// Plain, serializable conversation state with a stable id.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct Conversation {
    /// Stable for the life of the conversation. Used as the prompt-cache key
    /// and, on Grok, for the per-conversation routing headers.
    pub id: ConversationId,
    /// Append-only message history.
    pub messages: Vec<Message>,
    /// Creation time, RFC 3339, for stores and diagnostics.
    pub created_at: String,
}

/// A conversation's stable identifier. Printable, serde, and cheap to clone.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[non_exhaustive]
pub struct ConversationId(String);

impl ConversationId {
    /// Mint a fresh id.
    pub fn new() -> Self {
        Self(uuid_like())
    }

    /// Parse an id from its string form.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Serialization`] when the string is empty.
    pub fn parse(s: &str) -> Result<Self> {
        if s.is_empty() {
            return Err(Error::Serialization("empty conversation id".to_string()));
        }
        Ok(Self(s.to_string()))
    }

    /// The id as a string.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl Default for ConversationId {
    fn default() -> Self {
        Self::new()
    }
}

/// A monotonic, opaque id string. The concrete scheme (a UUIDv7) is a private
/// detail of the conversation id.
fn uuid_like() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};

    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    format!("conv-{nanos:032x}")
}

impl Conversation {
    /// A new, empty conversation with a fresh id.
    pub fn new() -> Self {
        Self {
            id: ConversationId::new(),
            messages: Vec::new(),
            created_at: String::new(),
        }
    }

    /// An empty conversation with a caller-supplied id.
    pub fn with_id(id: ConversationId) -> Self {
        Self {
            id,
            messages: Vec::new(),
            created_at: String::new(),
        }
    }

    /// Append a message. The only mutator used by the agent loop.
    pub fn push(&mut self, message: Message) {
        self.messages.push(message);
    }

    /// The message history.
    pub fn messages(&self) -> &[Message] {
        &self.messages
    }

    /// Drop all turns, keep the id.
    pub fn clear(&mut self) {
        self.messages.clear();
    }

    /// Load from a store, when the agent has one configured.
    ///
    /// # Errors
    ///
    /// Returns the store's error unchanged.
    pub async fn load(store: &dyn ConversationStore, id: &ConversationId) -> Result<Option<Self>> {
        store.load(id).await
    }

    /// Save to a store, when the agent has one configured.
    ///
    /// # Errors
    ///
    /// Returns the store's error unchanged.
    pub async fn save(&self, store: &dyn ConversationStore) -> Result<()> {
        store.save(self).await
    }
}

impl Default for Conversation {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn push_appends_and_clear_keeps_id() {
        let mut conversation = Conversation::new();
        let id = conversation.id.clone();
        conversation.push(Message::user("hi"));
        assert_eq!(conversation.messages().len(), 1);
        conversation.clear();
        assert!(conversation.messages().is_empty());
        assert_eq!(conversation.id, id);
    }

    #[test]
    fn ids_are_distinct() {
        assert_ne!(ConversationId::new(), ConversationId::new());
    }

    #[test]
    fn history_is_append_only() {
        // The only mutator is `push`; there is no API to edit or remove a
        // message that was already sent, so earlier messages are unchanged
        // after later appends.
        let mut conversation = Conversation::new();
        conversation.push(Message::user("first"));
        let first_text = conversation.messages()[0].text();

        conversation.push(Message::assistant_text("second"));
        conversation.push(Message::user("third"));

        assert_eq!(conversation.messages().len(), 3);
        assert_eq!(conversation.messages()[0].text(), first_text);
        assert_eq!(conversation.messages()[0].text(), "first");
        assert_eq!(conversation.messages()[1].text(), "second");
        assert_eq!(conversation.messages()[2].text(), "third");
    }
}
