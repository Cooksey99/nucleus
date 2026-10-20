//! The agent surface: [`Turn`] and the supporting data types.
//!
//! `Agent` and `AgentBuilder` are not defined yet: their construction and driver
//! methods depend on the provider and tool traits, which arrive with the core
//! implementation task. This module currently holds only the data types that
//! describe the result of a turn and the per-agent configuration that does not
//! depend on those traits.
//!
//! See `docs/design/core-api.md` §8 for the full design.

use serde::{Deserialize, Serialize};

use crate::event::{StopReason, Usage};
use crate::message::Message;

/// The result of one completed turn.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct Turn {
    /// The assistant message as appended to the conversation.
    pub message: Message,
    /// Token usage for the turn.
    pub usage: Usage,
    /// Why the turn ended.
    pub stop_reason: StopReason,
    /// Retrieval queries the agent issued for this turn, in order.
    #[serde(default)]
    pub knowledge_queries: Vec<String>,
}

/// A caller decision about whether to run a tool call.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum Approval {
    /// The tool may run.
    Allow,
    /// The tool must not run.
    Deny,
}

/// How retrieved knowledge is attached to a turn.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[non_exhaustive]
pub enum KnowledgeMode {
    /// Retrieve and inject automatically, prefixed with the system prompt.
    #[default]
    Inject,
    /// Expose retrieval as a tool the model may call.
    Tool,
}
