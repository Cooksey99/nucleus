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
use crate::message::{Message, ToolCallId};
use crate::tool::ToolAnnotations;

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
    Approve,
    /// Refuse; a denial goes back to the model as a tool result with
    /// `is_error: true`.
    Deny,
}

/// The tool-call information a hook sees before the tool runs.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct ToolCall {
    /// The id of the call.
    pub id: ToolCallId,
    /// The model-facing tool name.
    pub name: String,
    /// Raw arguments JSON, as the model produced it.
    pub input: Box<serde_json::value::RawValue>,
    /// Untrusted annotations from the tool, when it declares any. Reaches the
    /// approval hook as context and is never used to skip approval.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub annotations: Option<ToolAnnotations>,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_approval_hook_sees_the_call_and_returns_an_approval() {
        let call = ToolCall {
            id: ToolCallId("call_1".to_string()),
            name: "exec".to_string(),
            input: serde_json::value::RawValue::from_string(r#"{"command":"ls"}"#.to_string())
                .expect("valid raw json"),
            annotations: Some(ToolAnnotations {
                destructive_hint: true,
                ..ToolAnnotations::default()
            }),
        };

        // The approval gate's shape, per §8.1: `Fn(&ToolCall) -> Approval`.
        let gate = |call: &ToolCall| -> Approval {
            if call
                .annotations
                .as_ref()
                .is_some_and(|annotations| annotations.destructive_hint)
            {
                Approval::Deny
            } else {
                Approval::Approve
            }
        };

        assert_eq!(gate(&call), Approval::Deny);
        let json = serde_json::to_string(&call).expect("the call serializes");
        assert!(json.contains("\"exec\""));
    }
}
