//! The message model: [`Message`], [`Part`] and the pieces a part carries.
//!
//! An assistant message is an **ordered list of parts**, because providers
//! interleave thinking, tool calls and text and the order must replay exactly.
//! Tool-call input and provider-specific items are stored as raw JSON so they
//! replay byte for byte. The system prompt belongs to the agent and is never
//! stored in a [`Message`].
//!
//! See `docs/design/core-api.md` §4 for the full design.

use serde::{Deserialize, Serialize};

/// One message in a conversation.
///
/// [`Role`] is only [`Role::User`] or [`Role::Assistant`]: the system prompt
/// belongs to the `Agent` and is never stored here. Tool results travel in a
/// `User` message, as they do on the wire.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct Message {
    /// Who sent this message.
    pub role: Role,
    /// Ordered parts. Order is significant and is replayed as-is.
    pub parts: Vec<Part>,
}

/// The role a [`Message`] can have.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum Role {
    /// A user message. Tool results also travel in a user message.
    User,
    /// An assistant message.
    Assistant,
}

/// One ordered piece of a [`Message`].
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub enum Part {
    /// Visible text.
    Text {
        /// The text content.
        text: String,
    },

    /// Reasoning / thinking, for display and (where the provider replays it)
    /// for storage inside a [`Part::Provider`] alongside it.
    Reasoning {
        /// The reasoning text.
        text: String,
        /// Whether this is a summary rather than the full reasoning.
        summary: bool,
    },

    /// A tool call the model asked for. `input` is raw JSON so it replays byte
    /// for byte.
    ToolCall {
        /// The provider's tool-call id.
        id: ToolCallId,
        /// The tool's model-facing name.
        name: String,
        /// The raw argument JSON, as the model produced it.
        input: Box<serde_json::value::RawValue>,
    },

    /// The result sent back for a tool call.
    ToolResult {
        /// The id of the call this answers.
        call_id: ToolCallId,
        /// The tool's output.
        output: ToolOutput,
        /// Whether the tool failed.
        is_error: bool,
    },

    /// Retrieved knowledge attached to a user turn, kept separate from what the
    /// user typed so apps can render or hide citations.
    Knowledge {
        /// The retrieved text.
        text: String,
        /// Where the text came from.
        sources: Vec<KnowledgeSource>,
    },

    /// Opaque provider-specific output, replayed unchanged to the same dialect.
    /// See [`ProviderItem`].
    Provider(ProviderItem),
}

impl Message {
    /// A user message with a single text part.
    pub fn user(text: impl Into<String>) -> Self {
        Self::assistant_text_with_role(Role::User, text)
    }

    /// An assistant message with a single text part.
    pub fn assistant_text(text: impl Into<String>) -> Self {
        Self::assistant_text_with_role(Role::Assistant, text)
    }

    fn assistant_text_with_role(role: Role, text: impl Into<String>) -> Self {
        Self::parts(role, vec![Part::Text { text: text.into() }])
    }

    /// A message with the given role and parts.
    pub fn parts(role: Role, parts: Vec<Part>) -> Self {
        Self { role, parts }
    }

    /// Concatenated [`Part::Text`] parts, the common case.
    pub fn text(&self) -> String {
        let mut out = String::new();
        for part in &self.parts {
            if let Part::Text { text } = part {
                out.push_str(text);
            }
        }
        out
    }

    /// The tool calls in this message, as `(id, name, raw arguments)`.
    pub fn tool_calls(&self) -> impl Iterator<Item = (&ToolCallId, &str, &str)> {
        self.parts.iter().filter_map(|part| match part {
            Part::ToolCall { id, name, input } => Some((id, name.as_str(), input.get())),
            _ => None,
        })
    }

    /// The [`Part::Provider`] items in order.
    pub fn provider_items(&self) -> impl Iterator<Item = &ProviderItem> {
        self.parts.iter().filter_map(|part| match part {
            Part::Provider(item) => Some(item),
            _ => None,
        })
    }
}

/// A provider's own tool-call id, passed through unchanged.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[non_exhaustive]
pub struct ToolCallId(pub String);

/// A tool's result. `content` is what the model reads; `structured` is the JSON
/// form when the tool produced one.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct ToolOutput {
    /// The text the model reads.
    pub content: String,
    /// The structured JSON form, when the tool produced one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub structured: Option<serde_json::Value>,
}

impl ToolOutput {
    /// A text-only output: what the model reads, with no structured form.
    pub fn text(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            structured: None,
        }
    }
}

/// A source citation for a [`Part::Knowledge`]. Line ranges are 1-based and
/// inclusive; a whole-file citation has `start_line == 1` and `end_line == 0`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct KnowledgeSource {
    /// The source file path.
    pub file: String,
    /// The first line of the excerpt (1-based).
    pub start_line: u32,
    /// The last line of the excerpt (1-based, inclusive).
    pub end_line: u32,
}

/// A provider-specific output item stored verbatim and replayed unchanged.
///
/// Never contains a tool call or tool result: those are ordinary [`Part`]s and
/// are never dropped. Examples: an Anthropic `thinking` block, an OpenAI/xAI
/// `reasoning` item with `encrypted_content`, a `compaction` item.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct ProviderItem {
    /// Endpoint dialect, e.g. "anthropic", "openai", "xai".
    pub dialect: String,
    /// The model that produced the item.
    pub model: String,
    /// A short kind tag for diagnostics ("thinking", "reasoning", "phase", …).
    pub kind: String,
    /// Hash of the rendered system prompt + tool set at the time of capture.
    /// Empty for dialects that do not bind to a prefix.
    pub prefix_fingerprint: String,
    /// The item exactly as received.
    pub raw: Box<serde_json::value::RawValue>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::value::RawValue;

    /// A `RawValue` from a literal, for tests.
    fn raw(json: &str) -> Box<RawValue> {
        RawValue::from_string(json.to_string()).expect("valid raw json")
    }

    #[test]
    fn text_concatenates_only_text_parts() {
        let message = Message::parts(
            Role::Assistant,
            vec![
                Part::Reasoning {
                    text: "hm".to_string(),
                    summary: false,
                },
                Part::Text {
                    text: "hello ".to_string(),
                },
                Part::Text {
                    text: "world".to_string(),
                },
            ],
        );
        assert_eq!(message.text(), "hello world");
    }

    #[test]
    fn user_message_is_text() {
        let message = Message::user("hi");
        assert_eq!(message.role, Role::User);
        assert_eq!(message.text(), "hi");
    }

    #[test]
    fn part_order_survives_serde_round_trip() {
        let message = Message::parts(
            Role::Assistant,
            vec![
                Part::Reasoning {
                    text: "think".to_string(),
                    summary: false,
                },
                Part::Text {
                    text: "answer".to_string(),
                },
                Part::ToolCall {
                    id: ToolCallId("call_1".to_string()),
                    name: "search".to_string(),
                    input: raw(r#"{"query":"rust"}"#),
                },
                Part::ToolResult {
                    call_id: ToolCallId("call_1".to_string()),
                    output: ToolOutput {
                        content: "ok".to_string(),
                        structured: None,
                    },
                    is_error: false,
                },
                Part::Knowledge {
                    text: "excerpt".to_string(),
                    sources: vec![KnowledgeSource {
                        file: "src/lib.rs".to_string(),
                        start_line: 1,
                        end_line: 10,
                    }],
                },
                Part::Provider(ProviderItem {
                    dialect: "anthropic".to_string(),
                    model: "claude".to_string(),
                    kind: "thinking".to_string(),
                    prefix_fingerprint: String::new(),
                    raw: raw(r#"{"type":"thinking"}"#),
                }),
            ],
        );

        let json = serde_json::to_string(&message).expect("serializes");
        let back: Message = serde_json::from_str(&json).expect("deserializes");

        assert_eq!(back.parts.len(), 6);
        assert!(matches!(back.parts[0], Part::Reasoning { .. }));
        assert!(matches!(back.parts[1], Part::Text { .. }));
        assert!(matches!(back.parts[2], Part::ToolCall { .. }));
        assert!(matches!(back.parts[3], Part::ToolResult { .. }));
        assert!(matches!(back.parts[4], Part::Knowledge { .. }));
        assert!(matches!(back.parts[5], Part::Provider(_)));
        assert_eq!(serde_json::to_string(&back).expect("reserializes"), json);
    }

    #[test]
    fn tool_call_raw_json_replays_byte_for_byte() {
        // Deliberately odd key order and whitespace: the bytes must survive.
        let original = r#"{"z":1,   "a":[1, 2,  3], "nested":{"k":"v"}}"#;
        let message = Message::parts(
            Role::Assistant,
            vec![Part::ToolCall {
                id: ToolCallId("call_1".to_string()),
                name: "search".to_string(),
                input: raw(original),
            }],
        );

        let json = serde_json::to_string(&message).expect("serializes");
        let back: Message = serde_json::from_str(&json).expect("deserializes");

        let (_, name, input) = back.tool_calls().next().expect("one tool call");
        assert_eq!(name, "search");
        assert_eq!(input, original);
    }

    #[test]
    fn knowledge_part_citations_survive_serde_round_trip() {
        let message = Message::parts(
            Role::User,
            vec![Part::Knowledge {
                text: "retrieved excerpt".to_string(),
                sources: vec![
                    KnowledgeSource {
                        file: "src/a.rs".to_string(),
                        start_line: 3,
                        end_line: 7,
                    },
                    // A whole-file citation: start_line == 1, end_line == 0.
                    KnowledgeSource {
                        file: "docs/b.md".to_string(),
                        start_line: 1,
                        end_line: 0,
                    },
                ],
            }],
        );

        let json = serde_json::to_string(&message).expect("serializes");
        let back: Message = serde_json::from_str(&json).expect("deserializes");

        match &back.parts[0] {
            Part::Knowledge { text, sources } => {
                assert_eq!(text, "retrieved excerpt");
                assert_eq!(sources.len(), 2);
                assert_eq!(sources[0].file, "src/a.rs");
                assert_eq!(sources[0].start_line, 3);
                assert_eq!(sources[0].end_line, 7);
                assert_eq!(sources[1].file, "docs/b.md");
                assert_eq!(sources[1].start_line, 1);
                assert_eq!(sources[1].end_line, 0);
            }
            other => panic!("expected a knowledge part, got {other:?}"),
        }
    }
}
