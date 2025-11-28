use serde::{Deserialize, Serialize};

/// Type of request being made to the server.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum RequestType {
    /// Chat with AI (streaming response)
    Chat,
    /// Edit mode with AI assistance (streaming response)
    Edit,
    /// Add content to knowledge base
    Add,
    /// Index a directory for RAG
    Index,
    /// Get knowledge base statistics
    Stats,
}

/// Type of streaming response chunk.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ChunkType {
    /// Partial response content (multiple chunks per request)
    Chunk,
    /// Final response with complete content
    Done,
    /// An error occurred
    Error,
}

/// A message in conversation history.
///
/// **Note:** This may be identical to the `ollama::Message`.
/// This is kept separate for readability and future modifications
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    /// Role of the message sender.
    ///
    /// **Note:** These are model-dependent.
    /// For Qwen models, these are typically valid values:
    /// - `"system"` - Instructions/context for the AI
    /// - `"user"` - Input from the user
    /// - `"assistant"` - Response from the AI
    pub role: String,

    /// The text content of the message.
    pub content: String,
}

/// Request from client to server.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Request {
    /// Type of request to perform.
    #[serde(rename = "type")]
    pub request_type: RequestType,

    /// The main content/query for the request.
    ///
    /// For chat/edit: the user's message
    /// For add: the text to add to knowledge base
    /// For index: the directory path to index
    /// For stats: ignored
    pub content: String,

    /// Optional working directory context.
    ///
    /// Can be used by the AI to understand the user's current location.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pwd: Option<String>,

    /// Optional conversation history for chat/edit requests.
    ///
    /// Allows maintaining context across multiple interactions.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub history: Option<Vec<Message>>,
}

/// Streaming response chunk sent to client.
///
/// Responses are sent as a stream of JSON objects, one per line.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamChunk {
    /// Type of chunk being sent.
    #[serde(rename = "type")]
    pub chunk_type: ChunkType,

    /// The content of this chunk.
    ///
    /// For "chunk" type: partial response text
    /// For "done" type: complete response text
    /// For "error" type: empty (error details in `error` field)
    pub content: String,

    /// Error message if chunk_type is "error".
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

impl StreamChunk {
    pub fn chunk(content: impl Into<String>) -> Self {
        Self {
            chunk_type: ChunkType::Chunk,
            content: content.into(),
            error: None,
        }
    }

    pub fn done(content: impl Into<String>) -> Self {
        Self {
            chunk_type: ChunkType::Done,
            content: content.into(),
            error: None,
        }
    }

    pub fn error(error: impl Into<String>) -> Self {
        Self {
            chunk_type: ChunkType::Error,
            content: String::new(),
            error: Some(error.into()),
        }
    }
}
