use std::collections::HashMap;

/// A document stored in the vector database.
///
/// Documents are the fundamental unit of storage in the RAG system. Each document
/// contains the original text content, its vector embedding for similarity search,
/// and optional metadata for tracking source information.
///
/// # Example
///
/// ```no_run
/// use std::collections::HashMap;
/// # use nucleus_core::rag::Document;
///
/// let embedding = vec![0.1, 0.2, 0.3];
/// let doc = Document::new("doc_1", "Hello world", embedding)
///     .with_metadata("source", "user_input")
///     .with_metadata("timestamp", "2024-01-01");
/// ```
#[derive(Debug, Clone)]
pub struct Document {
    pub id: String,
    pub content: String,
    pub embedding: Vec<f32>,
    pub metadata: HashMap<String, String>,
}

impl Document {
    pub fn new(
        id: impl Into<String>,
        content: impl Into<String>,
        embedding: Vec<f32>,
    ) -> Self {
        Self {
            id: id.into(),
            content: content.into(),
            embedding,
            metadata: HashMap::new(),
        }
    }
    
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }
}

/// A search result containing a document and its similarity score.
///
/// Returned by vector search operations, ordered by descending similarity score.
/// Higher scores indicate better matches to the query.
///
/// # Score Range
///
/// Similarity scores typically range from -1.0 to 1.0 when using cosine similarity:
/// - `1.0` - Identical vectors (perfect match)
/// - `0.0` - Orthogonal vectors (no similarity)
/// - `-1.0` - Opposite vectors (completely dissimilar)
///
/// In practice, most scores will be between 0.0 and 1.0 for text embeddings.
#[derive(Debug, Clone)]
pub struct SearchResult {
    pub document: Document,
    pub score: f32,
}
