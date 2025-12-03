//! mistral.rs provider implementation (placeholder).
//!
//! TODO: Implement the Provider trait for mistral.rs

use super::types::*;
use async_trait::async_trait;

/// mistral.rs provider (not yet implemented).
pub struct MistralRsProvider {
    // TODO: Add fields for mistral.rs model
}

impl MistralRsProvider {
    pub fn new() -> Result<Self> {
        Err(ProviderError::Other("MistralRsProvider not yet implemented".to_string()))
    }
}

#[async_trait]
impl Provider for MistralRsProvider {
    async fn chat(
        &self,
        _request: ChatRequest,
        _callback: impl FnMut(ChatResponse) + Send,
    ) -> Result<()> {
        Err(ProviderError::Other("MistralRsProvider not yet implemented".to_string()))
    }
    
    async fn embed(&self, _text: &str, _model: &str) -> Result<Vec<f32>> {
        Err(ProviderError::Other("MistralRsProvider not yet implemented".to_string()))
    }
}