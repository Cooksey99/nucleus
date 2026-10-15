//! Provider factory for creating LLM providers based on configuration.

use super::types::*;
use super::MistralRsProvider;
use crate::{provider::GrokProvider, Config};
use crate::tool::PluginRegistry;
use std::sync::Arc;
use tracing::info;

/// Creates a provider instance based on configuration.
///
/// Supported providers:
/// - `"mistralrs"` - mistral.rs in-process provider
/// - `"grok"` - xAI Grok (account OAuth or API key)
pub async fn create_provider(
    config: &Config,
    registry: Arc<PluginRegistry>,
) -> Result<Arc<dyn Provider>> {
    let provider_type = config.llm.provider.to_lowercase();

    info!("Creating provider: {}", provider_type);

    match provider_type.as_str() {
        "mistralrs" => {
            info!("Using mistral.rs provider with model: {}", config.llm.model);
            let provider = MistralRsProvider::new(config, registry).await?;
            Ok(Arc::new(provider))
        }
        "grok" => {
            info!("Using Grok provider");
            let provider = GrokProvider::new(config)?;
            Ok(Arc::new(provider))
        }
        _ => Err(ProviderError::Other(format!(
            "Unknown provider type: {provider_type}. Supported: mistralrs, grok"
        ))),
    }
}
