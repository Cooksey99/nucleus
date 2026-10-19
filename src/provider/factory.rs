//! Provider factory for creating LLM providers based on configuration.

use super::types::*;
use crate::tool::PluginRegistry;
use crate::Config;
use std::sync::Arc;
use tracing::info;

/// Creates a provider instance based on configuration.
///
/// Supported providers:
/// - `"mistralrs"` - mistral.rs in-process provider (`local` feature)
/// - `"grok"` - xAI Grok (`hosted` feature; account login needs `experimental`)
pub async fn create_provider(
    config: &Config,
    registry: Arc<PluginRegistry>,
) -> Result<Arc<dyn Provider>> {
    let provider_type = config.llm.provider.to_lowercase();

    info!("Creating provider: {}", provider_type);

    match provider_type.as_str() {
        "mistralrs" => {
            #[cfg(feature = "local")]
            {
                info!("Using mistral.rs provider with model: {}", config.llm.model);
                let provider = super::MistralRsProvider::new(config, registry).await?;
                Ok(Arc::new(provider))
            }
            #[cfg(not(feature = "local"))]
            {
                let _ = registry;
                Err(ProviderError::Other(
                    "mistral.rs requires the `local` feature".into(),
                ))
            }
        }
        "grok" => {
            #[cfg(feature = "hosted")]
            {
                info!("Using Grok provider");
                let provider = super::GrokProvider::new(config)?;
                Ok(Arc::new(provider))
            }
            #[cfg(not(feature = "hosted"))]
            {
                let _ = (config, registry);
                Err(ProviderError::Other(
                    "Grok requires the `hosted` feature".into(),
                ))
            }
        }
        _ => Err(ProviderError::Other(format!(
            "Unknown provider type: {provider_type}. Supported: mistralrs, grok"
        ))),
    }
}
