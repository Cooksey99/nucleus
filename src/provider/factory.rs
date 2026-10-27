//! Provider factory for creating LLM providers based on configuration.

use super::types::*;
use crate::tool::PluginRegistry;
use crate::Config;
use std::sync::Arc;
use tracing::info;

/// Creates a provider instance based on configuration.
///
/// Supported providers:
/// - `"grok"` - xAI Grok (`hosted` feature; account login needs `experimental`)
///
/// Local models use a `local//path.gguf` model string on `AgentBuilder`.
pub async fn create_provider(
    config: &Config,
    registry: Arc<PluginRegistry>,
) -> Result<Arc<dyn Provider>> {
    let _ = registry;
    let provider_type = config.llm.provider.to_lowercase();

    info!("Creating provider: {}", provider_type);

    match provider_type.as_str() {
        "grok" => {
            #[cfg(feature = "hosted")]
            {
                info!("Using Grok provider");
                let provider = super::GrokProvider::new(config)?;
                Ok(Arc::new(provider))
            }
            #[cfg(not(feature = "hosted"))]
            {
                Err(ProviderError::Other(
                    "Grok requires the `hosted` feature".into(),
                ))
            }
        }
        _ => Err(ProviderError::Other(format!(
            "Unknown provider type: {provider_type}. Supported: grok"
        ))),
    }
}
