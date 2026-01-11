//! Provider factory for creating LLM providers based on configuration.

use super::types::*;
#[cfg(any(target_os = "macos", feature = "coreml"))]
use super::CoreMLProvider;
use super::{MistralRsProvider, OllamaProvider};
use crate::Config;
use nucleus_plugin::PluginRegistry;
use std::sync::Arc;
use tracing::info;

/// Creates a provider instance based on configuration.
///
/// Supported providers:
/// - `"ollama"` - Ollama API provider
/// - `"mistralrs"` - mistral.rs in-process provider
/// - `"coreml"` - CoreML inference (macOS only, requires `coreml` feature)
pub async fn create_provider(
    config: &Config,
    registry: Arc<PluginRegistry>,
) -> Result<Arc<dyn Provider>> {
    let provider_type = config.llm.provider.to_lowercase();

    info!("Creating provider: {}", provider_type);

    match provider_type.as_str() {
        "ollama" => {
            info!("Using Ollama provider at {}", config.llm.base_url);
            Ok(Arc::new(OllamaProvider::new(config)))
        }
        "mistralrs" => {
            info!("Using mistral.rs provider with model: {}", config.llm.model);
            let provider = MistralRsProvider::new(config, registry).await?;
            Ok(Arc::new(provider))
        }
        #[cfg(any(target_os = "macos", feature = "coreml"))]
        "coreml" => {
            info!("Using CoreML provider with model: {}", config.llm.model);
            let provider = CoreMLProvider::new(config, registry).await?;
            Ok(provider)
        }
        #[cfg(not(any(target_os = "macos", feature = "coreml")))]
        "coreml" => Err(ProviderError::Other(
            "CoreML provider is only available on macOS".to_string(),
        )),
        _ => Err(ProviderError::Other(format!(
            "Unknown provider type: {}. Supported: ollama, mistralrs, coreml",
            provider_type
        ))),
    }
}
