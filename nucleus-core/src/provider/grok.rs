use async_trait::async_trait;

use crate::{Config, Provider};

pub struct GrokProvider {
    base_url: String,
    api_key: String,
    http_client: reqwest::Client,
    model: String,
}

impl GrokProvider {
    pub fn new(config: &Config) -> Result<Self> {
        // let api_key = config.llm.api_key
    }
}

#[async_trait]
impl Provider for GrokProvider {
    async fn chat() {}
    async fn embed() {}
}
