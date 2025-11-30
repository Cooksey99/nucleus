use nucleus_plugin::{Plugin, PluginError, PluginOutput, Permission, Result};
use async_trait::async_trait;
use serde::Deserialize;
use serde_json::Value;


pub struct WebSearchPlugin;

#[derive(Debug, Deserialize)]
struct WebSearchParams {
    query: String,
    max_results: Option<usize>,
}

impl WebSearchPlugin {
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl Plugin for WebSearchPlugin {
    fn name(&self) -> &str {
        "web_search"
    }

    fn description(&self) -> &str {
        "Search the web using DuckDuckGo"
    }

    fn parameter_schema(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "required": ["query", "max_results"],
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query"
                },
                "max_results": {
                    "type": "number",
                    "description": "The maximum number of results to return"
                }
            }
        })
    }

    fn required_permission(&self) -> Permission {
        Permission::READ_ONLY
    }

    async fn execute(&self, input: Value) -> Result<PluginOutput> {
        let params: WebSearchParams = serde_json::from_value(input)
            .map_err(|e| PluginError::InvalidInput(format!("Invalid parameters: {}", e)))?;
        let max = params.max_results;
        
        // TODO: Implement web search
        Ok(PluginOutput::new(String::from("Web search results")))
    }
}


#[tokio::test]
async fn test_web_search_execute() {
    let plugin = WebSearchPlugin::new();
    let input = serde_json::json!({ "query": "Rust programming", "max_results": 1 });

    let result = plugin.execute(input).await.unwrap();
    println!("Web search results:\n{}", result.content);
}



