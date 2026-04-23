use nucleus::{ChatManager, ChatManagerBuilder, Config};
use nucleus_plugin::{Permission, PluginRegistry};
use serde_json::json;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Load configuration
    let config = Config::load_or_default();
    let registry = PluginRegistry::new(Permission::NONE);

    // Create chat manager using builder pattern
    let mut chat_manager = ChatManagerBuilder::new()
        .with_config(config)
        .with_registry(registry)
        .build()
        .await?;

    // Define a JSON schema for structured output
    let person_schema = json!({
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "The person's full name"
            },
            "age": {
                "type": "number",
                "description": "The person's age in years"
            },
            "occupation": {
                "type": "string",
                "description": "The person's job or profession"
            },
            "hobbies": {
                "type": "array",
                "items": {
                    "type": "string"
                },
                "description": "List of hobbies or interests"
            }
        },
        "required": ["name", "age", "occupation"]
    });

    // Set structured output with schema and description
    chat_manager.set_structured_output(person_schema.clone());

    println!("Testing structured output with mistral.rs provider");
    println!("Schema: {}", serde_json::to_string_pretty(&person_schema)?);
    println!("\nAsking LLM to generate structured data...\n");

    let response = chat_manager
        .query(None, "Create a profile for a fictional software engineer named Alice who works at a tech startup.")
        .await?;

    println!("Raw response:\n{}", response);
    println!("DONE");

    // Try to parse the response as JSON (should be raw JSON now)
    match serde_json::from_str::<serde_json::Value>(&response) {
        Ok(parsed_json) => {
            println!("Successfully parsed as JSON:");
            println!("{}", serde_json::to_string_pretty(&parsed_json)?);
        }
        Err(e) => {
            println!("Failed to parse as JSON: {}", e);
            println!("Response was:\n{}", response);
        }
    }

    // Clear structured output for normal conversation
    chat_manager.clear_structured_output();

    println!("Now testing normal conversation (no structured output):");
    let normal_response = chat_manager
        .query(None, "What's the capital of Japan?")
        .await?;

    println!("{}", normal_response);

    Ok(())
}
