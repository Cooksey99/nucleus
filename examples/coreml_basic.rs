use nucleus_core::{Config, provider::{coreml::CoreMLProvider, ChatRequest, Message}};
use nucleus_plugin::{Permission, PluginRegistry};

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive("nucleus_core=info".parse().unwrap())
        )
        .init();

    let mut config = Config::default();
    config.llm.model = "models/Llama-3.1-8B-Instruct-CoreML/llama_3.1_coreml.mlpackage".to_string();
    
    let registry = PluginRegistry::new(Permission::NONE);
    
    println!("Loading CoreML model...");
    let provider = CoreMLProvider::new(&config, registry)
        .await
        .expect("Failed to create CoreML provider");
    
    println!("Model loaded successfully!");
    
    let messages = vec![
        Message {
            role: "user".to_string(),
            content: "What is 2+2?".to_string(),
            context: None,
        },
    ];
    
    let request = ChatRequest {
        messages,
        temperature: 0.0,
        top_p: 0.9,
        max_tokens: Some(100),
        stop_sequences: None,
    };
    
    println!("Running chat inference...");
    
    use nucleus_core::provider::Provider;
    let result = provider
        .chat(request, Box::new(|response| {
            if let Some(token) = response.content {
                print!("{}", token);
            }
        }))
        .await;
    
    println!();
    
    match result {
        Ok(_) => println!("\nChat completed successfully!"),
        Err(e) => eprintln!("\nChat failed: {}", e),
    }
}
