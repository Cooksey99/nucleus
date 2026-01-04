use std::{io::{self, Write}};

use nucleus::{provider::ProviderType, ChatManager};

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive("nucleus_core=info".parse().unwrap())
        )
        .init();

    let manager = ChatManager::builder()
        .with_provider(ProviderType::CoreML)
        .with_llm_model("apple/mistral-coreml")
        .build()
        .await
        .expect("Failed to create chat manager");

    let message = "Write me a short poem about Rust programming";
    println!("User: {}", message);
    println!("\nAssistant: ");
    io::stdout().flush().unwrap();

    // Stream response with live printing
    let start = std::time::Instant::now();
    let mut token_count = 0;
    let response = manager
        .query_stream(message, |chunk| {
            print!("{}", chunk);
            io::stdout().flush().unwrap();
            // Rough token estimation: ~4 chars per token
            token_count += chunk.len() / 4;
        })
        .await
        .expect("Failed to get response");
    let elapsed = start.elapsed();

    println!("\n\n--- Stream complete ---");
    println!("Total length: {} characters", response.len());
    println!("Time elapsed: {:.2}s", elapsed.as_secs_f64());
    println!("Est. tokens: {}", token_count);
    println!("Est. throughput: {:.1} tok/s", token_count as f64 / elapsed.as_secs_f64());
}
