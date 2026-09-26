use std::io::{self, Write};

use nucleus::{ChatManager, Config};
use nucleus_plugin::{Permission, PluginRegistry};

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive("nucleus_core=info".parse().unwrap()),
        )
        .init();

    let config = Config::load_or_default();
    let registry = PluginRegistry::new(Permission::NONE);
    let manager = ChatManager::new(config, registry)
        .await
        .expect("Failed to create chat manager");

    loop {
        println!("Enter message (reset to clear history):");
        std::io::stdout().flush().unwrap();

        let mut message = String::new();
        std::io::stdin().read_line(&mut message).expect("Unable to read line");
        let message = message.trim();
        if message.is_empty() {
            continue;
        }
        if message.eq_ignore_ascii_case("reset") {
            manager.clear_history().await;
            println!("History cleared.\n");
            continue;
        }

    // Stream response with live printing. Follow-ups continue the same conversation.
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
    println!(
        "Est. throughput: {:.1} tok/s",
        token_count as f64 / elapsed.as_secs_f64()
    );
    }

}
