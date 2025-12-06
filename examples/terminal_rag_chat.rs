use std::path::Path;

use nucleus::{ChatManager, Config};
use nucleus_plugin::{Permission, PluginRegistry};


#[tokio::main]
async fn main() {
    // Disable instrusive logs during messaging
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive("nucleus_core=info".parse().unwrap())
                .add_directive("mistralrs_core=off".parse().unwrap())
        )
        .init();

    let config = Config::load_or_default();
    let registry = PluginRegistry::new(Permission::READ_ONLY);

    let manager = ChatManager::new(config, registry).await.unwrap();    
    let doc_count = manager.knowledge_base_count().await;

    println!("Starting with {} docs\n\n", doc_count);

    let home = format!(
        "{}/.cache/huggingface/token",
        dirs::home_dir()
            .ok_or("Home directory missing").unwrap()
            .display()
    );

    let token = std::fs::read_to_string(home).ok();

    println!("HOME: {}", token.unwrap());

    let path = dirs::home_dir()
        .ok_or("Home directory missing").unwrap()
        .join("development/nucleus");
    println!("Path: {}", path.display());
    
    match manager.index_directory(&path).await {
        Ok(_) => {},
        Err(e) => {
            eprintln!("Error indexing directory: {:?}", e);
            std::process::exit(1);
        }
    }
    
    println!("Added {} docs\n\n", manager.knowledge_base_count().await - doc_count);

    let mut input = String::new();

    loop {        
        println!("Enter message: ");

        std::io::stdin().read_line(&mut input).unwrap();
        if input == "exit" || input == "quit" {
            break;
        }

        manager.query_stream(&input, |chunk| {
            print!("{}", chunk);
        }).await.unwrap();

        println!("\n")
    }

    
}
