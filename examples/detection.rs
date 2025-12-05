//! Example demonstrating Ollama detection when creating a nucleus server.
//!
//! This example shows what happens when you try to initialize nucleus
//! without Ollama installed or running.

use nucleus_core::{Config, Server};

#[tokio::main]
async fn main() {
    let config = Config::default();
    
    match Server::new(config).await {
        Ok(server) => {
            println!("Server initialized successfully!");
            println!("Ollama is installed and running.");
            
            if let Err(e) = server.start().await {
                eprintln!("Server error: {}", e);
            }
        }
        Err(e) => {
            eprintln!();
            eprintln!("Failed to initialize nucleus server:");
            eprintln!("{}", e);
            eprintln!();
            std::process::exit(1);
        }
    }
}
