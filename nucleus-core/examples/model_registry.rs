use nucleus_core::models::{Model, ModelRegistry};

fn main() {
    let registry = ModelRegistry::new();

    println!("=== All Models ===");
    for model in registry.all_models() {
        match model {
            Model::Chat(chat) => {
                println!(
                    "Chat: {} - {} tokens, temp: {}",
                    chat.name, chat.context_length, chat.default_temperature
                );
            }
            Model::Embedding(embed) => {
                println!(
                    "Embedding: {} - {} dims, {} tokens",
                    embed.name, embed.embedding_dim, embed.context_length,
                );
            }
        }
    }

    println!("\n=== Get Specific Model ===");
    if let Some(embed) = registry.get_embedding("qwen3-embedding-0.6b") {
        println!("Found: {}", embed.name);
        println!("HuggingFace: {}", embed.hf_repo);
        println!("Embedding dimension: {}", embed.embedding_dim);
        println!("Context length: {}", embed.context_length);
    }

    println!("\n=== Embedding Models Only ===");
    for embed in registry.embedding_models() {
        println!("{} ({})", embed.name, embed.id);
    }
}
