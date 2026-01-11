// The initial indexing in this example can take a few minutes

use nucleus::{ChatManager, Config};
use nucleus_plugin::{Permission, PluginRegistry};

#[tokio::main]
async fn main() {
    // Enable detailed logs during indexing
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive("nucleus_core=off".parse().unwrap())
                .add_directive("mistralrs_core=info".parse().unwrap()),
        )
        .init();

    let config = Config::load_or_default();
    let registry = PluginRegistry::new(Permission::READ_ONLY);

    let manager = ChatManager::builder(config, registry)
        .with_llm_model("Qwen/Qwen3-8B")
        .build()
        .await
        .unwrap();
    let doc_count = manager.knowledge_base_count().await;

    println!("Starting with {} docs\n\n", doc_count);

    let home = format!(
        "{}/.cache/huggingface/token",
        dirs::home_dir()
            .ok_or("Home directory missing")
            .unwrap()
            .display()
    );

    let token = std::fs::read_to_string(home).ok();

    println!("HOME: {}", token.unwrap());

    let path = dirs::home_dir()
        .ok_or("Home directory missing")
        .unwrap()
        .join("development/nucleus/nucleus-core/src");
    println!("Path: {}", path.display());

    match manager.index_directory(&path).await {
        Ok(_) => {}
        Err(e) => {
            eprintln!("Error indexing directory: {:?}", e);
            std::process::exit(1);
        }
    }

    println!(
        "Added {} docs\n\n",
        manager.knowledge_base_count().await - doc_count
    );

    let mut input = String::new();

    loop {
        println!("Enter message: ");

        std::io::stdin().read_line(&mut input).unwrap();
        if input == "exit" || input == "quit" {
            break;
        }

        manager
            .query_stream(&input, |chunk| {
                print!("{}", chunk);
            })
            .await
            .unwrap();

        println!("\n")
    }
}
