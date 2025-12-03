use std::sync::Arc;

use nucleus::{ChatManager, Config};
use nucleus_plugin::{Permission, PluginRegistry};

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive("nucleus_core=debug".parse().unwrap())
        )
        .init();

    let config = Config::load_or_default();
    let registry = PluginRegistry::new(Permission::NONE);
    let manager = ChatManager::new(config, Arc::new(registry)).await.unwrap();

    let message = "Hi!".to_string();
    println!("Message: {}", message);

    let response = manager.query(&message).await;

    println!("Response: {}", response.unwrap());
}
