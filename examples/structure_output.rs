use nucleus::{ChatManager, ChatManagerBuilder, Config};
use nucleus_plugin::{Permission, PluginRegistry};

#[tokio::main]
async fn main() {

    let config = Config::load_or_default();
    let registry = PluginRegistry::new(Permission::NONE);

    let chat_manager = ChatManager::new(config, registry).await.unwrap();

}
