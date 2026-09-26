# Configuration Reference

In Nucleus, everything is designed to work out of the box. To do this without sacrificing control and customization, we use reasonable defaults **and** a builder pattern.


Take this example, which loads our defaults with minimal setup:
```rust
use nucleus::{ChatManager, Config}
use nucleus_plugin::{Permission, PluginRegistry}

// References the nucles config file if it exists, otherwise, loads nucleus default config
let config = Config::load_or_default();

// Create a plugin registry with no permisisons
let registry = PluginRegistry::new(Permission::READ);
// Add the plugin to allow the LLM to access read file tooling
registry.add(ReadFilePlugin::new());

// Initialize the chat manager
// This sets the config and LLM to be ready for usage
let manager = ChatManager::new(config, registry).await?;

// Ask a question. The next query on this manager continues the conversation.
let response = manager.query("Summarize the README.md for me").await?;
println!("{}", response);
```


What if we didn't want to use the default LLM?

We can use the builder to specific this:
```rust
let manager = manager::builder(config, registry)
  .with_llm_model("Qwen/Qwen3-8B")
  .build().await.unwrap();
```

For more about the `ChatManager` builder methods, reference: **TBD**
