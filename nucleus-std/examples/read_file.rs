use nucleus_plugin::{Plugin, PluginRegistry, Permission};
use nucleus_std::ReadFilePlugin;
use std::sync::Arc;

#[tokio::main]
async fn main() {
    println!("Nucleus - ReadFilePlugin\n");

    // Create plugin registry with read permissions
    let mut registry = PluginRegistry::new(Permission::READ_ONLY);
    
    // Register the ReadFilePlugin
    let read_plugin = Arc::new(ReadFilePlugin::new());
    if registry.register(read_plugin) {
        println!("ReadFilePlugin registered\n");
    } else {
        eprintln!("Failed to register plugin (permission denied)");
        return;
    }

    // Test 1: Read the Cargo.toml file
    println!("Test 1: Reading Cargo.toml");
    let input = serde_json::json!({
        "path": "Cargo.toml"
    });
    
    match registry.execute("read_file", input).await {
        Ok(output) => {
            println!("Success!");
            println!("Content preview (first 200 chars):");
            println!("{}", &output.content.chars().take(200).collect::<String>());
            println!("...\n");
        }
        Err(e) => {
            eprintln!("Error: {}\n", e);
        }
    }

    // Test 2: Try to read a non-existent file
    println!("Test 2: Reading non-existent file");
    let input = serde_json::json!({
        "path": "/nonexistent/file.txt"
    });
    
    match registry.execute("read_file", input).await {
        Ok(_) => {
            println!("This shouldn't succeed!\n");
        }
        Err(e) => {
            println!("Expected error: {}\n", e);
        }
    }

    // Test 3: Show plugin specs (for LLM)
    println!("Test 3: Plugin specifications for LLM");
    let specs = registry.plugin_specs();
    println!("{}", serde_json::to_string_pretty(&specs).unwrap());
}
