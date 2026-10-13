use nucleus_plugin::{Permission, PluginRegistry};
use nucleus_std::SearchPlugin;

#[tokio::main]
async fn main() {
    println!("Nucleus - SearchPlugin\n");

    let mut registry = PluginRegistry::new(Permission::READ_ONLY);

    if registry.register(SearchPlugin::new()).await {
        println!("SearchPlugin registered\n");
    } else {
        eprintln!("Failed to register plugin (permission denied)");
        return;
    }

    // Test 1: Basic text search in current directory
    println!("Test 1: Search for 'tokio' in current directory");
    let input = serde_json::json!({
        "query": "tokio",
        "max_results": 5
    });

    match registry.execute("search", input).await {
        Ok(output) => {
            println!("Search results:\n{}", output.content);
            println!();
        }
        Err(e) => {
            eprintln!("Error: {}\n", e);
        }
    }

    // Test 2: Case-sensitive regex search
    println!("Test 2: Regex search for function definitions (case-sensitive)");
    let input = serde_json::json!({
        "query": "async fn [a-z_]+",
        "regex": true,
        "case_sensitive": true,
        "max_results": 10
    });

    match registry.execute("search", input).await {
        Ok(output) => {
            println!("Search results:\n{}", output.content);
            println!();
        }
        Err(e) => {
            eprintln!("Error: {}\n", e);
        }
    }

    // Test 3: Search in specific directory with custom exclude patterns
    println!("Test 3: Search in src/ directory");
    let input = serde_json::json!({
        "query": "Plugin",
        "path": "src",
        "max_results": 3
    });

    match registry.execute("search", input).await {
        Ok(output) => {
            println!("Search results:\n{}", output.content);
            println!();
        }
        Err(e) => {
            eprintln!("Error: {}\n", e);
        }
    }

    // Test 4: Search that should return no results
    println!("Test 4: Search for unlikely string");
    let input = serde_json::json!({
        "query": "xyzabc123notfound",
        "max_results": 5
    });

    match registry.execute("search", input).await {
        Ok(output) => {
            println!("Search results:\n{}", output.content);
            println!();
        }
        Err(e) => {
            eprintln!("Error: {}\n", e);
        }
    }

    // Test 5: Show plugin specs (for LLM)
    println!("Test 5: Plugin specifications for LLM");
    let specs = registry.plugin_specs().await;
    println!("{}", serde_json::to_string_pretty(&specs).unwrap());
}
