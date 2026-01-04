use nucleus_core::{Config, provider::coreml::CoreMLProvider};
use nucleus_plugin::{Permission, PluginRegistry};
use std::sync::Arc;

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive("nucleus_core=info".parse().unwrap())
        )
        .init();

    let mut config = Config::default();
    config.llm.model = "../models/Llama-3.1-8B-Instruct-CoreML/llama_3.1_coreml.mlpackage".to_string();
    
    let registry = PluginRegistry::new(Permission::NONE);
    
    println!("Loading CoreML model...");
    let provider = CoreMLProvider::new(&config, registry)
        .await
        .expect("Failed to create CoreML provider");
    
    println!("Model loaded successfully!");
    
    // Get input shape
    match provider.get_input_shape(4) {
        Ok(shape) => {
            println!("Input shape: {:?}", shape);
            
            // Create dummy input with correct shape
            let input_size: usize = shape.iter().map(|&d| d as usize).product();
            println!("Input size: {}", input_size);
            
            if input_size > 0 && input_size < 10000000 {
                let input = vec![0.5f32; input_size];
                let mut output = vec![0.0f32; input_size];
                
                println!("Running prediction...");
                match provider.predict(&input, &mut output) {
                    Ok(_) => {
                        println!("Prediction succeeded!");
                        println!("Output sample (first 10 values): {:?}", &output[..10.min(output.len())]);
                    }
                    Err(e) => {
                        eprintln!("Prediction failed: {}", e);
                    }
                }
            } else {
                println!("Input size is too large or invalid for testing");
            }
        }
        Err(e) => {
            eprintln!("Failed to get input shape: {}", e);
        }
    }
}
