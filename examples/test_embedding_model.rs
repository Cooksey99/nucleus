use mistralrs::{EmbeddingModelBuilder, Model};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    println!("Testing nomic-embed-text-v1.5 with mistralrs...");
    
    let model_path = "./models/nomic-embed-text-v1.5";
    
    println!("Loading model from: {}", model_path);
    let model = EmbeddingModelBuilder::new(model_path)
        .with_logging()
        .build()
        .await?;
    
    println!("Model loaded successfully!");
    
    let test_text = "Hello, this is a test embedding";
    println!("Generating embedding for: {}", test_text);
    
    let embedding = model.generate_embedding(test_text).await?;
    
    println!("Embedding dimension: {}", embedding.len());
    println!("First 5 values: {:?}", &embedding[..5.min(embedding.len())]);
    
    Ok(())
}
