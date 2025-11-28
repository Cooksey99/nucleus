use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use colored::Colorize;
use nucleus_core::config::Config;
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "nucleus")]
#[command(about = "CLI for managing nucleus AI engine configuration", long_about = None)]
#[command(version)]
struct Cli {
    #[arg(short, long, default_value = "config.yaml")]
    config: PathBuf,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    #[command(about = "Show current configuration")]
    Show,

    #[command(about = "Model management commands")]
    Model {
        #[command(subcommand)]
        command: ModelCommands,
    },
}

#[derive(Subcommand)]
enum ModelCommands {
    #[command(about = "Show current model")]
    Show,

    #[command(about = "Set the LLM model")]
    Set {
        #[arg(help = "Model name (e.g., 'llama3.2:latest' or 'hf.co/Qwen/Qwen3-30B-A3B-GGUF:Q4_K_M')")]
        model: String,
    },

    #[command(about = "List available models from Ollama")]
    List {
        #[arg(short, long, default_value = "http://localhost:11434")]
        url: String,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Show => show_config(&cli.config),
        Commands::Model { command } => match command {
            ModelCommands::Show => show_model(&cli.config),
            ModelCommands::Set { model } => set_model(&cli.config, &model),
            ModelCommands::List { url } => list_models(&url),
        },
    }
}

fn show_config(config_path: &PathBuf) -> Result<()> {
    let config = Config::load(config_path)
        .context("Failed to load config")?;

    println!("{}", "Current Configuration:".bold().green());
    println!();
    println!("{}", "LLM:".bold());
    println!("  Model:          {}", config.llm.model.cyan());
    println!("  Base URL:       {}", config.llm.base_url);
    println!("  Temperature:    {}", config.llm.temperature);
    println!("  Context Length: {}", config.llm.context_length);
    println!();
    println!("{}", "RAG:".bold());
    println!("  Embedding Model: {}", config.rag.embedding_model.cyan());
    println!("  Chunk Size:      {}", config.rag.chunk_size);
    println!("  Top K:           {}", config.rag.top_k);
    println!();
    println!("{}", "Storage:".bold());
    println!("  Vector DB:       {}", config.storage.vector_db_path);
    println!("  Chat History:    {}", config.storage.chat_history_path);

    Ok(())
}

fn show_model(config_path: &PathBuf) -> Result<()> {
    let config = Config::load(config_path)
        .context("Failed to load config")?;

    println!("{}: {}", "Current model".bold(), config.llm.model.cyan());
    Ok(())
}

fn set_model(config_path: &PathBuf, model: &str) -> Result<()> {
    let content = std::fs::read_to_string(config_path)
        .context("Failed to read config file")?;

    let mut config: serde_yaml::Value = serde_yaml::from_str(&content)
        .context("Failed to parse config")?;

    if let Some(llm) = config.get_mut("llm") {
        if let Some(llm_map) = llm.as_mapping_mut() {
            llm_map.insert(
                serde_yaml::Value::String("model".to_string()),
                serde_yaml::Value::String(model.to_string()),
            );
        }
    }

    let updated_content = serde_yaml::to_string(&config)
        .context("Failed to serialize config")?;

    std::fs::write(config_path, updated_content)
        .context("Failed to write config file")?;

    println!(
        "{} Model updated to: {}",
        "✓".green().bold(),
        model.cyan()
    );

    Ok(())
}

fn list_models(base_url: &str) -> Result<()> {
    use reqwest::blocking::Client;
    use serde::Deserialize;

    #[derive(Deserialize)]
    struct OllamaModel {
        name: String,
        #[allow(dead_code)]
        modified_at: String,
        size: u64,
    }

    #[derive(Deserialize)]
    struct OllamaResponse {
        models: Vec<OllamaModel>,
    }

    let client = Client::new();
    let url = format!("{}/api/tags", base_url);

    println!("{} Fetching models from {}...", "→".blue(), base_url);
    println!();

    let response = client
        .get(&url)
        .send()
        .context("Failed to connect to Ollama. Is it running?")?;

    if !response.status().is_success() {
        anyhow::bail!("Ollama returned error: {}", response.status());
    }

    let data: OllamaResponse = response
        .json()
        .context("Failed to parse Ollama response")?;

    if data.models.is_empty() {
        println!("{}", "No models found. Pull a model with 'ollama pull <model>'".yellow());
        return Ok(());
    }

    println!("{}", "Available models:".bold().green());
    println!();

    for model in data.models {
        let size_gb = model.size as f64 / (1024.0 * 1024.0 * 1024.0);
        println!(
            "  {} {} ({:.2} GB)",
            "•".cyan(),
            model.name.bold(),
            size_gb
        );
    }

    println!();
    println!("Use {} to set a model", "nucleus -c config.yaml model set <model>".bold());

    Ok(())
}
