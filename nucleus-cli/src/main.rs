use clap::{Args, Parser, Subcommand};
use nucleus_core::{grok_login, grok_logout, ChatManager, Config};
use nucleus_plugin::{Permission, PluginRegistry};
use std::error::Error;
use std::fs::read_dir;
use std::io;
use std::net::TcpListener;
use std::path::PathBuf;

#[derive(Debug, Parser)]
#[command(
    name = "nucleus",
    about = "Nucleus local model endpoint CLI",
    arg_required_else_help = true
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Debug, Subcommand)]
enum Commands {
    /// Run the endpoint in the foreground (implementation pending)
    Serve(ServeArgs),
    /// Start the endpoint in the background (implementation pending)
    Start,
    /// Stop the background endpoint (implementation pending)
    Stop,
    /// Show endpoint status (implementation pending)
    Status,
    /// List models downloaded in Nucleus
    List,
    /// Sign in to a Grok account (device-code OAuth)
    Login,
    /// Clear stored Grok credentials
    Logout,
}

#[derive(Debug, Args)]
struct ServeArgs {
    /// Address to bind as host:port
    #[arg(long, default_value = "0.0.0.0:8443")]
    listen: String,
    /// Model identifier
    #[arg(long, short = 'm', required = true)]
    models: Vec<String>,
}

#[tokio::main]
async fn main() {
    let cli = Cli::parse();

    match cli.command {
        Commands::Serve(args) => {
            if let Err(err) = run_serve(args).await {
                eprintln!("serve failed: {err}");
                std::process::exit(1);
            }
        }
        Commands::Start => {
            println!("start: not implemented yet");
        }
        Commands::Stop => {
            println!("stop: not implemented yet");
        }
        Commands::Status => {
            println!("status: not implemented yet");
        }
Commands::List => {
            if let Err(err) = run_list() {
                eprintln!("list failed: {err}");
                std::process::exit(1);
            }
        }
        Commands::Login => {
            if let Err(err) = run_login().await {
                eprintln!("login failed: {err}");
                std::process::exit(1);
            }
        }
        Commands::Logout => {
            if let Err(err) = run_logout() {
                eprintln!("logout failed: {err}");
                std::process::exit(1);
            }
        }
    }
}

async fn run_login() -> Result<(), Box<dyn Error>> {
    grok_login(|url, code| {
        println!("Open: {url}");
        println!("Code: {code}");
        println!("Waiting for approval...");
    })
    .await?;
    println!("Logged in (~/.nucleus/grok-auth.json)");
    Ok(())
}

fn run_logout() -> Result<(), Box<dyn Error>> {
    grok_logout()?;
    println!("Logged out.");
    Ok(())
}

async fn run_serve(args: ServeArgs) -> Result<(), Box<dyn Error>> {
    let mut managers = Vec::new();

    for model in args.models {
        println!("model: {}", model);
        let config = Config::new().with_model(model);
        let registry = PluginRegistry::new(Permission::READ_ONLY);
        let manager = ChatManager::new(config, registry).await?;
        managers.push(manager);
    }

    let listener = TcpListener::bind(&args.listen).map_err(io::Error::other)?;
    println!("Nucleus listening on {}", listener.local_addr()?);
    println!("Initialized {} model manager(s)", managers.len());

    for incoming in listener.incoming() {
        match incoming {
            Ok(_stream) => {}
            Err(err) => {
                eprintln!("Error: {err}")
            }
        }
    }

    Ok(())
}

fn run_list() -> Result<(), Box<dyn Error>> {
    let root = PathBuf::from("models");
    
    read_dir(root)?.for_each(|item| {
        if let Ok(val) = item {
        
            let name = val.file_name().display().to_string();
            if !name.starts_with(".") {
                println!("{}", name);
            }
        }
    });
    Ok(())
}
