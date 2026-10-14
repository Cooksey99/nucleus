use clap::{Parser, Subcommand};
use nucleus_core::{grok_login, grok_logout};
use std::error::Error;

#[derive(Debug, Parser)]
#[command(name = "nucleus", about = "Nucleus CLI", arg_required_else_help = true)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Debug, Subcommand)]
enum Commands {
    /// Sign in to a Grok account (device-code OAuth)
    Login,
    /// Clear stored Grok credentials
    Logout,
}

#[tokio::main]
async fn main() {
    let cli = Cli::parse();

    match cli.command {
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
