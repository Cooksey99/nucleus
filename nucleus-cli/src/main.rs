use clap::{Parser, Subcommand};

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
    Serve,
    /// Start the endpoint in the background (implementation pending)
    Start,
    /// Stop the background endpoint (implementation pending)
    Stop,
    /// Show endpoint status (implementation pending)
    Status,
}

fn main() {
    let cli = Cli::parse();

    match cli.command {
        Commands::Serve => {
            println!("serve: not implemented yet");
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
    }
}
