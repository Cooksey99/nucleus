use clap::{Args, Parser, Subcommand};
use std::io;
use std::net::TcpListener;

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
}

#[derive(Debug, Args)]
struct ServeArgs {
    /// Address to bind as host:port
    #[arg(long, default_value = "0.0.0.0:8443")]
    listen: String,
}

fn main() {
    let cli = Cli::parse();

    match cli.command {
        Commands::Serve(args) => {
            if let Err(err) = run_serve(args) {
                eprint!("serve failed: {err}");
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
    }
}

fn run_serve(args: ServeArgs) -> io::Result<()> {
    let listener = TcpListener::bind(&args.listen)?;
    println!("Nucleus listening on {}", listener.local_addr()?);

    for incoming in listener.incoming() {
        match incoming {
            Ok(_stream) => {}
            Err(err) => {
                eprint!("Error: {err}")
            }
        }
    }

    Ok(())
}
