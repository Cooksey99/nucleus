mod core;
mod utils;

use anyhow::Result;
use std::os::fd::AsRawFd;

use core::{run_io_loop, ServerManager};
use portable_pty::{native_pty_system, CommandBuilder};
use utils::{get_terminal_size, set_nonblocking, set_raw_mode, RestoreTermios};

fn main() -> Result<()> {
    ServerManager::ensure_server()?;

    let shell = std::env::var("SHELL").unwrap();

    println!("Starting PTY wrapper with shell: {}", shell);
    println!("AI commands available: /ai, /edit, /add, /index, /stats");

    let pty_system = native_pty_system();

    // Open PTY with current terminal size
    let terminal_size = get_terminal_size();
    let pair = pty_system.openpty(terminal_size)?;

    // Spawn shell as interactive shell with explicit .zshrc sourcing
    let mut cmd = CommandBuilder::new(&shell);
    cmd.arg("-i");  // Interactive shell
    cmd.arg("-c");  // Execute command
    cmd.arg("source ~/.zshrc 2>/dev/null; exec zsh -i");  // Source config then start interactive shell
    // Ensure proper environment for shell initialization
    cmd.env("TERM", "xterm-256color");
    let _child = pair.slave.spawn_command(cmd).unwrap();

    // Slave isn't needed, so it can be dropped
    drop(pair.slave);
    let mut master = pair.master;

    let stdin_fd = std::io::stdin().as_raw_fd();
    let original_termios = set_raw_mode(stdin_fd)?;
    set_nonblocking(stdin_fd)?;
    set_nonblocking(master.as_raw_fd().expect("Failed to unwrap Master"))?;

    // Restore terminal on exit
    let _guard = RestoreTermios {
        fd: stdin_fd,
        termios: original_termios,
    };

    run_io_loop(&mut master)
}
