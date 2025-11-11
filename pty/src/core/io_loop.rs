use super::command::Command;
use anyhow::Result;
use std::io::{Read, Write};

enum Mode {
    Terminal,
    AI,
}

impl Mode {
    fn prefix(&self) -> &str {
        match self {
            Mode::Terminal => "[$]",
            Mode::AI => "[AI]",
        }
    }
}

struct Session {
    mode: Mode,
    line_buffer: String,
}

impl Session {
    fn new() -> Self {
        Self {
            mode: Mode::Terminal,
            line_buffer: String::new(),
        }
    }

    fn toggle_mode(&mut self) {
        self.mode = match self.mode {
            Mode::Terminal => Mode::AI,
            Mode::AI => Mode::Terminal,
        };
    }
    fn show_mode_indicator(&self, writer: &mut dyn Write) -> Result<()> {
        // Send the prompt update to the shell
        // Configured for zsh
        let prompt_command = match self.mode {
            Mode::Terminal => "export PS1='%F{green}[$]%f '",
            Mode::AI => "export PS1='%F{magenta}[AI]%f '",
        };

        // Send command + newline to execute it
        writer.write_all(prompt_command.as_bytes())?;
        writer.write_all(b"\n")?;

        // Send 'clear' to refresh, or just newline
        writer.write_all(b"clear\n")?;
        writer.flush()?;
        Ok(())
    }
}

const CTRL_I: u8 = 9;
// const CMD_I: u8 = 25;

pub fn run_io_loop(master: &mut Box<dyn portable_pty::MasterPty + Send>) -> Result<()> {
    use std::io::{stdin, stdout};

    let mut stdin = stdin();
    let mut stdout = stdout();

    let mut reader = master.try_clone_reader()?;
    let mut writer = master.take_writer()?;

    let mut stdin_buf = [0u8; 4096];
    let mut pty_buf = [0u8; 4096];
    let mut session = Session::new();
    session.show_mode_indicator(&mut writer)?;

    loop {
        match stdin.read(&mut stdin_buf) {
            Ok(0) => break,
            Ok(n) => {
                if n == 1 && stdin_buf[0] == CTRL_I {
                    session.toggle_mode();
                    continue;
                }

                let input = String::from_utf8_lossy(&stdin_buf[..n]);

                for ch in input.chars() {
                    session.line_buffer.push(ch);

                    if ch == '\n' || ch == '\r' {
                        let line = session.line_buffer.trim_end();

                        let cmd_text = match session.mode {
                            Mode::AI => {
                                if line.starts_with('/') {
                                    line.to_string()
                                } else {
                                    format!("/ai {}", line)
                                }
                            }
                            Mode::Terminal => line.to_string(),
                        };

                        let cmd = Command::parse(&cmd_text);

                        match cmd {
                            Command::PassThrough => {
                                writer.write_all(session.line_buffer.as_bytes())?;
                                writer.flush()?;
                                session.line_buffer.clear();
                            }
                            _ => {
                                stdout.write_all(b"\r\n")?;
                                stdout.flush()?;

                                match cmd.execute() {
                                    Ok(Some(response)) => {
                                        stdout.write_all(response.as_bytes())?;
                                        stdout.write_all(b"\r\n")?;
                                    }
                                    Ok(None) => {}
                                    Err(e) => {
                                        let error_msg = format!("Error: {}\r\n", e);
                                        stdout.write_all(error_msg.as_bytes())?;
                                    }
                                }
                                stdout.flush()?;

                                session.line_buffer.clear();
                            }
                        }
                    } else {
                        writer.write_all(&[ch as u8])?;
                        writer.flush()?;
                    }
                }
            }
            Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {}
            Err(e) => return Err(e.into()),
        }

        match reader.read(&mut pty_buf) {
            Ok(0) => break,
            Ok(n) => {
                stdout.write_all(&pty_buf[..n])?;
                stdout.flush()?;
            }
            Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {}
            Err(e) => return Err(e.into()),
        }

        std::thread::sleep(std::time::Duration::from_micros(100));
    }

    Ok(())
}
