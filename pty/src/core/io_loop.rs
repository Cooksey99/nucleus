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

    fn colored_prompt(&self) -> String {
        match self {
            Mode::Terminal => format!("\x1b[0;32m{}\x1b[0m ", self.prefix()),
            Mode::AI => format!("\x1b[0;35m{}\x1b[0m ", self.prefix()),
        }
    }
}

struct Session {
    mode: Mode,
    line_buffer: String,
    input_buffer: String,
}

impl Session {
    fn new() -> Self {
        Self {
            mode: Mode::Terminal,
            line_buffer: String::new(),
            input_buffer: String::new(),
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
            Mode::Terminal => format!("export PS1='%F{{green}}{}%f '", self.mode.prefix()),
            Mode::AI => format!("export PS1='%F{{magenta}}{}%f '", self.mode.prefix()),
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

const CTRL_SLASH: u8 = 31;  // Ctrl-/ to toggle modes

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
                if n == 1 && stdin_buf[0] == CTRL_SLASH {
                    session.toggle_mode();
                    session.show_mode_indicator(&mut writer)?;
                    continue;
                }

                let input = String::from_utf8_lossy(&stdin_buf[..n]);

                for ch in input.chars() {
                    if ch == '\n' || ch == '\r' {
                        session.line_buffer.push(ch);
                        let line = session.line_buffer.trim_end();

                        let cmd_text = match session.mode {
                            Mode::AI => {
                                if line == "/exit" || line == "/quit" {
                                    session.toggle_mode();
                                    session.show_mode_indicator(&mut writer)?;
                                    session.line_buffer.clear();
                                    session.input_buffer.clear();
                                    continue;
                                } else if line.starts_with('/') {
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
                            }
                            _ => {
                                stdout.write_all(b"\r\n")?;
                                stdout.flush()?;

                                match cmd.execute() {
                                    Ok(Some(response)) => {
                                        // Process response to ensure proper line breaks
                                        for line in response.lines() {
                                            stdout.write_all(line.as_bytes())?;
                                            stdout.write_all(b"\r\n")?;
                                        }
                                    }
                                    Ok(None) => {}
                                    Err(e) => {
                                        let error_msg = format!("Error: {}\r\n", e);
                                        stdout.write_all(error_msg.as_bytes())?;
                                    }
                                }
                                stdout.flush()?;

                                // Show a new prompt after command execution
                                if matches!(session.mode, Mode::AI) {
                                    let prompt = session.mode.colored_prompt();
                                    stdout.write_all(prompt.as_bytes())?;
                                    stdout.flush()?;
                                }
                            }
                        }
                        session.line_buffer.clear();
                        session.input_buffer.clear();
                    } else {
                        match session.mode {
                            Mode::Terminal => {
                                writer.write_all(&[ch as u8])?;
                                writer.flush()?;
                            }
                            Mode::AI => {
                                // Handle backspace/delete in AI mode
                                if ch == '\x7f' || ch == '\x08' {
                                    if !session.input_buffer.is_empty() {
                                        session.input_buffer.pop();
                                        session.line_buffer.pop();
                                        // Move back, overwrite with space, move back again
                                        stdout.write_all(b"\x08 \x08")?;
                                        stdout.flush()?;
                                    }
                                } else {
                                    // In AI mode, echo the character to stdout for visual feedback
                                    session.input_buffer.push(ch);
                                    session.line_buffer.push(ch);
                                    stdout.write_all(&[ch as u8])?;
                                    stdout.flush()?;
                                }
                            }
                        }
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
