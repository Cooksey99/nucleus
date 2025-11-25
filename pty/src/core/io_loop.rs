use super::ai_client::ConversationHistory;
use super::command::Command;
use super::input_handler::{InputAction, InputHandler, Mode};
use super::output_handler::OutputHandler;
use anyhow::Result;
use std::io::Read;

/// Runs event loop to track various user inputs.
pub fn run_io_loop(master: &mut Box<dyn portable_pty::MasterPty + Send>) -> Result<()> {
    use std::io::stdin;

    let mut stdin = stdin();

    let pwd = std::env::current_dir()
        .ok()
        .and_then(|p| p.to_str().map(String::from));

    let mut reader = master.try_clone_reader()?;
    let mut writer = master.take_writer()?;

    let mut input_handler = InputHandler::new();
    let mut output_handler = OutputHandler::new(Box::new(std::io::stdout()));
    let mut conversation_history = ConversationHistory::new();

    let mut stdin_buf = vec![0u8; 8192];
    let mut pty_buf = vec![0u8; 262144];

    input_handler.show_mode_indicator(output_handler.get_writer())?;

    // Event loop to handle user inputs
    loop {
        match stdin.read(&mut stdin_buf) {
            Ok(0) => break,
            Ok(n) => {
                let actions = input_handler.process_stdin(&stdin_buf[..n])?;

                for action in actions {
                    match action {
                        InputAction::ToggleMode => {
                            input_handler.state_mut().toggle_mode();
                            input_handler.show_mode_indicator(output_handler.get_writer())?;
                        }
                        InputAction::CharacterInput(ch) => {
                            input_handler.handle_character(
                                ch,
                                &mut writer,
                                output_handler.get_writer(),
                            )?;
                        }
                        InputAction::LineComplete { cmd_text, raw_line } => {
                            let cmd = Command::parse(&cmd_text);
                            let line_bytes = raw_line.as_bytes();

                            match cmd {
                                Command::PassThrough => {
                                    writer.write_all(line_bytes)?;
                                    writer.flush()?;
                                }
                                _ => {
                                    output_handler.render_newline()?;

                                    match cmd
                                        .execute(pwd.as_deref(), Some(&mut conversation_history))
                                    {
                                        Ok(Some(response)) => {
                                            output_handler.render_command_response(&response)?;
                                        }
                                        Ok(None) => {}
                                        Err(e) => {
                                            output_handler.render_error(&e.to_string())?;
                                        }
                                    }

                                    if matches!(input_handler.state().mode(), Mode::AI) {
                                        let prompt = input_handler.colored_prompt();
                                        output_handler.render_prompt(&prompt)?;
                                    }
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
                output_handler.render_pty_output(&pty_buf[..n])?;
            }
            Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {}
            Err(e) => return Err(e.into()),
        }

        std::thread::sleep(std::time::Duration::from_micros(100));
    }

    Ok(())
}
