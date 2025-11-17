use anyhow::Result;
use std::io::Write;
use crate::utils::{Theme, ModeTheme};

/// Represents the current interaction mode of the PTY.
#[derive(Debug)]
pub enum Mode {
    /// Standard terminal mode for executing shell commands.
    Terminal,
    /// AI interaction mode for conversing with the AI assistant.
    AI,
}

impl Mode {
    /// Returns the theme for this mode
    pub fn theme<'a>(&self, theme: &'a Theme) -> &'a ModeTheme {
        match self {
            Mode::Terminal => &theme.terminal_mode,
            Mode::AI => &theme.ai_mode,
        }
    }
}

/// Tracks the current state of input processing.
/// 
/// Manages the current mode, line buffer (current line being typed),
/// and input buffer (for AI mode character-by-character input).
pub struct InputState {
    mode: Mode,
    line_buffer: String,
    input_buffer: String,
}

impl InputState {
    /// Creates a new InputState starting in Terminal mode.
    pub fn new() -> Self {
        Self {
            mode: Mode::Terminal,
            line_buffer: String::new(),
            input_buffer: String::new(),
        }
    }

    /// Returns a reference to the current mode.
    pub fn mode(&self) -> &Mode {
        &self.mode
    }

    /// Toggles between Terminal and AI modes.
    /// 
    /// # Example
    /// ```
    /// let mut state = InputState::new(); // starts in Terminal
    /// state.toggle_mode(); // now AI
    /// state.toggle_mode(); // back to Terminal
    /// ```
    pub fn toggle_mode(&mut self) {
        self.mode = match self.mode {
            Mode::Terminal => Mode::AI,
            Mode::AI => Mode::Terminal,
        };
    }

    /// Clears both the line buffer and input buffer.
    /// 
    /// Called after processing a complete line of input.
    pub fn clear_buffers(&mut self) {
        self.line_buffer.clear();
        self.input_buffer.clear();
    }
}

/// Represents actions resulting from processing stdin input.
/// 
/// These actions are returned by `process_stdin` and handled by the IO loop.
pub enum InputAction {
    /// User requested to toggle between Terminal and AI modes (Ctrl-/).
    ToggleMode,
    /// A single character was typed and needs to be handled.
    CharacterInput(char),
    /// A complete line was entered (newline/return pressed).
    LineComplete {
        /// The parsed command text (e.g., "/ai query" in AI mode).
        cmd_text: String,
        /// The raw line including the newline character.
        raw_line: String,
    },
}

/// Handles all keyboard input processing and mode management.
/// 
/// The InputHandler processes raw stdin bytes, manages input state,
/// handles character-by-character input, and generates InputActions.
pub struct InputHandler {
    state: InputState,
    theme: Theme,
}

impl InputHandler {
    /// Creates a new InputHandler with default state and theme.
    pub fn new() -> Self {
        Self {
            state: InputState::new(),
            theme: Theme::default(),
        }
    }

    /// Creates a new InputHandler with a custom theme.
    pub fn with_theme(theme: Theme) -> Self {
        Self {
            state: InputState::new(),
            theme,
        }
    }

    /// Returns an immutable reference to the input state.
    pub fn state(&self) -> &InputState {
        &self.state
    }

    /// Returns a mutable reference to the input state.
    pub fn state_mut(&mut self) -> &mut InputState {
        &mut self.state
    }

    /// Returns the colored prompt string for the current mode.
    pub fn colored_prompt(&self) -> String {
        let mode_theme = self.state.mode.theme(&self.theme);
        mode_theme.render_prompt()
    }

    /// Processes raw stdin bytes and generates input actions.
    /// 
    /// Handles special key combinations (Ctrl-/), accumulates characters
    /// into the line buffer, and detects complete lines.
    /// 
    /// # Special Handling
    /// - Ctrl-/ (byte 31) triggers mode toggle
    /// - In AI mode: "/exit" or "/quit" triggers mode toggle
    /// - In AI mode: non-slash input is prefixed with "/ai "
    /// 
    /// # Returns
    /// A vector of InputActions to be processed by the IO loop.
    pub fn process_stdin(&mut self, buf: &[u8]) -> Result<Vec<InputAction>> {
        const CTRL_SLASH: u8 = 31;

        if buf.len() == 1 && buf[0] == CTRL_SLASH {
            return Ok(vec![InputAction::ToggleMode]);
        }

        let input = String::from_utf8_lossy(buf);
        let mut actions = Vec::new();

        for ch in input.chars() {
            if ch == '\n' || ch == '\r' {
                self.state.line_buffer.push(ch);
                let line = self.state.line_buffer.trim_end();
                let raw_line = self.state.line_buffer.clone();

                let cmd_text = match self.state.mode {
                    Mode::AI => {
                        if line == "/exit" || line == "/quit" {
                            actions.push(InputAction::ToggleMode);
                            self.state.clear_buffers();
                            continue;
                        } else if line.starts_with('/') {
                            line.to_string()
                        } else {
                            format!("/ai {}", line)
                        }
                    }
                    Mode::Terminal => line.to_string(),
                };

                actions.push(InputAction::LineComplete { cmd_text, raw_line });
                self.state.clear_buffers();
            } else {
                actions.push(InputAction::CharacterInput(ch));
                self.state.line_buffer.push(ch);
            }
        }

        Ok(actions)
    }

    /// Handles the display and buffering of a single character.
    /// 
    /// Behavior differs based on mode:
    /// - Terminal mode: forwards character directly to PTY writer
    /// - AI mode: echoes to stdout and handles backspace (\x7f, \x08)
    /// 
    /// # Arguments
    /// - `ch`: The character to process
    /// - `writer`: PTY writer for Terminal mode
    /// - `stdout`: Direct stdout for AI mode display
    pub fn handle_character(
        &mut self,
        ch: char,
        _writer: &mut dyn Write,
        stdout: &mut dyn Write,
    ) -> Result<()> {
        match self.state.mode {
            Mode::Terminal => {
                // PTY handles echo in raw mode, no need to write character here
            }
            Mode::AI => {
                if ch == '\x7f' || ch == '\x08' {
                    if !self.state.input_buffer.is_empty() {
                        self.state.input_buffer.pop();
                        self.state.line_buffer.pop();
                        stdout.write_all(b"\x08 \x08")?;
                        stdout.flush()?;
                    }
                } else {
                    self.state.input_buffer.push(ch);
                    stdout.write_all(&[ch as u8])?;
                    stdout.flush()?;
                }
            }
        }
        Ok(())
    }

    /// Updates the shell prompt to reflect the current mode.
    /// 
    /// Sends zsh commands to update PS1 with themed mode indicators.
    /// Also clears the screen for a clean visual transition.
    pub fn show_mode_indicator(&self, writer: &mut dyn Write) -> Result<()> {
        let mode_theme = self.state.mode.theme(&self.theme);
        let prompt_command = mode_theme.zsh_prompt_command();

        writer.write_all(prompt_command.as_bytes())?;
        writer.write_all(b"\n")?;
        writer.write_all(b"clear\n")?;
        writer.flush()?;
        Ok(())
    }
}
