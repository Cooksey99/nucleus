use anyhow::Result;
use std::io::Write;

/// Handles rendering output to the terminal with various formatting capabilities.
/// 
/// The OutputHandler manages all output display, providing methods for rendering
/// different types of content like PTY output, command responses, errors, and prompts.
pub struct OutputHandler {
    stdout: Box<dyn Write + Send>,
}

impl OutputHandler {
    /// Creates a new OutputHandler with the given writer.
    /// 
    /// # Example
    /// ```
    /// let handler = OutputHandler::new(Box::new(std::io::stdout()));
    /// ```
    pub fn new(stdout: Box<dyn Write + Send>) -> Self {
        Self { stdout }
    }

    /// Renders raw PTY output directly to the terminal.
    /// 
    /// This passes through bytes from the PTY without modification,
    /// preserving all control sequences and formatting.
    pub fn render_pty_output(&mut self, buf: &[u8]) -> Result<()> {
        self.stdout.write_all(buf)?;
        self.stdout.flush()?;
        Ok(())
    }

    /// Renders a command response with proper line breaks.
    /// 
    /// Adds a leading newline and ensures each line ends with a carriage return
    /// and newline (\r\n) for proper terminal display.
    /// 
    /// # Example
    /// ```
    /// handler.render_command_response("Hello\nWorld")?;
    /// // Output:
    /// //
    /// // Hello
    /// // World
    /// ```
    pub fn render_command_response(&mut self, response: &str) -> Result<()> {
        self.stdout.write_all(b"\r\n")?;
        
        for line in response.lines() {
            self.stdout.write_all(line.as_bytes())?;
            self.stdout.write_all(b"\r\n")?;
        }
        
        self.stdout.flush()?;
        Ok(())
    }

    /// Renders an error message with "Error: " prefix.
    /// 
    /// # Example
    /// ```
    /// handler.render_error("Connection failed")?;
    /// // Output: Error: Connection failed
    /// ```
    pub fn render_error(&mut self, error: &str) -> Result<()> {
        let error_msg = format!("Error: {}\r\n", error);
        self.stdout.write_all(error_msg.as_bytes())?;
        self.stdout.flush()?;
        Ok(())
    }

    /// Renders a prompt string directly to the terminal.
    /// 
    /// Useful for displaying mode indicators or input prompts.
    /// 
    /// # Example
    /// ```
    /// handler.render_prompt("[AI] ")?;
    /// ```
    pub fn render_prompt(&mut self, prompt: &str) -> Result<()> {
        self.stdout.write_all(prompt.as_bytes())?;
        self.stdout.flush()?;
        Ok(())
    }

    /// Renders a newline (\r\n) to the terminal.
    /// 
    /// Uses carriage return + newline for proper terminal compatibility.
    pub fn render_newline(&mut self) -> Result<()> {
        self.stdout.write_all(b"\r\n")?;
        self.stdout.flush()?;
        Ok(())
    }

    /// Returns a mutable reference to the underlying writer.
    /// 
    /// Allows direct access to the writer for low-level operations
    /// when the provided rendering methods are insufficient.
    pub fn get_writer(&mut self) -> &mut dyn Write {
        &mut *self.stdout
    }
}
