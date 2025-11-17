use console::Style;

/// Zsh color names for prompt styling
#[derive(Debug, Clone, Copy)]
pub enum ZshColor {
    Green,
    Magenta,
    Cyan,
    Blue,
    Yellow,
    Red,
    White,
}

impl ZshColor {
    pub fn as_str(&self) -> &'static str {
        match self {
            ZshColor::Green => "green",
            ZshColor::Magenta => "magenta",
            ZshColor::Cyan => "cyan",
            ZshColor::Blue => "blue",
            ZshColor::Yellow => "yellow",
            ZshColor::Red => "red",
            ZshColor::White => "white",
        }
    }
}

/// All colors and styles for the terminal UI.
/// Change values here to customize the look and feel.
pub struct Theme {
    pub terminal_mode: ModeTheme,
    pub ai_mode: ModeTheme,
    pub messages: MessageTheme,
}

impl Theme {
    /// Default modern theme with clean colors
    pub fn default() -> Self {
        Self {
            terminal_mode: ModeTheme {
                symbol: "❯",
                label: None,
                zsh_color: ZshColor::Green,
                primary_color: Style::new().green().bold(),
                secondary_color: Style::new().dim(),
            },
            ai_mode: ModeTheme {
                symbol: "✨",
                label: Some("AI"),
                zsh_color: ZshColor::Magenta,
                primary_color: Style::new().magenta().bold(),
                secondary_color: Style::new().dim(),
            },
            messages: MessageTheme {
                success: Style::new().green(),
                error: Style::new().red().bold(),
                info: Style::new().cyan(),
                warning: Style::new().yellow(),
                dim: Style::new().dim(),
            },
        }
    }

    /// Alternative minimal theme
    pub fn minimal() -> Self {
        Self {
            terminal_mode: ModeTheme {
                symbol: ">",
                label: None,
                zsh_color: ZshColor::Cyan,
                primary_color: Style::new().cyan(),
                secondary_color: Style::new().dim(),
            },
            ai_mode: ModeTheme {
                symbol: "*",
                label: Some("AI"),
                zsh_color: ZshColor::Blue,
                primary_color: Style::new().blue(),
                secondary_color: Style::new().dim(),
            },
            messages: MessageTheme {
                success: Style::new().green(),
                error: Style::new().red(),
                info: Style::new().blue(),
                warning: Style::new().yellow(),
                dim: Style::new().dim(),
            },
        }
    }
}

/// Theme for a specific mode (Terminal or AI)
pub struct ModeTheme {
    pub symbol: &'static str,
    pub label: Option<&'static str>,
    pub zsh_color: ZshColor,
    pub primary_color: Style,
    pub secondary_color: Style,
}

impl ModeTheme {
    /// Renders the prompt with colors for display
    /// Example: "✨ AI │ " in magenta
    pub fn render_prompt(&self) -> String {
        let base = if let Some(label) = self.label {
            format!("{} {} │ ", self.symbol, label)
        } else {
            format!("{} ", self.symbol)
        };
        self.primary_color.apply_to(base).to_string()
    }

    /// Generates zsh command to prepend mode indicator to existing prompt
    pub fn zsh_prompt_command(&self) -> String {
        let color = self.zsh_color.as_str();
        let mode_indicator = if let Some(label) = self.label {
            format!("%B%F{{{color}}}{} {} %F{{240}}│%f%b ", self.symbol, label)
        } else {
            format!("%B%F{{{color}}}{}%f%b ", self.symbol)
        };
        
        format!("export PS1='{}${{OLD_PS1:-%# }}'", mode_indicator)
    }
}

/// Theme for different message types
pub struct MessageTheme {
    pub success: Style,
    pub error: Style,
    pub info: Style,
    pub warning: Style,
    pub dim: Style,
}
