//! Slash commands for interactive chats.
//!
//! A line that starts with `/` is a command. Anything else, including the word
//! `reset`, is a normal message and should be sent to the model.

/// What the caller should do after inspecting a line.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CommandEffect {
    /// Not a slash command. Send the line to the model.
    NotACommand,
    /// Command ran. Do not send the line to the model.
    Handled { message: String },
    /// Command asked the session to stop.
    Exit { message: String },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Builtin {
    Help,
    Reset,
    Exit,
}

struct Command {
    name: &'static str,
    summary: &'static str,
    builtin: Builtin,
}

const COMMANDS: &[Command] = &[
    Command {
        name: "help",
        summary: "List commands",
        builtin: Builtin::Help,
    },
    Command {
        name: "reset",
        summary: "Clear conversation history",
        builtin: Builtin::Reset,
    },
    Command {
        name: "exit",
        summary: "Leave the chat",
        builtin: Builtin::Exit,
    },
    Command {
        name: "quit",
        summary: "Leave the chat",
        builtin: Builtin::Exit,
    },
];

#[derive(Debug)]
pub(crate) enum Parsed {
    Builtin(Builtin),
    Unknown { name: String },
    Usage { usage: String },
}

pub(crate) fn parse(input: &str) -> Option<Parsed> {
    let rest = input.trim().strip_prefix('/')?;
    if rest.is_empty() {
        return Some(Parsed::Unknown {
            name: String::new(),
        });
    }

    let mut parts = rest.splitn(2, char::is_whitespace);
    let name = parts.next().unwrap_or("");
    let args = parts.next().unwrap_or("").trim();

    let Some(command) = COMMANDS
        .iter()
        .find(|command| command.name.eq_ignore_ascii_case(name))
    else {
        return Some(Parsed::Unknown {
            name: name.to_string(),
        });
    };

    if !args.is_empty() {
        return Some(Parsed::Usage {
            usage: format!("/{}", command.name),
        });
    }

    Some(Parsed::Builtin(command.builtin))
}

pub(crate) fn help_text() -> String {
    let mut text = String::from("Commands:");
    for command in COMMANDS {
        text.push_str(&format!("\n  /{:<6} {}", command.name, command.summary));
    }
    text
}

/// Effect of a line, before any manager state is changed.
pub(crate) enum Action {
    NotACommand,
    ClearHistory,
    Reply(String),
    Exit,
}

pub(crate) fn action(input: &str) -> Action {
    match parse(input) {
        None => Action::NotACommand,
        Some(Parsed::Builtin(Builtin::Help)) => Action::Reply(help_text()),
        Some(Parsed::Builtin(Builtin::Reset)) => Action::ClearHistory,
        Some(Parsed::Builtin(Builtin::Exit)) => Action::Exit,
        Some(Parsed::Unknown { name }) => {
            let shown = if name.is_empty() {
                "/".to_string()
            } else {
                format!("/{name}")
            };
            Action::Reply(format!("Unknown command '{shown}'. Try /help."))
        }
        Some(Parsed::Usage { usage }) => Action::Reply(format!("Usage: {usage}")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bare_words_are_messages() {
        assert!(parse("reset").is_none());
        assert!(parse("exit").is_none());
        assert!(parse("please /reset later").is_none());
    }

    #[test]
    fn known_commands_parse() {
        assert!(matches!(parse("/reset"), Some(Parsed::Builtin(Builtin::Reset))));
        assert!(matches!(parse("/RESET"), Some(Parsed::Builtin(Builtin::Reset))));
        assert!(matches!(parse("/quit"), Some(Parsed::Builtin(Builtin::Exit))));
        assert!(matches!(parse("/help"), Some(Parsed::Builtin(Builtin::Help))));
    }

    #[test]
    fn extra_args_are_usage_errors() {
        assert!(matches!(parse("/reset now"), Some(Parsed::Usage { .. })));
    }

    #[test]
    fn unknown_command_keeps_the_name() {
        match parse("/nope") {
            Some(Parsed::Unknown { name }) => assert_eq!(name, "nope"),
            other => panic!("unexpected {other:?}"),
        }
    }
}
