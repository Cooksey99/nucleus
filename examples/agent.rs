//! Run the agent tool loop against a scripted provider.
//!
//! Hosted providers do not implement the new [`Provider`] trait yet, so this
//! example stands one in: the first request returns a tool call, the tool runs,
//! and the second request returns the answer. No API key and no network.
//!
//! ```sh
//! cargo run --example agent
//! ```

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use nucleus::{
    Agent, Capabilities, ChatRequest, Conversation, Error, Event, EventStream, Message, Part,
    Provider, ProviderKind, Result, Role, StopReason, ThinkingSupport, Tool, ToolCallId,
    ToolContext, ToolOutput,
};
use serde_json::value::RawValue;

// `nucleus::Error` is a rich enum, which trips `clippy::result_large_err`.
#[allow(clippy::result_large_err)]
#[tokio::main]
async fn main() -> Result<()> {
    let weather = Arc::new(Weather);
    let script = Script::new(vec![
        vec![
            Event::ToolCall {
                id: ToolCallId::new("call_1"),
                name: "get_weather".to_string(),
                input: raw(r#"{"city":"Paris"}"#),
            },
            Event::Stop(StopReason::ToolUse),
        ],
        vec![
            Event::TextDelta {
                text: "It is sunny in Paris.".to_string(),
            },
            Event::Stop(StopReason::EndTurn),
        ],
    ]);

    let agent = Agent::builder()
        .model("openai/gpt-5.1")
        .provider(script.clone())
        .system("Be brief.")
        .tool(weather)
        .build()
        .await?;

    let mut conversation = Conversation::new();
    let turn = agent
        .run(
            &mut conversation,
            Message::user("What's the weather in Paris?"),
        )
        .await?;

    let requests = script.requests();
    println!(
        "system prompt sent: {}",
        requests
            .first()
            .and_then(|request| request.system.as_deref())
            .unwrap_or("(none)")
    );
    println!("provider calls: {}", requests.len());
    println!("---");
    for message in conversation.messages() {
        println!("{}: {}", role_name(message.role), render(message));
    }
    println!("---");
    println!("turn: {}", turn.message.text());

    assert_eq!(turn.message.text(), "It is sunny in Paris.");
    assert_eq!(requests.len(), 2);
    assert!(conversation
        .messages()
        .iter()
        .all(|message| message.text() != "Be brief."));
    Ok(())
}

/// A tool the scripted model can call.
struct Weather;

#[async_trait::async_trait]
impl Tool for Weather {
    fn name(&self) -> &str {
        "get_weather"
    }

    fn description(&self) -> &str {
        "Look up the weather in a city."
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": { "city": { "type": "string" } },
            "required": ["city"]
        })
    }

    async fn call(&self, arguments: Box<RawValue>, _context: ToolContext) -> Result<ToolOutput> {
        let value: serde_json::Value = serde_json::from_str(arguments.get())
            .map_err(|err| Error::Serialization(err.to_string()))?;
        let city = value
            .get("city")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("unknown");
        Ok(ToolOutput::text(format!("Sunny in {city}")))
    }
}

/// A provider that returns a fixed queue of event lists, one per request.
struct Script {
    replies: Mutex<VecDeque<Vec<Event>>>,
    seen: Mutex<Vec<ChatRequest>>,
}

impl Script {
    fn new(replies: Vec<Vec<Event>>) -> Arc<Self> {
        Arc::new(Self {
            replies: Mutex::new(VecDeque::from(replies)),
            seen: Mutex::new(Vec::new()),
        })
    }

    fn requests(&self) -> Vec<ChatRequest> {
        self.seen.lock().expect("script lock").clone()
    }
}

#[async_trait::async_trait]
impl Provider for Script {
    fn kind(&self) -> ProviderKind {
        ProviderKind::Custom
    }

    async fn chat_stream(&self, request: ChatRequest) -> Result<EventStream> {
        self.seen.lock().expect("script lock").push(request);
        let events = self
            .replies
            .lock()
            .expect("script lock")
            .pop_front()
            .unwrap_or_default();
        Ok(Box::pin(futures::stream::iter(events)))
    }

    fn capabilities(&self, _model: &str) -> Capabilities {
        Capabilities::streaming(ThinkingSupport::NotApplicable, true)
    }
}

fn raw(json: &str) -> Box<RawValue> {
    RawValue::from_string(json.to_string()).expect("tool input is json")
}

fn role_name(role: Role) -> &'static str {
    match role {
        Role::User => "user",
        Role::Assistant => "assistant",
        _ => "other",
    }
}

fn render(message: &Message) -> String {
    let mut bits = Vec::new();
    let text = message.text();
    if !text.is_empty() {
        bits.push(text);
    }
    for (_id, name, input) in message.tool_calls() {
        bits.push(format!("tool call {name} {input}"));
    }
    for part in &message.parts {
        if let Part::ToolResult {
            output, is_error, ..
        } = part
        {
            let flag = if *is_error { "error" } else { "ok" };
            bits.push(format!("tool result ({flag}): {}", output.content));
        }
    }
    if bits.is_empty() {
        "(empty)".to_string()
    } else {
        bits.join(" | ")
    }
}
