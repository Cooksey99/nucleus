use nucleus::{Agent, Conversation, Message, Result};

#[tokio::main]
async fn main() -> Result<()> {
    let agent = Agent::builder()
        .model("local//Users/cooksey/development/nucleus/.models/Llama-3.2-1B-Instruct-Q6_K.gguf")
        .system("You are Jarvis, a personal assistant.")
        .build()
        .await?;

    let mut conv = Conversation::new();
    let res = agent
        .run(&mut conv, Message::user("Hi. What's your name?"))
        .await?;

    println!("{}", res.message.text());
    Ok(())
}
