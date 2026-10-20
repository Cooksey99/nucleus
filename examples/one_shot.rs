//! Level 1: a single hosted completion.
//!
//! This mirrors the quick-start example in the README, written against the new
//! API. The library's internals are not implemented yet, so the example compiles
//! but panics when run.

use nucleus::Error;

// `nucleus::Error` is a rich enum (core-api.md §6), which trips
// `clippy::result_large_err` in any crate that returns it.
#[allow(clippy::result_large_err)]
#[tokio::main]
async fn main() -> Result<(), Error> {
    let summary = nucleus::complete("openai/gpt-5.1", "Summarize the design in one line.").await?;
    println!("{summary}");
    Ok(())
}
