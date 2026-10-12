# Phase 0.5 spikes

Short experiments that settle unknowns before the 0.2 design is written (see [ROADMAP.md](../../../ROADMAP.md), Phase 0.5).
Spike code lives in a throwaway `spikes/` directory and is never merged. Only these notes are.

Spikes make no paid API calls and use no API keys. Hosted-provider spikes work from API references, OpenAPI specs, official SDK source and SDK test fixtures. A question that only a live call could answer is recorded as open.

| Spike | Note |
|---|---|
| mistral.rs (latest crates.io release) | [mistralrs.md](mistralrs.md) |
| Gloo AI | [gloo.md](gloo.md) |
| OpenAI and xAI Responses API | [responses-api.md](responses-api.md) |
| Anthropic Messages API, and the HTTP mock tool | [anthropic-messages.md](anthropic-messages.md) |
| `rmcp` | [rmcp.md](rmcp.md) |
| Batch APIs | [batch.md](batch.md) |

## Note template

```markdown
# Spike: <topic>

Date: YYYY-MM-DD. Versions and doc pages checked: …

## Questions
The questions from the roadmap, plus any found along the way.

## Findings
One subsection per question. Each claim cites its source: a doc URL, SDK file and commit, or spike code output.
Say how each one was established: verified by running code, read in docs, or inferred.

## Impact on the design
Concrete changes to ROADMAP.md or the planned API, or "none".

## Open questions
Anything still unsettled, including what only a live call could answer.
```
