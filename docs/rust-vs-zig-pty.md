# Rust vs Zig for PTY Wrapper (with Go AI Backend)

## Your Architecture

```
┌─────────────────────────────────┐
│   PTY Wrapper (Rust or Zig)     │
│   - Terminal I/O handling       │
│   - Trigger detection           │
│   - Context capture             │
└──────────────┬──────────────────┘
               │ IPC (stdio/socket)
┌──────────────▼──────────────────┐
│   AI Backend (Go + Ollama)      │
│   - LLM calls to Ollama         │
│   - Tool execution              │
│   - Response formatting         │
└──────────────────────────────────┘
```

## Rust vs Zig for PTY

### Rust: Better Choice for This

**Verdict**: **Rust is the better fit** for PTY wrapper with Go backend.

### Why Rust Wins

#### 1. **Mature PTY Libraries**

**Rust**:
```rust
// portable-pty - cross-platform, well-maintained
use portable_pty::{CommandBuilder, PtySize, native_pty_system};

let pty_system = native_pty_system();
let pair = pty_system.openpty(PtySize {
    rows: 24,
    cols: 80,
    ..Default::default()
})?;

let mut cmd = CommandBuilder::new("zsh");
let mut child = pair.slave.spawn_command(cmd)?;
```

**Libraries**:
- `portable-pty` (1.5k+ stars, actively maintained)
- `nix` (low-level Unix APIs, 2.7k+ stars)
- Used by: Alacritty, Warp, VS Code terminal

**Zig**:
```zig
// No mature PTY library
// You'd write raw ioctl calls yourself
const posix = @import("std").posix;
// ... manual PTY setup with posix_openpt, grantpt, unlockpt
```

**Libraries**: None mature. You're on your own.

#### 2. **Async I/O Story**

**Rust** (Tokio):
```rust
use tokio::io::{AsyncReadExt, AsyncWriteExt};

// Handle PTY I/O concurrently
tokio::spawn(async move {
    let mut buf = [0u8; 4096];
    loop {
        let n = stdin.read(&mut buf).await?;
        pty_master.write_all(&buf[..n]).await?;
    }
});
```

Clean, mature async with `tokio`. Perfect for concurrent I/O.

**Zig**:
```zig
// Async is newer, less documented
const std = @import("std");
// Would use std.event or manual epoll/kqueue
```

Zig's async is still evolving. Less ecosystem support.

#### 3. **IPC with Go Backend**

**Option 1: Stdio Communication**

Both Rust and Zig can do this easily:

```rust
// Rust: spawn Go process, communicate via stdin/stdout
use std::process::{Command, Stdio};

let mut go_backend = Command::new("./ai-backend")
    .stdin(Stdio::piped())
    .stdout(Stdio::piped())
    .spawn()?;

// Send query
writeln!(go_backend.stdin.as_mut().unwrap(), 
    "{}", serde_json::to_string(&query)?)?;

// Read response
let response: Response = serde_json::from_reader(
    go_backend.stdout.as_mut().unwrap())?;
```

```zig
// Zig: similar, but more manual JSON handling
const std = @import("std");
const child = try std.ChildProcess.init(&.{"./ai-backend"}, allocator);
// ... write/read JSON
```

**Option 2: Unix Socket Communication**

```rust
// Rust: tokio provides excellent Unix socket support
use tokio::net::UnixStream;

let stream = UnixStream::connect("/tmp/ai-backend.sock").await?;
stream.write_all(query.as_bytes()).await?;
```

```zig
// Zig: manual socket handling
const socket = try std.net.connectUnixSocket("/tmp/ai-backend.sock");
```

**Rust advantage**: Better libraries for JSON, sockets, error handling.

#### 4. **Error Handling**

**Rust**:
```rust
use anyhow::{Result, Context};

fn setup_pty() -> Result<Master> {
    let pty = pty_system.openpty(size)
        .context("Failed to open PTY")?;
    
    let child = pty.slave.spawn_command(cmd)
        .context("Failed to spawn shell")?;
    
    Ok(pty.master)
}
```

Excellent error handling with `Result`, `?` operator, `anyhow`/`thiserror`.

**Zig**:
```zig
fn setupPty() !Master {
    const pty = try ptySystem.openpty(size);
    const child = try pty.slave.spawnCommand(cmd);
    return pty.master;
}
```

Also has try/catch, but less ecosystem tooling.

#### 5. **Development Velocity**

**Rust**:
- Extensive documentation
- Large community (r/rust, Discord)
- Many examples of terminal/PTY projects
- cargo for dependencies
- rustfmt, clippy for tooling

**Zig**:
- Smaller community
- Less documentation
- Fewer PTY examples
- Newer package manager
- Still pre-1.0

For a project you want to iterate on, Rust gives you more resources.

#### 6. **Cross-Platform**

**Rust**: `portable-pty` works on macOS, Linux, Windows (with ConPTY)

**Zig**: You'd write platform-specific code yourself

Since you're on macOS, both work, but Rust gives you portability for free.

## Concrete Rust PTY Implementation

### Project Structure

```
pty-wrapper/
├── Cargo.toml
├── src/
│   ├── main.rs          # Entry point, PTY setup
│   ├── pty_handler.rs   # PTY I/O management
│   ├── session.rs       # Context tracking
│   ├── ai_client.rs     # IPC with Go backend
│   └── input.rs         # Trigger detection
└── ai-backend/          # Go project
    ├── go.mod
    ├── main.go          # Ollama integration
    └── tools.go         # Tool implementations
```

### Cargo.toml

```toml
[package]
name = "pty-wrapper"
version = "0.1.0"
edition = "2021"

[dependencies]
portable-pty = "0.8"
tokio = { version = "1", features = ["full"] }
serde = { version = "1", features = ["derive"] }
serde_json = "1"
anyhow = "1"
nix = { version = "0.27", features = ["term", "signal"] }
```

### src/main.rs (Rust PTY)

```rust
use anyhow::Result;
use portable_pty::{CommandBuilder, PtySize, native_pty_system};
use tokio::io::{AsyncReadExt, AsyncWriteExt};

mod ai_client;
mod pty_handler;
mod session;

#[tokio::main]
async fn main() -> Result<()> {
    // Get user's shell
    let shell = std::env::var("SHELL")
        .unwrap_or_else(|_| "/bin/zsh".to_string());

    // Create PTY
    let pty_system = native_pty_system();
    let pair = pty_system.openpty(PtySize {
        rows: 24,
        cols: 80,
        pixel_width: 0,
        pixel_height: 0,
    })?;

    // Spawn shell
    let mut cmd = CommandBuilder::new(shell);
    let _child = pair.slave.spawn_command(cmd)?;
    
    // Drop slave so only master remains
    drop(pair.slave);

    // Get master PTY
    let mut master = pair.master;

    // Spawn Go AI backend
    let ai_client = ai_client::start_backend().await?;

    // Handle I/O
    pty_handler::handle_io(master, ai_client).await?;

    Ok(())
}
```

### src/pty_handler.rs

```rust
use anyhow::Result;
use portable_pty::MasterPty;
use tokio::io::{AsyncReadExt, AsyncWriteExt, stdin, stdout};

pub async fn handle_io(
    mut master: Box<dyn MasterPty>,
    ai_client: AiClient,
) -> Result<()> {
    let mut stdin = stdin();
    let mut stdout = stdout();
    
    let (mut reader, mut writer) = master.split();

    // User input → PTY
    tokio::spawn(async move {
        let mut buf = [0u8; 1024];
        loop {
            let n = stdin.read(&mut buf).await?;
            
            // Check for trigger (Ctrl-G = 0x07)
            if n == 1 && buf[0] == 0x07 {
                handle_ai_mode(&mut stdin, &mut stdout, &ai_client).await?;
                continue;
            }
            
            writer.write_all(&buf[..n]).await?;
        }
        Ok::<_, anyhow::Error>(())
    });

    // PTY → User output
    let mut buf = [0u8; 4096];
    loop {
        let n = reader.read(&mut buf).await?;
        if n == 0 { break; }
        stdout.write_all(&buf[..n]).await?;
        stdout.flush().await?;
    }

    Ok(())
}

async fn handle_ai_mode(
    stdin: &mut Stdin,
    stdout: &mut Stdout,
    ai_client: &AiClient,
) -> Result<()> {
    stdout.write_all(b"\r\n🤖 AI> ").await?;
    stdout.flush().await?;
    
    let query = read_line(stdin, stdout).await?;
    
    if query.is_empty() {
        return Ok(());
    }
    
    // Send to Go backend
    let response = ai_client.query(&query).await?;
    
    stdout.write_all(format!("\r\n{}\r\n\r\n", response).as_bytes()).await?;
    stdout.flush().await?;
    
    Ok(())
}
```

### src/ai_client.rs (IPC with Go)

```rust
use anyhow::Result;
use serde::{Deserialize, Serialize};
use tokio::process::{Command, Child, ChildStdin, ChildStdout};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};

#[derive(Serialize)]
struct Query {
    prompt: String,
    context: Context,
}

#[derive(Deserialize)]
struct Response {
    text: String,
    suggested_command: Option<String>,
}

pub struct AiClient {
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
}

impl AiClient {
    pub async fn start_backend() -> Result<Self> {
        let mut child = Command::new("./ai-backend")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .spawn()?;
        
        let stdin = child.stdin.take().unwrap();
        let stdout = BufReader::new(child.stdout.take().unwrap());
        
        Ok(Self { stdin, stdout })
    }
    
    pub async fn query(&mut self, prompt: &str) -> Result<String> {
        // Send query as JSON
        let query = Query {
            prompt: prompt.to_string(),
            context: get_context(),
        };
        
        let json = serde_json::to_string(&query)?;
        self.stdin.write_all(json.as_bytes()).await?;
        self.stdin.write_all(b"\n").await?;
        self.stdin.flush().await?;
        
        // Read response
        let mut line = String::new();
        self.stdout.read_line(&mut line).await?;
        
        let response: Response = serde_json::from_str(&line)?;
        Ok(response.text)
    }
}
```

### ai-backend/main.go (Go + Ollama)

```go
package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"
)

type Query struct {
	Prompt  string            `json:"prompt"`
	Context map[string]string `json:"context"`
}

type Response struct {
	Text             string  `json:"text"`
	SuggestedCommand *string `json:"suggested_command,omitempty"`
}

func main() {
	scanner := bufio.NewScanner(os.Stdin)
	
	for scanner.Scan() {
		line := scanner.Text()
		
		var query Query
		if err := json.Unmarshal([]byte(line), &query); err != nil {
			continue
		}
		
		// Call Ollama
		response := callOllama(query.Prompt, query.Context)
		
		// Send response
		resp := Response{Text: response}
		json.NewEncoder(os.Stdout).Encode(resp)
	}
}

func callOllama(prompt string, context map[string]string) string {
	// TODO: Actual Ollama API call
	// For now, mock response
	return fmt.Sprintf("Ollama response to: %s", prompt)
}
```

## Communication Protocol

### Option 1: Line-Delimited JSON (Simpler)

**Rust → Go**:
```json
{"prompt": "list files", "context": {"pwd": "/home/user"}}\n
```

**Go → Rust**:
```json
{"text": "ls -la", "suggested_command": "ls -la"}\n
```

Easy to implement, works well for request/response.

### Option 2: Unix Socket (More Robust)

Start Go backend as daemon listening on socket:

```go
// Go: Listen on Unix socket
listener, _ := net.Listen("unix", "/tmp/ai-backend.sock")
for {
    conn, _ := listener.Accept()
    go handleConnection(conn)
}
```

```rust
// Rust: Connect to socket
use tokio::net::UnixStream;
let stream = UnixStream::connect("/tmp/ai-backend.sock").await?;
```

Better for long-running backend, multiple clients.

## Why Not Zig?

**If you were doing pure Zig, it'd be fine.** But for Rust PTY + Go AI:

### Zig Disadvantages:

1. **No PTY library** - you'd write 200+ lines of raw ioctl calls
2. **Less JSON tooling** - more manual serialization
3. **Smaller community** - harder to find examples
4. **Pre-1.0** - APIs still changing
5. **Async story** - less mature than Tokio

### Zig Advantages:

1. **Simpler language** - easier to learn than Rust
2. **Faster compilation** - seconds vs minutes
3. **Smaller binary** - ~500KB vs ~2MB
4. **C interop** - trivial to call C libraries

**For this project**: The PTY library and async I/O make Rust the pragmatic choice.

## If You Really Want Zig

You'd need to implement PTY yourself:

```zig
const std = @import("std");
const posix = std.posix;

pub fn openPty() !Pty {
    // Open master PTY
    const master_fd = try posix.posix_openpt(
        posix.O.RDWR | posix.O.NOCTTY
    );
    
    // Grant access to slave
    try posix.grantpt(master_fd);
    try posix.unlockpt(master_fd);
    
    // Get slave path
    const slave_path = try posix.ptsname(master_fd);
    
    // Open slave
    const slave_fd = try posix.open(
        slave_path,
        posix.O.RDWR | posix.O.NOCTTY,
        0
    );
    
    return Pty{ .master = master_fd, .slave = slave_fd };
}
```

Doable, but you're reimplementing `portable-pty`.

## Recommendation

**Use Rust for PTY wrapper**, Go for Ollama backend.

**Why**:
1. `portable-pty` is battle-tested (used by Warp, Alacritty)
2. Tokio async is perfect for concurrent I/O
3. Better JSON/error handling for IPC
4. Larger community for help
5. You said you prefer Rust anyway!

**Architecture**:
```bash
# Build both
cd pty-wrapper && cargo build --release
cd ai-backend && go build -o ai-backend

# Run Rust wrapper (spawns Go backend)
./pty-wrapper/target/release/pty-wrapper
```

## Getting Started with Rust

```bash
cd ~/llm-workspace
cargo new pty-wrapper
cd pty-wrapper

# Add dependencies
cargo add portable-pty tokio serde serde_json anyhow nix

# Start with minimal example
cargo run
```

Then build incrementally like the Go walkthrough, but in Rust.

Zig is cool, but for this specific project (PTY + IPC), Rust is the pragmatic choice.
