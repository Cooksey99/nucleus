# Project Structure for Rust PTY + Go AI Backend

## Option 1: Monorepo (Recommended)

**Best for**: Single project where both components are tightly coupled.

```
ai-terminal/
├── Cargo.toml              # Rust workspace root
├── Cargo.lock
├── README.md
├── .gitignore
├── Makefile                # Build commands for both
│
├── pty-wrapper/            # Rust PTY binary
│   ├── Cargo.toml
│   └── src/
│       ├── main.rs
│       ├── pty_handler.rs
│       ├── ai_client.rs
│       └── session.rs
│
└── ai-backend/             # Go AI backend
    ├── go.mod
    ├── go.sum
    ├── main.go
    ├── ollama.go
    └── tools.go
```

### Setup Commands

```bash
# Create root directory
mkdir ai-terminal
cd ai-terminal

# Initialize Rust workspace
cat > Cargo.toml << 'EOF'
[workspace]
members = ["pty-wrapper"]
resolver = "2"
EOF

# Create Rust project
cargo new pty-wrapper
cd pty-wrapper
cargo add portable-pty tokio serde serde_json anyhow nix

# Create Go project
cd ..
mkdir ai-backend
cd ai-backend
go mod init github.com/cooksey/ai-terminal/ai-backend
```

### Makefile for Easy Building

```makefile
.PHONY: all build-rust build-go clean run install

all: build-rust build-go

build-rust:
	cargo build --release

build-go:
	cd ai-backend && go build -o ../target/release/ai-backend

clean:
	cargo clean
	cd ai-backend && go clean

run: build-rust build-go
	./target/release/pty-wrapper

install: build-rust build-go
	cp target/release/pty-wrapper /usr/local/bin/ai-terminal
	cp target/release/ai-backend /usr/local/bin/ai-terminal-backend

dev:
	cargo build
	cd ai-backend && go build -o ../target/debug/ai-backend
	./target/debug/pty-wrapper
```

### Usage

```bash
# Build everything
make

# Run
make run

# Development mode (faster compilation)
make dev

# Install system-wide
sudo make install
```

### .gitignore

```gitignore
# Rust
target/
Cargo.lock

# Go
ai-backend/ai-backend
*.exe
*.test

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
```

### How Rust Finds Go Binary

**Option A: Look in same directory as Rust binary**

```rust
// src/ai_client.rs
use std::env;
use std::path::PathBuf;

fn get_backend_path() -> PathBuf {
    // Get the directory of the current executable
    let exe_path = env::current_exe().unwrap();
    let exe_dir = exe_path.parent().unwrap();
    
    // Look for ai-backend in the same directory
    exe_dir.join("ai-backend")
}

pub async fn start_backend() -> Result<AiClient> {
    let backend_path = get_backend_path();
    
    let mut child = Command::new(&backend_path)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()?;
    
    // ...
}
```

**Option B: Embed path at compile time**

```rust
// src/ai_client.rs
const BACKEND_PATH: &str = env!("AI_BACKEND_PATH");

pub async fn start_backend() -> Result<AiClient> {
    let mut child = Command::new(BACKEND_PATH)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()?;
    // ...
}
```

```bash
# Build with custom path
AI_BACKEND_PATH=./target/release/ai-backend cargo build --release
```

**Option C: Search PATH**

```rust
// Just use the binary name - let PATH resolution handle it
pub async fn start_backend() -> Result<AiClient> {
    let mut child = Command::new("ai-backend")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()?;
    // ...
}
```

This works if you `make install` both binaries to `/usr/local/bin`.

## Option 2: Separate Repos with Submodule

**Best for**: When you want to version/release them independently.

```
# Separate repos
ai-terminal-pty/        # github.com/cooksey/ai-terminal-pty
└── (Rust code)

ai-terminal-backend/    # github.com/cooksey/ai-terminal-backend
└── (Go code)

# Main project that combines them
ai-terminal/
├── .gitmodules
├── pty-wrapper/        # Git submodule → ai-terminal-pty
└── ai-backend/         # Git submodule → ai-terminal-backend
```

### Setup

```bash
# Create main repo
mkdir ai-terminal
cd ai-terminal
git init

# Add submodules
git submodule add https://github.com/cooksey/ai-terminal-pty pty-wrapper
git submodule add https://github.com/cooksey/ai-terminal-backend ai-backend

# Clone elsewhere
git clone --recursive https://github.com/cooksey/ai-terminal
```

**Downside**: More complex, harder to coordinate changes across both.

## Option 3: Separate Repos, No Coupling

**Best for**: When they're truly independent tools.

```
ai-terminal-pty/        # Standalone Rust binary
├── Can work with any backend
└── Looks for "ai-backend" in PATH

ai-terminal-backend/    # Standalone Go binary
├── Can be used by multiple frontends
└── Listens on socket or reads stdin
```

Distribute separately:
```bash
# Install Rust wrapper
cargo install ai-terminal-pty

# Install Go backend
go install github.com/cooksey/ai-terminal-backend@latest
```

Users install both, and Rust wrapper finds Go backend via PATH.

## Recommendation: Monorepo (Option 1)

**Why**:
1. **Single `git clone`** - everything in one place
2. **Single `make` command** - builds both
3. **Easier development** - change both together
4. **Simpler releases** - version them together
5. **Clear dependency** - Rust wrapper expects Go backend

**When to use separate repos**:
- You want independent versioning
- Multiple teams working on different components
- Backend used by other projects too
- You want to publish to cargo/go separately

## Detailed Monorepo Setup

### 1. Create Structure

```bash
cd ~/llm-workspace
mkdir ai-terminal
cd ai-terminal

# Initialize git
git init
```

### 2. Create Rust Workspace

```bash
# Workspace root Cargo.toml
cat > Cargo.toml << 'EOF'
[workspace]
members = ["pty-wrapper"]
resolver = "2"

[workspace.dependencies]
portable-pty = "0.8"
tokio = { version = "1", features = ["full"] }
serde = { version = "1", features = ["derive"] }
serde_json = "1"
anyhow = "1"
nix = { version = "0.27", features = ["term", "signal"] }
EOF
```

### 3. Create Rust Project

```bash
cargo new pty-wrapper

# Update pty-wrapper/Cargo.toml to use workspace deps
cat > pty-wrapper/Cargo.toml << 'EOF'
[package]
name = "pty-wrapper"
version = "0.1.0"
edition = "2021"

[dependencies]
portable-pty = { workspace = true }
tokio = { workspace = true }
serde = { workspace = true }
serde_json = { workspace = true }
anyhow = { workspace = true }
nix = { workspace = true }
EOF
```

### 4. Create Go Project

```bash
mkdir ai-backend
cd ai-backend

go mod init github.com/cooksey/ai-terminal/ai-backend

# Optional: add Ollama client dependency
go get github.com/ollama/ollama/api
```

### 5. Create Makefile

```bash
cd ..
cat > Makefile << 'EOF'
.PHONY: all build-rust build-go clean run dev install test

# Default target
all: build

# Build everything in release mode
build: build-rust build-go

build-rust:
	@echo "Building Rust PTY wrapper..."
	cargo build --release

build-go:
	@echo "Building Go AI backend..."
	cd ai-backend && go build -o ../target/release/ai-backend

# Development build (faster compilation)
dev: dev-rust dev-go

dev-rust:
	@echo "Building Rust (dev mode)..."
	cargo build

dev-go:
	@echo "Building Go (dev mode)..."
	cd ai-backend && go build -o ../target/debug/ai-backend

# Clean build artifacts
clean:
	@echo "Cleaning..."
	cargo clean
	rm -f ai-backend/ai-backend

# Run the application
run: dev
	./target/debug/pty-wrapper

# Install system-wide
install: build
	@echo "Installing to /usr/local/bin..."
	sudo cp target/release/pty-wrapper /usr/local/bin/ai-terminal
	sudo cp target/release/ai-backend /usr/local/bin/ai-terminal-backend

# Uninstall
uninstall:
	@echo "Uninstalling..."
	sudo rm -f /usr/local/bin/ai-terminal
	sudo rm -f /usr/local/bin/ai-terminal-backend

# Run tests
test: test-rust test-go

test-rust:
	cargo test

test-go:
	cd ai-backend && go test ./...

# Format code
fmt:
	cargo fmt
	cd ai-backend && go fmt ./...

# Check code (lint)
check:
	cargo clippy
	cd ai-backend && go vet ./...
EOF
```

### 6. Create README

```bash
cat > README.md << 'EOF'
# AI Terminal

An AI-powered terminal wrapper that provides intelligent assistance.

## Architecture

- **pty-wrapper** (Rust): Terminal emulation and I/O handling
- **ai-backend** (Go): LLM integration via Ollama

## Building

```bash
make build
```

## Running

```bash
make run
```

Press `Ctrl-G` to activate AI mode.

## Installing

```bash
make install
```

Then run `ai-terminal` from anywhere.

## Development

```bash
# Quick dev build (faster)
make dev

# Run tests
make test

# Format code
make fmt

# Lint
make check
```

## Project Structure

```
ai-terminal/
├── pty-wrapper/      # Rust PTY wrapper
└── ai-backend/       # Go AI backend
```
EOF
```

### 7. Create .gitignore

```bash
cat > .gitignore << 'EOF'
# Rust
target/
**/*.rs.bk
*.pdb

# Go
ai-backend/ai-backend
*.exe
*.exe~
*.dll
*.so
*.dylib
*.test
*.out

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
Thumbs.db

# Logs
*.log
EOF
```

### 8. Verify Structure

```bash
tree -L 2
```

Should show:
```
ai-terminal/
├── Cargo.toml
├── Makefile
├── README.md
├── .gitignore
├── pty-wrapper/
│   ├── Cargo.toml
│   └── src/
└── ai-backend/
    ├── go.mod
    └── main.go
```

## Binary Location Strategy

### During Development

```rust
// pty-wrapper/src/ai_client.rs
fn get_backend_path() -> PathBuf {
    // Try these in order:
    // 1. Same directory as executable
    // 2. ../target/debug/ai-backend (dev)
    // 3. ../target/release/ai-backend (release)
    // 4. "ai-backend" in PATH
    
    let exe_dir = env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(|p| p.to_path_buf()));
    
    if let Some(dir) = exe_dir {
        // Check same directory first
        let path = dir.join("ai-backend");
        if path.exists() {
            return path;
        }
    }
    
    // Fall back to PATH
    PathBuf::from("ai-backend")
}
```

### After Installation

Both binaries go to `/usr/local/bin/`:
```bash
/usr/local/bin/ai-terminal          # Rust wrapper
/usr/local/bin/ai-terminal-backend  # Go backend
```

Rust wrapper can just spawn by name:
```rust
Command::new("ai-terminal-backend")
```

## Development Workflow

```bash
# Initial setup
cd ~/llm-workspace/ai-terminal
make dev

# Work on Rust
cd pty-wrapper
cargo check
cargo test

# Work on Go
cd ../ai-backend
go run main.go
go test

# Test together
cd ..
make run

# Ready to release
make build
./target/release/pty-wrapper
```

## Git Workflow

```bash
# Single repo, simple workflow
git add .
git commit -m "Add feature X"
git push

# Tag releases
git tag v0.1.0
git push --tags
```

That's it! Everything versioned together, builds with one command, easy to develop.
