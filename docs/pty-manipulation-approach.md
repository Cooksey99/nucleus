# PTY Manipulation Approach for Terminal AI Layer

## What is a PTY?

**PTY (Pseudo-Terminal)** is a pair of virtual character devices:
- **Master side**: Your AI layer controls this
- **Slave side**: The shell and programs run here

It simulates a real terminal, allowing programs to think they're running in an interactive terminal while you intercept and modify the I/O stream.

## Architecture

```
┌─────────────────────────────────────────────────┐
│              User's Terminal                     │
│            (iTerm, kitty, etc)                   │
└────────────────────┬────────────────────────────┘
                     │
                     │ stdin/stdout
                     │
┌────────────────────▼────────────────────────────┐
│           AI Wrapper Process (Go)                │
│  ┌────────────────────────────────────────────┐ │
│  │  Input Parser & Command Detector           │ │
│  │  - Detect AI invocation (special prefix)   │ │
│  │  - Parse natural language intent           │ │
│  └────────────────────────────────────────────┘ │
│  ┌────────────────────────────────────────────┐ │
│  │  LLM Client                                │ │
│  │  - API calls to Claude/OpenAI              │ │
│  │  - Tool execution (file ops, search, etc)  │ │
│  │  - Context management                      │ │
│  └────────────────────────────────────────────┘ │
│  ┌────────────────────────────────────────────┐ │
│  │  PTY Master                                │ │
│  │  - Read from/write to PTY                  │ │
│  │  - Buffer management                       │ │
│  └────────────────────────────────────────────┘ │
└────────────────────┬────────────────────────────┘
                     │
                     │ PTY Master ↔ PTY Slave
                     │
┌────────────────────▼────────────────────────────┐
│              PTY Slave (Shell)                   │
│                   zsh/bash                       │
└────────────────────┬────────────────────────────┘
                     │
                     │ Commands
                     │
┌────────────────────▼────────────────────────────┐
│              System/Programs                     │
└──────────────────────────────────────────────────┘
```

## How It Actually Works

### 1. Create PTY Pair

```go
package main

import (
    "github.com/creack/pty"
    "os"
    "os/exec"
)

func main() {
    // Start the user's shell
    cmd := exec.Command(os.Getenv("SHELL"))
    
    // Create PTY
    ptmx, err := pty.Start(cmd)
    if err != nil {
        panic(err)
    }
    defer ptmx.Close()
    
    // Now ptmx is the master side
    // cmd runs in the slave side
    
    // Your wrapper sits in the middle
    handleIO(ptmx, os.Stdin, os.Stdout)
}
```

### 2. Intercept and Route I/O

```go
func handleIO(ptmx *os.File, stdin io.Reader, stdout io.Writer) {
    // Goroutine 1: User input → AI layer → PTY
    go func() {
        buf := make([]byte, 1024)
        for {
            n, err := stdin.Read(buf)
            if err != nil {
                return
            }
            
            input := string(buf[:n])
            
            // Check if AI invocation (e.g., starts with "/ai")
            if isAICommand(input) {
                handleAIRequest(input, ptmx, stdout)
            } else {
                // Pass through to shell
                ptmx.Write(buf[:n])
            }
        }
    }()
    
    // Goroutine 2: PTY output → User terminal
    go func() {
        io.Copy(stdout, ptmx)
    }()
    
    // Wait for shell to exit
    cmd.Wait()
}
```

### 3. AI Command Detection

```go
func isAICommand(input string) bool {
    // Detect special prefix (e.g., "/ai", "ai:", ctrl-g)
    trimmed := strings.TrimSpace(input)
    return strings.HasPrefix(trimmed, "/ai ") ||
           strings.HasPrefix(trimmed, "ai:") ||
           input[0] == 0x07 // Bell character (ctrl-g)
}

func handleAIRequest(input string, ptmx *os.File, stdout io.Writer) {
    // Parse user intent
    query := strings.TrimPrefix(input, "/ai ")
    
    // Call LLM
    response := callLLM(query)
    
    // Display response to user
    fmt.Fprintf(stdout, "\n🤖 AI: %s\n\n", response)
    
    // If AI suggests a command, optionally execute it
    if response.SuggestedCommand != "" {
        // Show command
        fmt.Fprintf(stdout, "$ %s\n", response.SuggestedCommand)
        
        // Execute in PTY
        ptmx.Write([]byte(response.SuggestedCommand + "\n"))
    }
}
```

### 4. Context Awareness

```go
type Session struct {
    WorkingDir    string
    CommandHistory []string
    LastOutput    string
    Environment   map[string]string
}

func (s *Session) captureOutput(ptmx *os.File, stdout io.Writer) {
    buf := make([]byte, 4096)
    for {
        n, _ := ptmx.Read(buf)
        output := string(buf[:n])
        
        // Store for AI context
        s.LastOutput += output
        
        // Parse for directory changes
        if strings.Contains(output, "cd ") {
            s.updateWorkingDir()
        }
        
        // Forward to user
        stdout.Write(buf[:n])
    }
}
```

## Full Implementation Example

### Project Structure

```
ai-terminal-wrapper/
├── main.go              # Entry point, PTY setup
├── pty/
│   ├── handler.go       # PTY I/O management
│   └── session.go       # Session state tracking
├── ai/
│   ├── client.go        # LLM API client
│   ├── tools.go         # Tool implementations
│   └── context.go       # Context builder
├── parser/
│   ├── intent.go        # Parse user intent
│   └── commands.go      # Command detection
└── ui/
    ├── renderer.go      # Format AI responses
    └── prompts.go       # Interactive prompts
```

### main.go

```go
package main

import (
    "github.com/creack/pty"
    "os"
    "os/exec"
    "syscall"
)

func main() {
    shell := os.Getenv("SHELL")
    if shell == "" {
        shell = "/bin/bash"
    }
    
    cmd := exec.Command(shell)
    cmd.Env = append(os.Environ(), "AI_WRAPPER=1")
    
    // Start shell with PTY
    ptmx, err := pty.Start(cmd)
    if err != nil {
        panic(err)
    }
    defer ptmx.Close()
    
    // Set terminal to raw mode
    oldState, err := setupRawMode(os.Stdin.Fd())
    if err != nil {
        panic(err)
    }
    defer syscall.SetTermios(int(os.Stdin.Fd()), oldState)
    
    // Handle window size changes
    go handleWindowSize(ptmx)
    
    // Start I/O handling
    session := NewSession()
    session.Run(ptmx, os.Stdin, os.Stdout)
    
    cmd.Wait()
}

func setupRawMode(fd uintptr) (*syscall.Termios, error) {
    termios, err := syscall.IoctlGetTermios(int(fd), syscall.TIOCGETA)
    if err != nil {
        return nil, err
    }
    
    newTermios := *termios
    newTermios.Lflag &^= syscall.ECHO | syscall.ICANON
    
    if err := syscall.IoctlSetTermios(int(fd), syscall.TIOCSETA, &newTermios); err != nil {
        return nil, err
    }
    
    return termios, nil
}
```

### pty/session.go

```go
package pty

import (
    "io"
    "os"
)

type Session struct {
    workingDir     string
    commandHistory []string
    lastOutput     string
    aiMode         bool
    buffer         []byte
}

func NewSession() *Session {
    wd, _ := os.Getwd()
    return &Session{
        workingDir: wd,
        buffer:     make([]byte, 0, 4096),
    }
}

func (s *Session) Run(ptmx *os.File, stdin io.Reader, stdout io.Writer) {
    // Channel for coordinating I/O
    done := make(chan bool)
    
    // User input → PTY
    go s.handleUserInput(ptmx, stdin, stdout, done)
    
    // PTY output → User
    go s.handleShellOutput(ptmx, stdout, done)
    
    <-done
}

func (s *Session) handleUserInput(ptmx *os.File, stdin io.Reader, stdout io.Writer, done chan bool) {
    buf := make([]byte, 1024)
    for {
        n, err := stdin.Read(buf)
        if err != nil {
            done <- true
            return
        }
        
        data := buf[:n]
        
        // Detect AI trigger (e.g., Ctrl-G)
        if len(data) == 1 && data[0] == 0x07 {
            s.enterAIMode(ptmx, stdin, stdout)
            continue
        }
        
        // Pass through to shell
        ptmx.Write(data)
    }
}

func (s *Session) handleShellOutput(ptmx *os.File, stdout io.Writer, done chan bool) {
    buf := make([]byte, 4096)
    for {
        n, err := ptmx.Read(buf)
        if err != nil {
            done <- true
            return
        }
        
        // Capture output for AI context
        s.lastOutput = string(buf[:n])
        
        // Forward to user terminal
        stdout.Write(buf[:n])
    }
}

func (s *Session) enterAIMode(ptmx *os.File, stdin io.Reader, stdout io.Writer) {
    // Display AI prompt
    stdout.Write([]byte("\n🤖 AI> "))
    
    // Read user query
    query := s.readLine(stdin, stdout)
    
    // Build context
    ctx := s.buildContext()
    
    // Call AI
    response := callAI(query, ctx)
    
    // Display and optionally execute
    s.displayAIResponse(response, ptmx, stdout)
}
```

## Invocation Methods

### Option 1: Special Prefix

```bash
$ /ai list all .go files in this directory
🤖 AI: ls *.go
$ ls *.go
main.go  session.go  handler.go
```

### Option 2: Keyboard Shortcut

```bash
$ <Ctrl-G>
🤖 AI> list all .go files
AI: ls *.go
Execute? [Y/n]: y
main.go  session.go  handler.go
```

### Option 3: Natural Language Fallback

```bash
$ show me all go files
bash: show: command not found
🤖 Did you mean: ls *.go? [Y/n]
```

## Pros and Cons

### Pros

**Universal Compatibility**: Works with any terminal emulator

**No Terminal Switch**: Users keep their preferred terminal (iTerm, kitty, Alacritty)

**Transparent**: Shell and programs don't know they're wrapped

**Full Context**: Can see all I/O, track state, command history

**Flexible UI**: Can inject AI responses anywhere in the stream

**Portable**: Single binary that works across Unix-like systems

**Testing**: Easy to test - just feed it input and check output

### Cons

**Complexity**: PTY programming is tricky (terminal control codes, buffering)

**Terminal Emulation**: Must handle ANSI escape codes, resize events, signals

**Race Conditions**: Coordinating multiple goroutines reading/writing

**Raw Mode**: Must set terminal to raw mode (disables line editing temporarily)

**Screen-based Programs**: Programs like `vim`, `less`, `htop` use alternate screen - harder to intercept

**Performance**: Extra layer adds latency (though minimal if done right)

**Signal Handling**: Must properly forward signals (Ctrl-C, Ctrl-Z) to shell

**Lost Features**: Harder to implement rich UI like Warp's blocks and search

## Technical Challenges

### 1. ANSI Escape Sequences

Shell output contains control codes:

```
\x1b[32m  # Green color
\x1b[2J   # Clear screen
\x1b[H    # Move cursor home
```

Must parse these to know actual content vs. formatting.

### 2. Terminal Size

When terminal resizes, must forward to PTY:

```go
func handleWindowSize(ptmx *os.File) {
    ch := make(chan os.Signal, 1)
    signal.Notify(ch, syscall.SIGWINCH)
    
    for range ch {
        if err := pty.InheritSize(os.Stdin, ptmx); err != nil {
            log.Printf("resize error: %v", err)
        }
    }
}
```

### 3. Signal Propagation

Ctrl-C, Ctrl-Z must reach shell:

```go
func handleSignals(cmd *exec.Cmd) {
    ch := make(chan os.Signal, 1)
    signal.Notify(ch, syscall.SIGINT, syscall.SIGTERM)
    
    go func() {
        for sig := range ch {
            cmd.Process.Signal(sig)
        }
    }()
}
```

### 4. Alternate Screen

Programs like `vim` use alternate screen buffer. Must detect and handle:

```go
// Detect alternate screen enter
if bytes.Contains(output, []byte("\x1b[?1049h")) {
    inAlternateScreen = true
    // Don't intercept output
}
```

## Libraries and Tools

### Go

- `github.com/creack/pty` - Best Go PTY library
- `github.com/gliderlabs/ssh` - If adding SSH support
- `github.com/gdamore/tcell` - Terminal UI (if adding TUI)

### Rust

- `portable-pty` - Cross-platform PTY
- `nix` crate - Unix primitives
- `crossterm` - Terminal manipulation

### Testing

- `expect` scripts - Automate terminal interaction
- `tmux` - Can use as reference implementation
- `script` command - Record/replay terminal sessions

## Minimal Viable Implementation

Start simple:

```go
// main.go - ~100 lines
func main() {
    cmd := exec.Command(os.Getenv("SHELL"))
    ptmx, _ := pty.Start(cmd)
    defer ptmx.Close()
    
    go io.Copy(os.Stdout, ptmx)  // PTY → User
    
    go func() {
        buf := make([]byte, 1)
        for {
            os.Stdin.Read(buf)
            if buf[0] == 0x07 {  // Ctrl-G
                handleAI(ptmx)
            } else {
                ptmx.Write(buf)  // User → PTY
            }
        }
    }()
    
    cmd.Wait()
}
```

Then incrementally add:
1. Context tracking
2. Command detection
3. Better UI
4. Error handling
5. Signal handling
6. Window resize

## Comparison to tmux

`tmux` is essentially a PTY multiplexer - your AI wrapper would be similar but simpler:

**tmux**:
- Manages multiple PTYs
- Persists sessions
- Complex terminal emulation
- ~50k lines of C

**AI wrapper**:
- Single PTY
- No persistence needed (initially)
- Minimal terminal handling
- ~2-3k lines of Go

## Next Steps

1. Build minimal PTY wrapper (100 lines)
2. Add AI trigger (Ctrl-G detection)
3. Integrate LLM API call
4. Add context capture (cwd, last output)
5. Handle signals and resize
6. Polish UI and error handling

The PTY approach is very achievable - it's how terminal multiplexers, terminal recorders, and expect work. The hard parts are the details (signals, escape codes, edge cases), not the core concept.
