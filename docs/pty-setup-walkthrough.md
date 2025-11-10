# PTY Setup Walkthrough

## Overview

We'll build a minimal PTY wrapper in incremental steps, testing each piece as we go.

## Step 1: Project Setup

### Initialize Go Module

```bash
mkdir pty-wrapper
cd pty-wrapper
go mod init github.com/cooksey/pty-wrapper
```

### Install PTY Library

```bash
go get github.com/creack/pty
```

This is the de facto standard Go PTY library. It handles the OS-specific details (ioctl calls, terminal setup, etc.).

## Step 2: Minimal PTY Pass-Through

### Goal
Create a program that wraps your shell but doesn't do anything special yet - just passes input/output through transparently.

### File: `main.go`

```go
package main

import (
	"io"
	"os"
	"os/exec"
	"os/signal"
	"syscall"

	"github.com/creack/pty"
)

func main() {
	// Get the user's shell from environment
	shell := os.Getenv("SHELL")
	if shell == "" {
		shell = "/bin/sh" // Fallback
	}

	// Create command to run shell
	cmd := exec.Command(shell)
	
	// Inherit environment variables
	cmd.Env = os.Environ()

	// Start the shell with a PTY
	// This creates the master/slave pair and starts the command
	ptmx, err := pty.Start(cmd)
	if err != nil {
		panic(err)
	}
	defer func() { _ = ptmx.Close() }()

	// Handle window size changes
	// When you resize your terminal, we need to tell the PTY
	ch := make(chan os.Signal, 1)
	signal.Notify(ch, syscall.SIGWINCH)
	go func() {
		for range ch {
			if err := pty.InheritSize(os.Stdin, ptmx); err != nil {
				// Log error but don't crash
			}
		}
	}()
	
	// Set initial window size
	_ = pty.InheritSize(os.Stdin, ptmx)

	// Set stdin in raw mode so we get individual keystrokes
	// Without this, the OS buffers until you hit Enter
	oldState, err := setRawMode(os.Stdin)
	if err != nil {
		panic(err)
	}
	defer func() { _ = restoreMode(os.Stdin, oldState) }()

	// Copy data between user terminal and PTY
	// This is the core of the wrapper
	go func() {
		// User input → PTY
		io.Copy(ptmx, os.Stdin)
	}()
	
	// PTY output → User terminal (blocks until shell exits)
	io.Copy(os.Stdout, ptmx)

	// Wait for shell to exit
	_ = cmd.Wait()
}

// setRawMode puts the terminal in raw mode
// This means we get keystrokes immediately, not line-buffered
func setRawMode(f *os.File) (*syscall.Termios, error) {
	fd := int(f.Fd())
	
	// Get current terminal settings
	oldState, err := syscall.IoctlGetTermios(fd, syscall.TIOCGETA)
	if err != nil {
		return nil, err
	}

	// Make a copy and modify it
	newState := *oldState
	
	// Disable canonical mode (line buffering) and echo
	newState.Lflag &^= syscall.ICANON | syscall.ECHO | syscall.ISIG
	
	// Set minimum bytes to return on read
	newState.Cc[syscall.VMIN] = 1
	newState.Cc[syscall.VTIME] = 0

	// Apply new settings
	if err := syscall.IoctlSetTermios(fd, syscall.TIOCSETA, &newState); err != nil {
		return nil, err
	}

	return oldState, nil
}

// restoreMode restores terminal to previous state
func restoreMode(f *os.File, state *syscall.Termios) error {
	return syscall.IoctlSetTermios(int(f.Fd()), syscall.TIOCSETA, state)
}
```

### What's Happening Here?

1. **`pty.Start(cmd)`**: Creates master/slave PTY pair, launches shell on slave side
2. **Raw mode**: Lets us see individual keystrokes instead of waiting for Enter
3. **Window size**: Forwards terminal resize events to PTY
4. **`io.Copy`**: Shuttles bytes between user terminal and PTY

### Test It

```bash
go build -o wrapper
./wrapper
```

You should see your shell prompt. Try:
- Typing commands
- Running `ls`, `pwd`, etc.
- Using arrow keys, Ctrl-C
- Opening `vim` or `less`

It should feel exactly like your normal shell. That's the goal - transparent pass-through.

## Step 3: Add Trigger Detection

### Goal
Detect when user presses Ctrl-G (or another trigger) to invoke AI mode.

### Update `main.go`

Replace the simple `io.Copy(ptmx, os.Stdin)` with:

```go
func main() {
	// ... (previous setup code remains the same)

	// Instead of simple io.Copy, intercept input
	go handleUserInput(ptmx, os.Stdin, os.Stdout)
	
	// PTY output → User terminal (unchanged)
	io.Copy(os.Stdout, ptmx)

	_ = cmd.Wait()
}

// handleUserInput intercepts user keystrokes
func handleUserInput(ptmx *os.File, stdin io.Reader, stdout io.Writer) {
	buf := make([]byte, 1024)
	
	for {
		n, err := stdin.Read(buf)
		if err != nil {
			return
		}

		data := buf[:n]

		// Check if this is our AI trigger
		// Ctrl-G is ASCII 0x07 (bell character)
		if len(data) == 1 && data[0] == 0x07 {
			handleAIMode(ptmx, stdin, stdout)
			continue
		}

		// Not AI trigger - pass through to shell
		_, err = ptmx.Write(data)
		if err != nil {
			return
		}
	}
}

// handleAIMode is called when user triggers AI
func handleAIMode(ptmx *os.File, stdin io.Reader, stdout io.Writer) {
	// For now, just show a message
	stdout.Write([]byte("\r\n🤖 AI mode activated! (not implemented yet)\r\n"))
	
	// Could prompt for input here
	// stdout.Write([]byte("AI> "))
	// query := readLine(stdin, stdout)
	// ... call LLM ...
}
```

### What Changed?

- **`handleUserInput`**: Reads keystrokes one at a time, checks for Ctrl-G
- **`handleAIMode`**: Placeholder for AI functionality
- **`\r\n`**: Carriage return + newline (required in raw mode)

### Test It

```bash
go build -o wrapper
./wrapper
```

Press **Ctrl-G** - you should see the AI message appear.

### Alternative Triggers

**Ctrl-G** (0x07):
```go
if len(data) == 1 && data[0] == 0x07 { ... }
```

**Double Escape** (ESC ESC):
```go
var lastKey byte
if len(data) == 1 && data[0] == 0x1b && lastKey == 0x1b { ... }
lastKey = data[0]
```

**Prefix like `/ai`**:
```go
var inputBuffer string
inputBuffer += string(data)

if strings.HasPrefix(inputBuffer, "/ai ") {
	// Extract query
	query := strings.TrimPrefix(inputBuffer, "/ai ")
	handleAIRequest(query)
	inputBuffer = ""
}
```

## Step 4: Read User Query

### Goal
When AI mode is triggered, read a line of input from the user.

### Add `readLine` Function

```go
// readLine reads a line of text from user
// Handles backspace, allows editing
func readLine(stdin io.Reader, stdout io.Writer) string {
	var line []byte
	buf := make([]byte, 1)

	for {
		_, err := stdin.Read(buf)
		if err != nil {
			return string(line)
		}

		char := buf[0]

		switch char {
		case 0x0d, 0x0a: // Enter (carriage return or newline)
			stdout.Write([]byte("\r\n"))
			return string(line)

		case 0x7f, 0x08: // Backspace or Delete
			if len(line) > 0 {
				line = line[:len(line)-1]
				// Move cursor back, write space, move back again
				stdout.Write([]byte("\b \b"))
			}

		case 0x03: // Ctrl-C
			stdout.Write([]byte("^C\r\n"))
			return ""

		case 0x1b: // Escape
			stdout.Write([]byte("\r\n"))
			return ""

		default:
			// Regular character
			if char >= 32 && char <= 126 {
				line = append(line, char)
				stdout.Write(buf) // Echo character
			}
		}
	}
}
```

### Update `handleAIMode`

```go
func handleAIMode(ptmx *os.File, stdin io.Reader, stdout io.Writer) {
	stdout.Write([]byte("\r\n🤖 AI> "))
	
	query := readLine(stdin, stdout)
	
	if query == "" {
		return // User cancelled
	}

	// Placeholder response
	stdout.Write([]byte("You asked: " + query + "\r\n"))
	stdout.Write([]byte("(LLM call would happen here)\r\n"))
}
```

### Test It

Press **Ctrl-G**, type a query, hit Enter. You should see it echoed back.

## Step 5: Capture Context

### Goal
Track what's happening in the shell so we can provide context to the LLM.

### Add Session State

```go
type Session struct {
	workingDir     string
	commandHistory []string
	lastOutput     []byte
	outputBuffer   *bytes.Buffer
}

func NewSession() *Session {
	wd, _ := os.Getwd()
	return &Session{
		workingDir:   wd,
		outputBuffer: &bytes.Buffer{},
	}
}

// captureOutput wraps io.Copy to save output for context
func (s *Session) captureOutput(dst io.Writer, src io.Reader) {
	buf := make([]byte, 4096)
	for {
		n, err := src.Read(buf)
		if err != nil {
			return
		}

		data := buf[:n]

		// Save to buffer (keep last 10KB)
		s.outputBuffer.Write(data)
		if s.outputBuffer.Len() > 10*1024 {
			// Trim to last 10KB
			s.outputBuffer.Next(s.outputBuffer.Len() - 10*1024)
		}

		// Forward to user
		dst.Write(data)
	}
}

// GetContext returns current context for LLM
func (s *Session) GetContext() map[string]string {
	return map[string]string{
		"working_dir":  s.workingDir,
		"last_output":  s.outputBuffer.String(),
		"shell":        os.Getenv("SHELL"),
		"user":         os.Getenv("USER"),
	}
}
```

### Update `main.go`

```go
func main() {
	// ... (setup code)

	session := NewSession()

	// Pass session to handlers
	go handleUserInput(ptmx, os.Stdin, os.Stdout, session)
	
	// Use captureOutput instead of io.Copy
	session.captureOutput(os.Stdout, ptmx)

	_ = cmd.Wait()
}

func handleUserInput(ptmx *os.File, stdin io.Reader, stdout io.Writer, session *Session) {
	// ... (same as before, but pass session to handleAIMode)
	handleAIMode(ptmx, stdin, stdout, session)
}

func handleAIMode(ptmx *os.File, stdin io.Reader, stdout io.Writer, session *Session) {
	stdout.Write([]byte("\r\n🤖 AI> "))
	query := readLine(stdin, stdout)

	if query == "" {
		return
	}

	// Get context
	ctx := session.GetContext()
	
	stdout.Write([]byte("Query: " + query + "\r\n"))
	stdout.Write([]byte("Working Dir: " + ctx["working_dir"] + "\r\n"))
	stdout.Write([]byte("Last Output (truncated): " + ctx["last_output"][:100] + "...\r\n"))
}
```

### Test It

1. Run `./wrapper`
2. Execute some commands: `ls`, `pwd`, `echo "hello"`
3. Press **Ctrl-G** and enter a query
4. You should see the context captured (working dir, recent output)

## Step 6: Add LLM Call (Skeleton)

### Goal
Show where the LLM integration would go.

### Add AI Client Structure

```go
package main

// AIClient handles LLM API calls
type AIClient struct {
	apiKey string
	model  string
}

func NewAIClient(apiKey string) *AIClient {
	return &AIClient{
		apiKey: apiKey,
		model:  "claude-3-5-sonnet-20241022",
	}
}

// Query sends a prompt to the LLM
func (c *AIClient) Query(prompt string, context map[string]string) (string, error) {
	// TODO: Implement actual API call
	// For now, just return a mock response
	
	return "I would help with: " + prompt, nil
}
```

### Update `handleAIMode`

```go
func handleAIMode(ptmx *os.File, stdin io.Reader, stdout io.Writer, session *Session, aiClient *AIClient) {
	stdout.Write([]byte("\r\n🤖 AI> "))
	query := readLine(stdin, stdout)

	if query == "" {
		return
	}

	// Show thinking indicator
	stdout.Write([]byte("Thinking...\r\n"))

	// Get context and call AI
	ctx := session.GetContext()
	response, err := aiClient.Query(query, ctx)
	
	if err != nil {
		stdout.Write([]byte("Error: " + err.Error() + "\r\n"))
		return
	}

	// Display response
	stdout.Write([]byte("\r\n" + response + "\r\n\r\n"))

	// If response contains a command suggestion, optionally execute
	// if cmd := extractCommand(response); cmd != "" {
	//     stdout.Write([]byte("Execute: " + cmd + "? [y/N] "))
	//     if readLine(stdin, stdout) == "y" {
	//         ptmx.Write([]byte(cmd + "\n"))
	//     }
	// }
}
```

### Update `main.go`

```go
func main() {
	// ... (setup code)

	session := NewSession()
	aiClient := NewAIClient(os.Getenv("ANTHROPIC_API_KEY"))

	go handleUserInput(ptmx, os.Stdin, os.Stdout, session, aiClient)
	
	session.captureOutput(os.Stdout, ptmx)

	_ = cmd.Wait()
}
```

## Step 7: Signal Handling

### Goal
Properly handle signals like Ctrl-C, Ctrl-Z so they reach the shell.

### Add Signal Forwarding

```go
func main() {
	// ... (after pty.Start)

	// Forward signals to shell process
	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM, syscall.SIGQUIT)
	
	go func() {
		for sig := range sigCh {
			// Forward to shell's process group
			if cmd.Process != nil {
				cmd.Process.Signal(sig)
			}
		}
	}()

	// ... (rest of main)
}
```

## Complete Minimal Structure

After all steps, your project structure:

```
pty-wrapper/
├── go.mod
├── go.sum
├── main.go          # Entry point, PTY setup, signal handling
├── session.go       # Session state and context tracking
├── input.go         # User input handling and line reading
├── ai.go            # AI client (LLM API calls)
└── README.md
```

## Build and Use

```bash
# Build
go build -o ai-wrapper

# Run directly
./ai-wrapper

# Or add to PATH
sudo cp ai-wrapper /usr/local/bin/
ai-wrapper

# Or alias in your shell config
alias aiterm="~/pty-wrapper/ai-wrapper"
```

## Testing Checklist

- [ ] Basic shell commands work (`ls`, `pwd`, `cd`)
- [ ] Colors and formatting preserved
- [ ] Ctrl-C interrupts running commands
- [ ] Ctrl-D exits shell
- [ ] Ctrl-G triggers AI mode
- [ ] Can type and edit AI queries
- [ ] Window resize works (try `echo $COLUMNS`)
- [ ] Full-screen programs work (`vim`, `less`, `htop`)
- [ ] Background jobs work (`sleep 10 &`)

## Next Steps After Minimal Version Works

1. **Implement real LLM API calls** (Anthropic, OpenAI)
2. **Add tool execution** (file operations, grep, etc.)
3. **Better output parsing** (ANSI code handling)
4. **Command history tracking** (detect prompts vs output)
5. **Configuration file** (API keys, preferences)
6. **Error handling and logging**
7. **Installation script**

## Common Issues

**Issue**: Backspace doesn't work  
**Fix**: Make sure you're in raw mode and handling 0x7f

**Issue**: Vim/less doesn't work  
**Fix**: Ensure window size is being forwarded with `pty.InheritSize`

**Issue**: Shell prompt looks weird  
**Fix**: You're in raw mode - this is normal, PTY handles it

**Issue**: Can't exit  
**Fix**: Ctrl-D should work. If not, check that signals are forwarded

## Key Files to Reference

- `github.com/creack/pty` examples: https://github.com/creack/pty/tree/master/examples
- tmux source (terminal handling): https://github.com/tmux/tmux
- Go syscall package docs: https://pkg.go.dev/syscall

The beauty of this approach: start with ~100 lines of very simple code (step 2), test it works, then add features incrementally!
