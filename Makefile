.PHONY: all build clean run install help

BINARY_NAME=llm-workspace
AI_SERVER=llm-server
PTY_DIR=pty
AI_DIR=ai

all: build

build: build-ai build-pty

build-ai:
	@echo "Building AI server..."
	cd $(AI_DIR) && go build -o ../$(AI_SERVER) .

build-pty:
	@echo "Building PTY..."
	cd $(PTY_DIR) && cargo build --release
	@cp target/release/pty ./$(BINARY_NAME)

clean:
	@echo "Cleaning..."
	rm -f $(AI_SERVER) $(BINARY_NAME)
	cd $(PTY_DIR) && cargo clean
	cd $(AI_DIR) && go clean

run: build
	@./$(BINARY_NAME)

install: build
	@echo "Installing to /usr/local/bin..."
	@cp $(BINARY_NAME) /usr/local/bin/
	@cp $(AI_SERVER) /usr/local/bin/
	@echo "Done! You can now run '$(BINARY_NAME)' from anywhere"

dev: build-ai
	cd $(PTY_DIR) && cargo run

help:
	@echo "Available targets:"
	@echo "  make build    - Build both AI server and PTY"
	@echo "  make clean    - Clean build artifacts"
	@echo "  make run      - Build and run"
	@echo "  make dev      - Quick development build (debug mode)"
	@echo "  make install  - Install to /usr/local/bin"
