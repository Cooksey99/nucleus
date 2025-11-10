.PHONY: help setup install-deps pull-model pull-embedding-model pull-all-models run build clean clean-all clean-data test

help:
	@echo "Available commands:"
	@echo "  make setup               - Complete first-time setup"
	@echo "  make install-deps        - Install Go dependencies"
	@echo "  make pull-model          - Pull the LLM model from config.yaml"
	@echo "  make pull-embedding-model - Pull the embedding model from config.yaml"
	@echo "  make pull-all-models     - Pull both LLM and embedding models"
	@echo "  make run                 - Run the application"
	@echo "  make build               - Build binary"
	@echo "  make clean               - Remove built files only (SAFE - keeps your data)"
	@echo "  make clean-data          - Remove vector DB and chat history (DESTRUCTIVE)"
	@echo "  make clean-all           - Remove everything including data (DESTRUCTIVE)"
	@echo "  make test                - Run tests"

setup: check-ollama install-deps pull-all-models
	@echo "Setup complete! Run 'make run' to start."

check-ollama:
	@which ollama > /dev/null || (echo "Ollama not found. Installing via Homebrew..." && brew install ollama)
	@echo "Ollama found"
	@pgrep -x ollama > /dev/null || (echo "🔧 Starting Ollama..." && ollama serve > /dev/null 2>&1 &)
	@sleep 2

install-deps:
	@echo "Installing Go dependencies..."
	@test -f go.mod || go mod init llm-workspace
	@go get github.com/ollama/ollama/api
	@go get github.com/philippgille/chromem-go
	@go get gopkg.in/yaml.v3
	@go mod tidy
	@echo "Dependencies installed"

pull-model:
	@echo "Reading model from config.yaml..."
	@MODEL=$$(grep 'model:' config.yaml | head -1 | awk '{print $$2}' | tr -d '"'); \
	echo "Pulling LLM model: $$MODEL..."; \
	ollama pull $$MODEL
	@echo "LLM model ready"

pull-embedding-model:
	@echo "Reading embedding model from config.yaml..."
	@EMBED_MODEL=$$(grep 'embedding_model:' config.yaml | awk '{print $$2}' | tr -d '"'); \
	echo "Pulling embedding model: $$EMBED_MODEL..."; \
	ollama pull $$EMBED_MODEL
	@echo "Embedding model ready"

pull-all-models: pull-model pull-embedding-model
	@echo "All models ready"

run:
	@go run main.go

build:
	@echo "Building binary..."
	@go build -o llm-app main.go
	@echo "Built: ./llm-app"

clean:
	@echo "Removing built files..."
	@rm -f llm-app
	@echo "Clean complete (data preserved)"

clean-data:
	@echo "⚠️  WARNING: This will delete your vector database and chat history!"
	@echo "Press Ctrl+C to cancel, or wait 5 seconds to continue..."
	@sleep 5
	@rm -rf data/
	@echo "Data deleted"

clean-all: clean clean-data
	@echo "Everything cleaned"

test:
	@go test ./...
