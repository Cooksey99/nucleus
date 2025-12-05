#!/bin/bash

# Nucleus Build and Run Script with Dependencies
# Usage: ./build_and_run.sh [options]

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
BUILD_MODE="debug"
RUN_EXAMPLE=""
FEATURES=""
CLEAN=false
TEST=false
START_DEPS=false
STOP_DEPS=false

# Function to display usage
usage() {
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  -r, --release         Build in release mode"
    echo "  -e, --example NAME    Run specific example (read_file_line, write_file, rag_indexing, refactor_assistant, detection)"
    echo "  -f, --features FEAT   Enable specific features (std, dev, full)"
    echo "  -c, --clean           Clean build artifacts before building"
    echo "  -t, --test            Run tests"
    echo "  -d, --deps            Start dependencies (Ollama + Qdrant)"
    echo "  -s, --stop            Stop dependencies (Ollama + Qdrant)"
    echo "  -h, --help            Display this help message"
    echo ""
    echo "Examples:"
    echo "  $0 --deps                           # Start Ollama and Qdrant"
    echo "  $0 --deps --example read_file_line  # Start deps and run example"
    echo "  $0 --release                        # Build in release mode"
    echo "  $0 --stop                           # Stop all dependencies"
    echo "  $0 --deps --test                    # Start deps and run tests"
    exit 1
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -r|--release)
            BUILD_MODE="release"
            shift
            ;;
        -e|--example)
            RUN_EXAMPLE="$2"
            shift 2
            ;;
        -f|--features)
            FEATURES="$2"
            shift 2
            ;;
        -c|--clean)
            CLEAN=true
            shift
            ;;
        -t|--test)
            TEST=true
            shift
            ;;
        -d|--deps)
            START_DEPS=true
            shift
            ;;
        -s|--stop)
            STOP_DEPS=true
            shift
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            usage
            ;;
    esac
done

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check if Ollama is running
is_ollama_running() {
    curl -s http://localhost:11434/api/tags >/dev/null 2>&1
}

# Function to check if Qdrant is running
is_qdrant_running() {
    curl -s http://localhost:6333/collections >/dev/null 2>&1
}

# Function to start Ollama
start_ollama() {
    if is_ollama_running; then
        echo -e "${GREEN}Ollama is already running${NC}"
    else
        echo -e "${YELLOW}Starting Ollama...${NC}"
        if command_exists ollama; then
            # Start Ollama in background
            nohup ollama serve > /tmp/ollama.log 2>&1 &
            echo $! > /tmp/ollama.pid
            
            # Wait for Ollama to be ready
            echo -e "${BLUE}Waiting for Ollama to start...${NC}"
            for i in {1..30}; do
                if is_ollama_running; then
                    echo -e "${GREEN}Ollama started successfully!${NC}"
                    return 0
                fi
                sleep 1
            done
            echo -e "${RED}Failed to start Ollama${NC}"
            return 1
        else
            echo -e "${RED}Ollama not found. Please install from https://ollama.ai${NC}"
            return 1
        fi
    fi
}

# Function to start Qdrant
start_qdrant() {
    if is_qdrant_running; then
        echo -e "${GREEN}Qdrant is already running${NC}"
    else
        echo -e "${YELLOW}Starting Qdrant...${NC}"
        
        # Try to find Qdrant executable
        QDRANT_CMD=""
        if command_exists qdrant; then
            QDRANT_CMD="qdrant"
        elif [ -f "C:\Users\frigont\Desktop\qdrant.exe" ]; then
            QDRANT_CMD="/c/Users/frigont/Desktop/qdrant.exe"
        elif [ -f "/c/Users/frigont/Desktop/qdrant.exe" ]; then
            QDRANT_CMD="/c/Users/frigont/Desktop/qdrant.exe"
        fi
        
        # Try native Qdrant first
        if [ -n "$QDRANT_CMD" ]; then
            # Start Qdrant natively in background
            nohup "$QDRANT_CMD" > /tmp/qdrant.log 2>&1 &
            echo $! > /tmp/qdrant.pid
            
            # Wait for Qdrant to be ready
            echo -e "${BLUE}Waiting for Qdrant to start...${NC}"
            for i in {1..30}; do
                if is_qdrant_running; then
                    echo -e "${GREEN}Qdrant started successfully!${NC}"
                    return 0
                fi
                sleep 1
            done
            echo -e "${RED}Failed to start Qdrant${NC}"
            return 1
        # Fall back to Docker if available
        elif command_exists docker && docker info >/dev/null 2>&1; then
            # Check if Qdrant container already exists
            if docker ps -a --format '{{.Names}}' 2>/dev/null | grep -q '^qdrant$'; then
                docker start qdrant >/dev/null 2>&1
            else
                docker run -d --name qdrant -p 6334:6334 -p 6333:6333 \
                    -v $(pwd)/qdrant_storage:/qdrant/storage \
                    qdrant/qdrant >/dev/null 2>&1
            fi
            
            # Wait for Qdrant to be ready
            echo -e "${BLUE}Waiting for Qdrant to start...${NC}"
            for i in {1..30}; do
                if is_qdrant_running; then
                    echo -e "${GREEN}Qdrant started successfully!${NC}"
                    return 0
                fi
                sleep 1
            done
            echo -e "${RED}Failed to start Qdrant${NC}"
            return 1
        else
            echo -e "${RED}Qdrant not found. Please install Qdrant:${NC}"
            echo -e "${YELLOW}  - Windows: Download from https://github.com/qdrant/qdrant/releases${NC}"
            echo -e "${YELLOW}  - Or use Docker: https://qdrant.tech/documentation/quick-start/${NC}"
            return 1
        fi
    fi
}

# Function to stop dependencies
stop_dependencies() {
    echo -e "${YELLOW}Stopping dependencies...${NC}"
    
    # Stop Ollama
    if [ -f /tmp/ollama.pid ]; then
        OLLAMA_PID=$(cat /tmp/ollama.pid)
        if ps -p $OLLAMA_PID > /dev/null 2>&1; then
            kill $OLLAMA_PID
            rm /tmp/ollama.pid
            echo -e "${GREEN}Ollama stopped${NC}"
        fi
    fi
    
    # Stop Qdrant (native process)
    if [ -f /tmp/qdrant.pid ]; then
        QDRANT_PID=$(cat /tmp/qdrant.pid)
        if ps -p $QDRANT_PID > /dev/null 2>&1; then
            kill $QDRANT_PID
            rm /tmp/qdrant.pid
            echo -e "${GREEN}Qdrant stopped${NC}"
        fi
    # Stop Qdrant (Docker container)
    elif command_exists docker && docker info >/dev/null 2>&1; then
        if docker ps --format '{{.Names}}' 2>/dev/null | grep -q '^qdrant$'; then
            docker stop qdrant >/dev/null 2>&1
            echo -e "${GREEN}Qdrant stopped${NC}"
        fi
    fi
    
    echo -e "${GREEN}Dependencies stopped${NC}"
}

# Stop dependencies if requested
if [ "$STOP_DEPS" = true ]; then
    stop_dependencies
    exit 0
fi

# Start dependencies if requested
if [ "$START_DEPS" = true ]; then
    echo -e "${BLUE}=== Starting Dependencies ===${NC}"
    start_ollama
    echo ""
    start_qdrant
    echo ""
fi

# Clean if requested
if [ "$CLEAN" = true ]; then
    echo -e "${YELLOW}Cleaning build artifacts...${NC}"
    cargo clean
    echo -e "${GREEN}Clean complete!${NC}"
    echo ""
fi

# Build command
BUILD_CMD="cargo build"
if [ "$BUILD_MODE" = "release" ]; then
    BUILD_CMD="$BUILD_CMD --release"
fi
if [ -n "$FEATURES" ]; then
    BUILD_CMD="$BUILD_CMD --features $FEATURES"
fi

# Run tests if requested
if [ "$TEST" = true ]; then
    echo -e "${YELLOW}Running tests...${NC}"
    if [ "$BUILD_MODE" = "release" ]; then
        cargo test --release
    else
        cargo test
    fi
    echo -e "${GREEN}Tests complete!${NC}"
    echo ""
fi

# Build the library
echo -e "${YELLOW}Building nucleus ($BUILD_MODE mode)...${NC}"
$BUILD_CMD
echo -e "${GREEN}Build complete!${NC}"
echo ""

# Run example if specified
if [ -n "$RUN_EXAMPLE" ]; then
    echo -e "${YELLOW}Running example: $RUN_EXAMPLE${NC}"
    RUN_CMD="cargo run --example $RUN_EXAMPLE"
    if [ "$BUILD_MODE" = "release" ]; then
        RUN_CMD="$RUN_CMD --release"
    fi
    if [ -n "$FEATURES" ]; then
        RUN_CMD="$RUN_CMD --features $FEATURES"
    fi
    echo ""
    $RUN_CMD
else
    echo -e "${GREEN}Build successful! Use --example to run an example.${NC}"
    echo ""
    echo "Available examples:"
    echo "  - read_file_line"
    echo "  - write_file"
    echo "  - rag_indexing"
    echo "  - refactor_assistant"
    echo "  - detection"
    echo ""
    if [ "$START_DEPS" = true ]; then
        echo -e "${BLUE}Dependencies are running:${NC}"
        echo "  - Ollama: http://localhost:11434"
        echo "  - Qdrant: http://localhost:6334"
        echo ""
        echo -e "${YELLOW}Use --stop to stop dependencies when done${NC}"
    fi
fi
