package main

import (
	"bufio"
	"context"
	"fmt"
	"log"
	"os"
	"strings"

	"llm-workspace/config"
	"llm-workspace/fileops"
	"llm-workspace/rag"

	"github.com/ollama/ollama/api"
)

func main() {
	cfg, err := config.Load()
	if err != nil {
		log.Fatalf("Failed to load config: %v", err)
	}

	client, err := api.ClientFromEnvironment()
	if err != nil {
		log.Fatalf("Failed to create Ollama client: %v", err)
	}

	ragManager, err := rag.NewManager(cfg, client)
	if err != nil {
		log.Fatalf("Failed to initialize RAG: %v", err)
	}

	fileManager := fileops.NewManager(cfg, client, ragManager)

	fmt.Println("Local LLM with RAG Ready!")
	fmt.Printf("Model: %s\n", cfg.LLM.Model)
	fmt.Printf("Knowledge Base: %d documents\n", ragManager.Count())
	fmt.Println("\nCommands:")
	fmt.Println("  /add <text>       - Add knowledge to vector DB")
	fmt.Println("  /index <path>     - Index a directory (code files)")
	fmt.Println("  /edit <request>   - Enable file editing mode")
	fmt.Println("  /stats            - Show knowledge base stats")
	fmt.Println("  /quit             - Exit")
	fmt.Println("\nType your message:")

	scanner := bufio.NewScanner(os.Stdin)
	ctx := context.Background()

	for {
		fmt.Print("\n> ")
		if !scanner.Scan() {
			break
		}

		input := strings.TrimSpace(scanner.Text())
		if input == "" {
			continue
		}

		if input == "/quit" {
			fmt.Println("Goodbye!")
			break
		}

		if input == "/stats" {
			fmt.Printf("Knowledge base contains %d documents\n", ragManager.Count())
			continue
		}

		if strings.HasPrefix(input, "/add ") {
			content := strings.TrimPrefix(input, "/add ")
			err := ragManager.AddKnowledge(ctx, content, "user_input")
			if err != nil {
				fmt.Printf("Error adding knowledge: %v\n", err)
			} else {
				fmt.Println("Added to knowledge base")
			}
			continue
		}

		if strings.HasPrefix(input, "/index ") {
			dirPath := strings.TrimPrefix(input, "/index ")
			fmt.Printf("Indexing directory: %s\n", dirPath)
			err := ragManager.IndexDirectory(ctx, dirPath)
			if err != nil {
				fmt.Printf("Error indexing: %v\n", err)
			}
			continue
		}

		if strings.HasPrefix(input, "/edit ") {
			request := strings.TrimPrefix(input, "/edit ")
			fmt.Println("🔧 File editing mode enabled")
			response, err := fileManager.ChatWithTools(ctx, request)
			if err != nil {
				fmt.Printf("Error: %v\n", err)
			} else if response != "" {
				fmt.Printf("\n%s\n", response)
			}
			continue
		}

		response, err := fileManager.Chat(ctx, input)
		if err != nil {
			fmt.Printf("Error: %v\n", err)
			continue
		}

		fmt.Printf("\n%s\n", response)
	}
}
