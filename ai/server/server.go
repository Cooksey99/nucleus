package server

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net"
	"os"
	"os/signal"
	"strings"
	"syscall"

	"llm-workspace/config"
	"llm-workspace/fileops"
	"llm-workspace/rag"

	"github.com/ollama/ollama/api"
)

const SocketPath = "/tmp/llm-workspace.sock"

type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type Request struct {
	Type    string     `json:"type"`
	Content string     `json:"content"`
	Pwd     *string    `json:"pwd,omitempty"`
	History *[]Message `json:"history,omitempty"`
}

type Response struct {
	Success bool   `json:"success"`
	Content string `json:"content"`
	Error   string `json:"error,omitempty"`
}

type StreamChunk struct {
	Type    string `json:"type"` // "chunk" or "done" or "error"
	Content string `json:"content"`
	Error   string `json:"error,omitempty"`
}

type Server struct {
	cfg         *config.Config
	client      *api.Client
	ragManager  *rag.Manager
	fileManager *fileops.Manager
	ctx         context.Context
}

func New() (*Server, error) {
	cfg, err := config.Load()
	if err != nil {
		return nil, fmt.Errorf("failed to load config: %w", err)
	}

	client, err := api.ClientFromEnvironment()
	if err != nil {
		return nil, fmt.Errorf("failed to create Ollama client: %w", err)
	}

	ragManager, err := rag.NewManager(cfg, client)
	if err != nil {
		return nil, fmt.Errorf("failed to initialize RAG: %w", err)
	}

	fileManager := fileops.NewManager(cfg, client, ragManager)

	return &Server{
		cfg:         cfg,
		client:      client,
		ragManager:  ragManager,
		fileManager: fileManager,
		ctx:         context.Background(),
	}, nil
}

func (s *Server) handleRequest(req Request) Response {
	switch req.Type {
	case "chat":
		response, err := s.fileManager.ChatWithTools(s.ctx, req.Content)
		if err != nil {
			return Response{Success: false, Error: err.Error()}
		}
		return Response{Success: true, Content: response}

	case "edit":
		response, err := s.fileManager.ChatWithTools(s.ctx, req.Content)
		if err != nil {
			return Response{Success: false, Error: err.Error()}
		}
		return Response{Success: true, Content: response}

	case "add":
		err := s.ragManager.AddKnowledge(s.ctx, req.Content, "user_input")
		if err != nil {
			return Response{Success: false, Error: err.Error()}
		}
		return Response{Success: true, Content: "Added to knowledge base"}

	case "index":
		err := s.ragManager.IndexDirectory(s.ctx, req.Content)
		if err != nil {
			return Response{Success: false, Error: err.Error()}
		}
		return Response{Success: true, Content: fmt.Sprintf("Indexed directory: %s", req.Content)}

	case "stats":
		count := s.ragManager.Count()
		return Response{
			Success: true,
			Content: fmt.Sprintf("Knowledge base contains %d documents", count),
		}

	default:
		return Response{Success: false, Error: "unknown request type"}
	}
}

func (s *Server) handleConnectionStreaming(conn net.Conn, req Request) {
	encoder := json.NewEncoder(conn)
	
	switch req.Type {
	case "chat", "edit":
		// Convert history to api.Message format
		var history []api.Message
		if req.History != nil {
			for _, msg := range *req.History {
				history = append(history, api.Message{
					Role:    msg.Role,
					Content: msg.Content,
				})
			}
		}
		
		// For chat/edit, stream the response in real-time
		response, err := s.fileManager.ChatWithToolsStream(s.ctx, req.Content, history, func(chunk string) {
			// Send each chunk as it arrives
			streamChunk := StreamChunk{Type: "chunk", Content: chunk}
			if err := encoder.Encode(streamChunk); err != nil {
				log.Printf("Failed to encode chunk: %v", err)
			}
		})
		
		if err != nil {
			chunk := StreamChunk{Type: "error", Error: err.Error()}
			encoder.Encode(chunk)
			return
		}
		
		// Send done signal with final response
		done := StreamChunk{Type: "done", Content: response}
		encoder.Encode(done)
		
	default:
		// For other requests, use normal response
		resp := s.handleRequest(req)
		chunk := StreamChunk{Type: "done", Content: resp.Content}
		if !resp.Success {
			chunk.Type = "error"
			chunk.Error = resp.Error
		}
		encoder.Encode(chunk)
	}
}

func (s *Server) handleConnection(conn net.Conn) {
	defer conn.Close()

	decoder := json.NewDecoder(conn)

	var req Request
	if err := decoder.Decode(&req); err != nil {
		log.Printf("Failed to decode request: %v", err)
		return
	}

	// Use streaming handler
	s.handleConnectionStreaming(conn, req)
}

func (s *Server) Start() error {
	if err := os.RemoveAll(SocketPath); err != nil {
		return fmt.Errorf("failed to remove old socket: %w", err)
	}

	listener, err := net.Listen("unix", SocketPath)
	if err != nil {
		return fmt.Errorf("failed to create socket: %w", err)
	}
	defer listener.Close()

	if err := os.Chmod(SocketPath, 0600); err != nil {
		return fmt.Errorf("failed to set socket permissions: %w", err)
	}

	log.Printf("AI Server listening on %s", SocketPath)
	log.Printf("Model: %s", s.cfg.LLM.Model)
	log.Printf("Knowledge Base: %d documents", s.ragManager.Count())

	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, os.Interrupt, syscall.SIGTERM)

	go func() {
		<-sigChan
		log.Println("Shutting down...")
		listener.Close()
		os.RemoveAll(SocketPath)
		os.Exit(0)
	}()

	for {
		conn, err := listener.Accept()
		if err != nil {
			if strings.Contains(err.Error(), "use of closed network connection") {
				break
			}
			log.Printf("Accept error: %v", err)
			continue
		}

		go s.handleConnection(conn)
	}

	return nil
}
