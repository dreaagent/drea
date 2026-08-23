package agent

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/dreaagent/drea/internal/config"
	"github.com/dreaagent/drea/internal/llm"
	"github.com/dreaagent/drea/internal/tool"
	"github.com/dreaagent/drea/internal/ui"
)

// TestTextRecoveredToolCallIsRecorded verifies that when a model prints a tool
// call as JSON text (instead of using the native tool_calls channel), the
// recovered tool call is recorded in the assistant message, not just in the
// local res variable. This is a regression test for a transcript-accuracy bug
// where the assistant turn had empty Content and nil ToolCalls even though a
// tool was executed.
func TestTextRecoveredToolCallIsRecorded(t *testing.T) {
	root := t.TempDir()

	var turn int
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		turn++
		if turn == 1 {
			fmt.Fprint(w, sse(
				`{"choices":[{"delta":{"content":"[{\"name\":\"write_file\",\"arguments\":{\"path\":\"hello.txt\",\"content\":\"hi\"}}]"}}]}`,
				`{"choices":[{"delta":{},"finish_reason":"stop"}]}`,
			))
			return
		}
		fmt.Fprint(w, sse(
			`{"choices":[{"delta":{"content":"Done."}}]}`,
			`{"choices":[{"delta":{},"finish_reason":"stop"}]}`,
		))
	}))
	defer srv.Close()

	cfg := config.Defaults(config.Saved{})
	cfg.Workdir = root
	cfg.AutoApprove = true
	cfg.Persist = false
	if err := cfg.Normalize(); err != nil {
		t.Fatal(err)
	}
	client := llm.NewClient(srv.URL, "", "m", 0, 5*time.Second, false, "")
	ag := New(&cfg, client, tool.NewRegistry(root), ui.New())

	if err := ag.Run(context.Background(), "make hello.txt"); err != nil {
		t.Fatal(err)
	}

	data, err := os.ReadFile(filepath.Join(root, "hello.txt"))
	if err != nil || string(data) != "hi" {
		t.Fatalf("file content = %q err %v", data, err)
	}

	// Find the assistant turn that triggered the write_file tool.
	var found bool
	for _, m := range ag.messages {
		if m.Role != llm.RoleAssistant {
			continue
		}
		for _, tc := range m.ToolCalls {
			if tc.Function.Name == "write_file" {
				found = true
				if m.Content != "" {
					t.Fatalf("recorded assistant content should be empty after text recovery, got %q", m.Content)
				}
			}
		}
	}
	if !found {
		t.Fatalf("write_file tool call not recorded in any assistant message; transcript: %+v", ag.messages)
	}
}
