package agent

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/dreaagent/drea/internal/config"
	"github.com/dreaagent/drea/internal/llm"
	"github.com/dreaagent/drea/internal/tool"
	"github.com/dreaagent/drea/internal/ui"
)

// TestJSONModeReplyPseudoToolIsRecorded verifies that a JSON-mode endpoint
// which emits reply pseudo-tools has the concatenated reply recorded in the
// assistant message, not an empty content field. This is a regression test for
// the "empty assistant answer [ ]" bug.
func TestJSONModeReplyPseudoToolIsRecorded(t *testing.T) {
	root := t.TempDir()

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprint(w, sse(
			`{"choices":[{"delta":{"content":"[{\"name\":\"reply\",\"arguments\":{\"message\":\"Hello\"}},{\"name\":\"reply\",\"arguments\":{\"message\":\"world\"}}]"}}]}`,
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
	client := llm.NewClient(srv.URL, "", "m", 0, 5*time.Second, true, "")
	ag := New(&cfg, client, tool.NewRegistry(root), ui.New())

	if err := ag.Run(context.Background(), "say hello world"); err != nil {
		t.Fatal(err)
	}

	last := ag.messages[len(ag.messages)-1]
	if last.Role != llm.RoleAssistant {
		t.Fatalf("expected assistant message, got %+v", last)
	}
	want := "Hello\nworld"
	if last.Content != want {
		t.Fatalf("expected assistant content %q, got %q (message: %+v)", want, last.Content, last)
	}
	if len(last.ToolCalls) != 0 {
		t.Fatalf("expected no tool calls, got %+v", last.ToolCalls)
	}
}

// TestJSONModeReplyMixedWithRealTools verifies that a JSON-mode reply mixed
// with real tools is dropped from the current turn so the real tools execute.
func TestJSONModeReplyMixedWithRealTools(t *testing.T) {
	root := t.TempDir()

	var turn int
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		turn++
		if turn == 1 {
			fmt.Fprint(w, sse(
				`{"choices":[{"delta":{"content":"[{\"name\":\"reply\",\"arguments\":{\"message\":\"I will write the file\"}},{\"name\":\"write_file\",\"arguments\":{\"path\":\"hello.txt\",\"content\":\"hi\"}}]"}}]}`,
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
	client := llm.NewClient(srv.URL, "", "m", 0, 5*time.Second, true, "")
	ag := New(&cfg, client, tool.NewRegistry(root), ui.New())

	if err := ag.Run(context.Background(), "make hello.txt"); err != nil {
		t.Fatal(err)
	}

	data, err := os.ReadFile(filepath.Join(root, "hello.txt"))
	if err != nil || string(data) != "hi" {
		t.Fatalf("file content = %q err %v", data, err)
	}

	last := ag.messages[len(ag.messages)-1]
	if last.Role != llm.RoleAssistant || !strings.Contains(last.Content, "Done.") {
		t.Fatalf("expected final assistant prose, got %+v", last)
	}

	// The turn that mixed reply with write_file should have the reply dropped
	// from the recorded assistant message, so the transcript matches what was
	// actually dispatched. Also ensure the remaining tool call is not duplicated
	// due to slice aliasing between res.ToolCalls and the recorded message.
	for _, m := range ag.messages {
		if m.Role != llm.RoleAssistant {
			continue
		}
		var writeCount int
		for _, tc := range m.ToolCalls {
			switch tc.Function.Name {
			case "reply":
				t.Fatalf("reply pseudo-tool should not be recorded in assistant message when mixed with real tools; message: %+v", m)
			case "write_file":
				writeCount++
			}
		}
		if writeCount > 1 {
			t.Fatalf("write_file tool call was duplicated in recorded assistant message; message: %+v", m)
		}
	}
}

// TestJSONModeSecondTurnAfterFinalReply verifies that after a final JSON-mode
// prose reply (the turn that prints usage), the next user message still gets a
// meaningful response. This is a regression test for the "after the final reply
// every subsequent turn returns []" bug.
func TestJSONModeSecondTurnAfterFinalReply(t *testing.T) {
	root := t.TempDir()

	var turn int
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		turn++
		switch turn {
		case 1:
			fmt.Fprint(w, sse(
				`{"choices":[{"delta":{"content":"[{\"name\":\"reply\",\"arguments\":{\"message\":\"Task complete.\"}}]"}}]}`,
				`{"choices":[{"delta":{},"finish_reason":"stop"}]}`,
			))
		case 2:
			fmt.Fprint(w, sse(
				`{"choices":[{"delta":{"content":"[{\"name\":\"reply\",\"arguments\":{\"message\":\"Sure, here is more.\"}}]"}}]}`,
				`{"choices":[{"delta":{},"finish_reason":"stop"}]}`,
			))
		default:
			t.Errorf("unexpected turn %d", turn)
		}
	}))
	defer srv.Close()

	cfg := config.Defaults(config.Saved{})
	cfg.Workdir = root
	cfg.AutoApprove = true
	cfg.Persist = false
	if err := cfg.Normalize(); err != nil {
		t.Fatal(err)
	}
	client := llm.NewClient(srv.URL, "", "m", 0, 5*time.Second, true, "")
	ag := New(&cfg, client, tool.NewRegistry(root), ui.New())

	if err := ag.Run(context.Background(), "do something"); err != nil {
		t.Fatal(err)
	}
	if err := ag.Run(context.Background(), "do something else"); err != nil {
		t.Fatal(err)
	}

	if turn != 2 {
		t.Fatalf("expected 2 model turns, got %d", turn)
	}
	last := ag.messages[len(ag.messages)-1]
	if last.Role != llm.RoleAssistant || !strings.Contains(last.Content, "Sure, here is more.") {
		t.Fatalf("expected second-turn assistant prose, got %+v", last)
	}
}

// TestJSONModeEmptyReplyDoesNotStall verifies that a reply pseudo-tool with an
// empty message is not treated as a final reply. The agent should continue and
// ask the model for a real action instead of returning [] to the user.
func TestJSONModeEmptyReplyDoesNotStall(t *testing.T) {
	root := t.TempDir()

	var turn int
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		turn++
		switch turn {
		case 1:
			fmt.Fprint(w, sse(
				`{"choices":[{"delta":{"content":"[{\"name\":\"reply\",\"arguments\":{\"message\":\"\"}}]"}}]}`,
				`{"choices":[{"delta":{},"finish_reason":"stop"}]}`,
			))
		case 2:
			fmt.Fprint(w, sse(
				`{"choices":[{"delta":{"content":"[{\"name\":\"reply\",\"arguments\":{\"message\":\"Here is the real answer.\"}}]"}}]}`,
				`{"choices":[{"delta":{},"finish_reason":"stop"}]}`,
			))
		default:
			t.Errorf("unexpected turn %d", turn)
		}
	}))
	defer srv.Close()

	cfg := config.Defaults(config.Saved{})
	cfg.Workdir = root
	cfg.AutoApprove = true
	cfg.Persist = false
	if err := cfg.Normalize(); err != nil {
		t.Fatal(err)
	}
	client := llm.NewClient(srv.URL, "", "m", 0, 5*time.Second, true, "")
	ag := New(&cfg, client, tool.NewRegistry(root), ui.New())

	if err := ag.Run(context.Background(), "ask something"); err != nil {
		t.Fatal(err)
	}

	if turn != 2 {
		t.Fatalf("expected 2 model turns, got %d", turn)
	}
	last := ag.messages[len(ag.messages)-1]
	if last.Role != llm.RoleAssistant || !strings.Contains(last.Content, "Here is the real answer.") {
		t.Fatalf("expected non-empty final reply, got %+v", last)
	}
}

// TestJSONModeEmptyArrayOnSecondTurnIsNotSwallowed verifies that if the model
// emits a bare [] on a follow-up turn, the harness does not record an empty
// assistant message and stall. The schema now forbids empty arrays, but the
// client also treats them as ordinary prose so the user sees what happened.
func TestJSONModeEmptyArrayOnSecondTurnIsNotSwallowed(t *testing.T) {
	root := t.TempDir()

	var turn int
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		turn++
		switch turn {
		case 1:
			fmt.Fprint(w, sse(
				`{"choices":[{"delta":{"content":"[{\"name\":\"reply\",\"arguments\":{\"message\":\"Done.\"}}]"}}]}`,
				`{"choices":[{"delta":{},"finish_reason":"stop"}]}`,
			))
		case 2:
			fmt.Fprint(w, sse(
				`{"choices":[{"delta":{"content":"[]"}}]}`,
				`{"choices":[{"delta":{},"finish_reason":"stop"}]}`,
			))
		default:
			t.Errorf("unexpected turn %d", turn)
		}
	}))
	defer srv.Close()

	cfg := config.Defaults(config.Saved{})
	cfg.Workdir = root
	cfg.AutoApprove = true
	cfg.Persist = false
	if err := cfg.Normalize(); err != nil {
		t.Fatal(err)
	}
	client := llm.NewClient(srv.URL, "", "m", 0, 5*time.Second, true, "")
	ag := New(&cfg, client, tool.NewRegistry(root), ui.New())

	if err := ag.Run(context.Background(), "first"); err != nil {
		t.Fatal(err)
	}
	if err := ag.Run(context.Background(), "second"); err != nil {
		t.Fatal(err)
	}

	last := ag.messages[len(ag.messages)-1]
	if last.Role != llm.RoleAssistant || last.Content != "[]" {
		t.Fatalf("expected assistant message preserving the bare [] as prose, got %+v", last)
	}
}
