package agent

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/dreaagent/drea/internal/config"
	"github.com/dreaagent/drea/internal/llm"
	"github.com/dreaagent/drea/internal/tool"
	"github.com/dreaagent/drea/internal/ui"
)

// TestJSONModePlainProseIsDisplayed verifies that a JSON-mode endpoint which
// returns ordinary prose (not a tool call or reply pseudo-tool) is shown to the
// user. This is a regression test for the "AI answer becomes [ ]" bug: the
// transcript recorded the content, but the UI path that printed it was removed.
func TestJSONModePlainProseIsDisplayed(t *testing.T) {
	root := t.TempDir()

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprint(w, sse(
			`{"choices":[{"delta":{"content":"I am done."}}]}`,
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

	if err := ag.Run(context.Background(), "say you are done"); err != nil {
		t.Fatal(err)
	}

	last := ag.messages[len(ag.messages)-1]
	if last.Role != llm.RoleAssistant || !strings.Contains(last.Content, "I am done.") {
		t.Fatalf("expected assistant prose in transcript, got %+v", last)
	}
}
