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

func TestTrimForModelBoundsAndKeepsEnds(t *testing.T) {
	big := strings.Repeat("A", maxToolResultChars) + "TAILMARKER"
	out := trimForModel(big)
	if len(out) > maxToolResultChars+128 {
		t.Fatalf("trimmed output too large: %d", len(out))
	}
	if !strings.HasPrefix(out, "AAAA") {
		t.Error("head not preserved")
	}
	if !strings.Contains(out, "TAILMARKER") {
		t.Error("tail not preserved")
	}
	if s := "short"; trimForModel(s) != s {
		t.Error("small input should be unchanged")
	}
}

func TestEstimateTokensCountsToolCalls(t *testing.T) {
	msgs := []llm.Message{
		{Role: llm.RoleUser, Content: strings.Repeat("x", 400)},
		{Role: llm.RoleAssistant, ToolCalls: []llm.ToolCall{
			{Function: llm.FunctionCall{Name: "run_command", Arguments: strings.Repeat("y", 400)}},
		}},
	}
	if got := estimateTokens(msgs); got < 190 {
		t.Fatalf("estimate too low: %d", got)
	}
}

// TestMaybeCompactPreservesStructure drives a compaction using a scripted
// endpoint that returns a summary, and asserts the system prompt and the recent
// tail survive while the middle is replaced by a single summary message.
func TestMaybeCompactPreservesStructure(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprint(w, sse(
			`{"choices":[{"delta":{"content":"SUMMARY-OF-OLD"}}]}`,
			`{"choices":[{"delta":{},"finish_reason":"stop"}]}`,
		))
	}))
	defer srv.Close()

	cfg := config.Defaults(config.Saved{})
	cfg.Persist = false
	cfg.ContextTokens = 1 // force compaction
	client := llm.NewClient(srv.URL, "", "m", 0, 5*time.Second, false, "")
	ag := New(&cfg, client, tool.NewRegistry(t.TempDir()), ui.New())

	ag.messages = []llm.Message{{Role: llm.RoleSystem, Content: "SYSTEM-PROMPT"}}
	for i := 0; i < 20; i++ {
		ag.messages = append(ag.messages, llm.Message{Role: llm.RoleUser, Content: fmt.Sprintf("msg %d", i)})
	}
	last := ag.messages[len(ag.messages)-1].Content

	ag.maybeCompact(context.Background())

	if ag.messages[0].Content != "SYSTEM-PROMPT" {
		t.Fatal("system prompt not preserved as first message")
	}
	if !strings.Contains(ag.messages[1].Content, "SUMMARY-OF-OLD") {
		t.Fatalf("second message should hold the summary, got %q", ag.messages[1].Content)
	}
	if ag.messages[len(ag.messages)-1].Content != last {
		t.Fatal("most recent message not preserved")
	}
	if len(ag.messages) >= 22 {
		t.Fatalf("history not compacted: still %d messages", len(ag.messages))
	}
}

func TestMaybeCompactNoOpUnderBudget(t *testing.T) {
	cfg := config.Defaults(config.Saved{})
	cfg.Persist = false
	cfg.ContextTokens = 100000
	ag := New(&cfg, nil, tool.NewRegistry(t.TempDir()), ui.New())
	ag.messages = []llm.Message{
		{Role: llm.RoleSystem, Content: "sys"},
		{Role: llm.RoleUser, Content: "small"},
	}
	before := len(ag.messages)
	ag.maybeCompact(context.Background()) // must not call the (nil) client
	if len(ag.messages) != before {
		t.Fatal("compaction ran while under budget")
	}
}

// TestMaybeCompactRunsWithCacheKey verifies that context compaction still
// happens when a prompt cache key is configured. A cache miss after compaction
// is expected; the point is that caching must work *before* compaction is
// needed.
func TestMaybeCompactRunsWithCacheKey(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprint(w, sse(
			`{"choices":[{"delta":{"content":"SUMMARY"}}]}`,
			`{"choices":[{"delta":{},"finish_reason":"stop"}]}`,
		))
	}))
	defer srv.Close()

	cfg := config.Defaults(config.Saved{})
	cfg.Persist = false
	cfg.ContextTokens = 1 // force compaction
	cfg.PromptCacheKey = "session-stable"
	client := llm.NewClient(srv.URL, "", "m", 0, 5*time.Second, false, "")
	ag := New(&cfg, client, tool.NewRegistry(t.TempDir()), ui.New())
	for i := 0; i < 20; i++ {
		ag.messages = append(ag.messages, llm.Message{Role: llm.RoleUser, Content: fmt.Sprintf("msg %d", i)})
	}
	before := len(ag.messages)
	ag.maybeCompact(context.Background())
	if len(ag.messages) >= before {
		t.Fatalf("compaction did not run with prompt cache key set; still %d messages", len(ag.messages))
	}
}
