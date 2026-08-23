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

func newTestAgent(t *testing.T, verify string) *Agent {
	t.Helper()
	cfg := config.Defaults(config.Saved{})
	cfg.Persist = false
	cfg.Workdir = t.TempDir()
	cfg.CommandTimeout = 10 * time.Second
	cfg.Verify = verify
	if err := cfg.Normalize(); err != nil {
		t.Fatal(err)
	}
	return New(&cfg, nil, tool.NewRegistry(cfg.Workdir), ui.New())
}

func TestVerifyPassIsNotRetried(t *testing.T) {
	ag := newTestAgent(t, "true")
	if fb, retry := ag.verify(context.Background()); retry || fb != "" {
		t.Fatalf("passing verify should not retry: fb=%q retry=%v", fb, retry)
	}
}

func TestVerifyFailureFeedsBack(t *testing.T) {
	ag := newTestAgent(t, "echo boom >&2; exit 3")
	fb, retry := ag.verify(context.Background())
	if !retry {
		t.Fatal("failing verify should request a retry")
	}
	if !strings.Contains(fb, "exit code 3") || !strings.Contains(fb, "boom") {
		t.Fatalf("feedback missing status/output: %q", fb)
	}
}

func TestVerifyDisabledWhenUnset(t *testing.T) {
	ag := newTestAgent(t, "")
	if _, retry := ag.verify(context.Background()); retry {
		t.Fatal("no verify command should never retry")
	}
}

func TestVerifyRoundsAreBounded(t *testing.T) {
	ag := newTestAgent(t, "false")
	rounds := 0
	for {
		_, retry := ag.verify(context.Background())
		if !retry {
			break
		}
		rounds++
		if rounds > config.DefaultVerifyAttempts+2 {
			t.Fatal("verify retried past the bound")
		}
	}
	if rounds != config.DefaultVerifyAttempts {
		t.Fatalf("expected %d retryable rounds, got %d", config.DefaultVerifyAttempts, rounds)
	}
}

// TestRunVerificationLoopSelfCorrects drives the whole loop: the model reports
// "done" each turn, the verify command fails the first time and passes the
// second, so the agent must run a second turn before returning.
func TestRunVerificationLoopSelfCorrects(t *testing.T) {
	var turns int
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		turns++
		fmt.Fprint(w, sse(
			`{"choices":[{"delta":{"content":"done"}}]}`,
			`{"choices":[{"delta":{},"finish_reason":"stop"}]}`,
		))
	}))
	defer srv.Close()

	// Fails the first invocation (no marker yet), passes afterwards.
	verify := `f="$PWD/.verified"; if [ -f "$f" ]; then exit 0; else : > "$f"; exit 1; fi`
	ag := newTestAgent(t, verify)
	ag.client = llm.NewClient(srv.URL, "", "m", 0, 5*time.Second, false, "")

	if err := ag.Run(context.Background(), "do it"); err != nil {
		t.Fatal(err)
	}
	if turns != 2 {
		t.Fatalf("expected 2 model turns (fail then pass), got %d", turns)
	}
}
