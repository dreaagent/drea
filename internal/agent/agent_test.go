package agent

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
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

// TestAgentLoopExecutesToolThenStops runs the full loop against a scripted SSE
// endpoint: turn 1 asks to write a file, turn 2 gives a final answer.
func TestAgentLoopExecutesToolThenStops(t *testing.T) {
	root := t.TempDir()

	var turn int
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		turn++
		if turn == 1 {
			fmt.Fprint(w, sse(
				`{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"c1","type":"function","function":{"name":"write_file","arguments":"{\"path\":\"hello.txt\",\"content\":\"hi\"}"}}]}}]}`,
				`{"choices":[{"delta":{},"finish_reason":"tool_calls"}]}`,
			))
			return
		}
		fmt.Fprint(w, sse(
			`{"choices":[{"delta":{"content":"Done, wrote hello.txt."}}]}`,
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
	if turn != 2 {
		t.Fatalf("expected 2 model turns, got %d", turn)
	}
	data, err := os.ReadFile(filepath.Join(root, "hello.txt"))
	if err != nil || string(data) != "hi" {
		t.Fatalf("file content = %q err %v", data, err)
	}
	// Conversation should contain the tool result message.
	var sawTool bool
	for _, m := range ag.messages {
		if m.Role == llm.RoleTool && strings.Contains(m.Content, "hello.txt") {
			sawTool = true
		}
	}
	if !sawTool {
		t.Error("tool result not recorded in conversation")
	}
}

// TestCommandPolicyBlocksDeniedCommand verifies a run_command matching the deny
// policy is refused without executing, and the refusal is fed back to the model.
// TestAgentReplyToolEndsTurn verifies that a native "reply" tool call is
// treated as assistant prose and ends the turn without being dispatched.
func TestAgentReplyToolEndsTurn(t *testing.T) {
	root := t.TempDir()

	var turn int
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		turn++
		if turn == 1 {
			fmt.Fprint(w, sse(
				`{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"c1","type":"function","function":{"name":"reply","arguments":"{\"message\":\"Hello from the reply tool.\"}"}}]}}]}`,
				`{"choices":[{"delta":{},"finish_reason":"tool_calls"}]}`,
			))
			return
		}
		t.Error("unexpected second turn")
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

	if err := ag.Run(context.Background(), "say hello"); err != nil {
		t.Fatal(err)
	}
	if turn != 1 {
		t.Fatalf("expected 1 model turn, got %d", turn)
	}
	last := ag.messages[len(ag.messages)-1]
	if last.Role != llm.RoleAssistant || last.Content != "Hello from the reply tool." {
		t.Fatalf("expected assistant reply message, got %+v", last)
	}
}

// TestMixedReplyAndToolsKeepsReply verifies that when a turn contains both a
// reply pseudo-tool and real tools, the reply is shown to the user and kept in
// the assistant message alongside the real tool calls.
func TestMixedReplyAndToolsKeepsReply(t *testing.T) {
	root := t.TempDir()

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprint(w, sse(
			`{"choices":[{"delta":{"content":"Starting work."}}]}`,
			`{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"c1","type":"function","function":{"name":"write_file","arguments":"{\"path\":\"x.txt\",\"content\":\"x\"}"}}]}}]}`,
			`{"choices":[{"delta":{"tool_calls":[{"index":1,"id":"c2","type":"function","function":{"name":"reply","arguments":"{\"message\":\"I will continue.\"}"}}]}}]}`,
			`{"choices":[{"delta":{},"finish_reason":"tool_calls"}}]}`,
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

	if err := ag.Run(context.Background(), "write x.txt and tell me you will continue"); err != nil {
		t.Fatal(err)
	}

	data, err := os.ReadFile(filepath.Join(root, "x.txt"))
	if err != nil || string(data) != "x" {
		t.Fatalf("file content = %q err %v", data, err)
	}

	var found bool
	for _, m := range ag.messages {
		if m.Role == llm.RoleAssistant && strings.Contains(m.Content, "Starting work.") && strings.Contains(m.Content, "I will continue.") && len(m.ToolCalls) == 1 && m.ToolCalls[0].Function.Name == "write_file" {
			found = true
		}
	}
	if !found {
		for _, m := range ag.messages {
			if m.Role == llm.RoleAssistant {
				t.Logf("assistant msg: content=%q tools=%v", m.Content, m.ToolCalls)
			}
		}
		t.Error("assistant message should keep both the reply content and the real tool call")
	}
}

// TestMixedReplyLastContinues verifies that when a reply pseudo-tool is the
// last call in a mixed turn, the harness classifies it after running the real
// tools and auto-continues when the model answers CONTINUE.
func TestMixedReplyLastContinues(t *testing.T) {
	root := t.TempDir()

	var turn int
	var sawClassifier bool
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		turn++

		body, _ := io.ReadAll(r.Body)
		isClassifier := strings.Contains(string(body), "You are classifying your latest reply")
		if isClassifier {
			sawClassifier = true
		}

		if turn == 1 && !isClassifier {
			fmt.Fprint(w, sse(
				`{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"c1","type":"function","function":{"name":"write_file","arguments":"{\"path\":\"x.txt\",\"content\":\"x\"}"}}]}}]}`,
				`{"choices":[{"delta":{"tool_calls":[{"index":1,"id":"c2","type":"function","function":{"name":"reply","arguments":"{\"message\":\"I wrote x.txt; I will now create y.txt.\"}"}}]}}]}`,
				`{"choices":[{"delta":{},"finish_reason":"tool_calls"}]}`,
			))
			return
		}
		if isClassifier {
			if strings.Contains(string(body), `"response_format"`) {
				fmt.Fprint(w, sse(
					`{"choices":[{"delta":{"content":"[{\"name\":\"classify_reply\",\"arguments\":{\"action\":\"CONTINUE_AUTONOMOUSLY\"}}]"}}]}`,
					`{"choices":[{"delta":{},"finish_reason":"stop"}}]}`,
				))
			} else {
				fmt.Fprint(w, sse(
					`{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"c-classify","type":"function","function":{"name":"classify_reply","arguments":"{\"action\":\"CONTINUE_AUTONOMOUSLY\"}"}}]}}]}`,
					`{"choices":[{"delta":{},"finish_reason":"tool_calls"}}]}`,
				))
			}
			return
		}
		// After CONTINUE_AUTONOMOUSLY, the agent loops again.
		fmt.Fprint(w, sse(
			`{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"c3","type":"function","function":{"name":"write_file","arguments":"{\"path\":\"y.txt\",\"content\":\"y\"}"}}]}}]}`,
			`{"choices":[{"delta":{"tool_calls":[{"index":1,"id":"c4","type":"function","function":{"name":"reply","arguments":"{\"message\":\"Done.\"}"}}]}}]}`,
			`{"choices":[{"delta":{},"finish_reason":"tool_calls"}]}`,
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
	ag.SetInteractive(true)

	if err := ag.Run(context.Background(), "write x.txt and y.txt"); err != nil {
		t.Fatal(err)
	}
	if !sawClassifier {
		t.Fatal("expected a classifier request after the mixed reply")
	}
	for _, name := range []string{"x.txt", "y.txt"} {
		if _, err := os.Stat(filepath.Join(root, name)); err != nil {
			t.Fatalf("%s was not created: %v", name, err)
		}
	}
}

// TestInteractiveReplyContinuation verifies that in interactive mode a pure
// reply turn is classified: if the model answers CONTINUE_AUTONOMOUSLY, the
// harness appends a system instruction and resumes the loop.
func TestInteractiveReplyContinuation(t *testing.T) {
	testInteractiveReplyContinuation(t, false)
}

// TestInteractiveReplyContinuationJSONMode is the JSON-mode regression: the
// classifier must also work when the endpoint forces JSON output, receiving a
// classify_reply tool call instead of plain prose.
func TestInteractiveReplyContinuationJSONMode(t *testing.T) {
	testInteractiveReplyContinuation(t, true)
}

func testInteractiveReplyContinuation(t *testing.T, jsonMode bool) {
	root := t.TempDir()

	var turn int
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		turn++

		body, _ := io.ReadAll(r.Body)
		isClassifier := strings.Contains(string(body), "You are classifying your latest reply")

		if turn == 1 {
			fmt.Fprint(w, sse(
				`{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"c1","type":"function","function":{"name":"reply","arguments":"{\"message\":\"Status update: still working.\"}"}}]}}]}`,
				`{"choices":[{"delta":{},"finish_reason":"tool_calls"}}]}`,
			))
			return
		}
		if turn == 2 && isClassifier {
			if strings.Contains(string(body), `"response_format"`) {
				fmt.Fprint(w, sse(
					`{"choices":[{"delta":{"content":"[{\"name\":\"classify_reply\",\"arguments\":{\"action\":\"CONTINUE_AUTONOMOUSLY\"}}]"}}]}`,
					`{"choices":[{"delta":{},"finish_reason":"stop"}}]}`,
				))
			} else {
				fmt.Fprint(w, sse(
					`{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"c-classify","type":"function","function":{"name":"classify_reply","arguments":"{\"action\":\"CONTINUE_AUTONOMOUSLY\"}"}}]}}]}`,
					`{"choices":[{"delta":{},"finish_reason":"tool_calls"}}]}`,
				))
			}
			return
		}
		if turn == 3 {
			fmt.Fprint(w, sse(
				`{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"c2","type":"function","function":{"name":"write_file","arguments":"{\"path\":\"done.txt\",\"content\":\"done\"}"}}]}}]}`,
				`{"choices":[{"delta":{},"finish_reason":"tool_calls"}}]}`,
			))
			return
		}
		if turn == 4 && !isClassifier {
			fmt.Fprint(w, sse(
				`{"choices":[{"delta":{"content":"Task complete."}}]}`,
				`{"choices":[{"delta":{},"finish_reason":"stop"}}]}`,
			))
			return
		}
		t.Errorf("unexpected turn %d (classifier=%v)", turn, isClassifier)
	}))
	defer srv.Close()

	cfg := config.Defaults(config.Saved{})
	cfg.Workdir = root
	cfg.AutoApprove = true
	cfg.Persist = false
	if err := cfg.Normalize(); err != nil {
		t.Fatal(err)
	}
	client := llm.NewClient(srv.URL, "", "m", 0, 5*time.Second, jsonMode, "")
	ag := New(&cfg, client, tool.NewRegistry(root), ui.New())
	ag.SetInteractive(true)

	if err := ag.Run(context.Background(), "do some work"); err != nil {
		t.Fatal(err)
	}
	if turn != 4 {
		t.Fatalf("expected 4 turns (reply, classify, continue, final), got %d", turn)
	}
	data, err := os.ReadFile(filepath.Join(root, "done.txt"))
	if err != nil || string(data) != "done" {
		t.Fatalf("continuation did not run write_file: content=%q err=%v", data, err)
	}
}

// TestRecordThoughtsFallback verifies that when the model streams reasoning but
// does not emit a record_thoughts tool call, the harness asks it to record its
// thoughts in a follow-up request and appends the result to the transcript.
func TestRecordThoughtsFallback(t *testing.T) {
	testRecordThoughtsFallback(t, false)
}

func TestRecordThoughtsFallbackJSONMode(t *testing.T) {
	testRecordThoughtsFallback(t, true)
}

func testRecordThoughtsFallback(t *testing.T, jsonMode bool) {
	root := t.TempDir()

	var turn int
	var sawFollowUp bool
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		turn++

		body, _ := io.ReadAll(r.Body)
		isFollowUp := strings.Contains(string(body), "You just produced reasoning")
		if isFollowUp {
			sawFollowUp = true
		}

		if turn == 1 && !isFollowUp {
			if jsonMode {
				fmt.Fprint(w, sse(
					`{"choices":[{"delta":{"reasoning_content":"I need to think about the best approach."}}]}`,
					`{"choices":[{"delta":{"content":"[{\"name\":\"reply\",\"arguments\":{\"message\":\"Status update: still working.\"}}]"}}]}`,
					`{"choices":[{"delta":{},"finish_reason":"stop"}]}`,
				))
			} else {
				fmt.Fprint(w, sse(
					`{"choices":[{"delta":{"reasoning_content":"I need to think about the best approach."}}]}`,
					`{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"c1","type":"function","function":{"name":"reply","arguments":"{\"message\":\"Status update: still working.\"}"}}]}}]}`,
					`{"choices":[{"delta":{},"finish_reason":"tool_calls"}]}`,
				))
			}
			return
		}
		if isFollowUp {
			if jsonMode {
				fmt.Fprint(w, sse(
					`{"choices":[{"delta":{"content":"[{\"name\":\"record_thoughts\",\"arguments\":{\"thoughts\":\"I should continue investigating.\"}}]"}}]}`,
					`{"choices":[{"delta":{},"finish_reason":"stop"}]}`,
				))
			} else {
				fmt.Fprint(w, sse(
					`{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"c2","type":"function","function":{"name":"record_thoughts","arguments":"{\"thoughts\":\"I should continue investigating.\"}"}}]}}]}`,
					`{"choices":[{"delta":{},"finish_reason":"tool_calls"}]}`,
				))
			}
			return
		}
		t.Errorf("unexpected turn %d (followUp=%v)", turn, isFollowUp)
	}))
	defer srv.Close()

	cfg := config.Defaults(config.Saved{})
	cfg.Workdir = root
	cfg.AutoApprove = true
	cfg.Persist = false
	if err := cfg.Normalize(); err != nil {
		t.Fatal(err)
	}
	client := llm.NewClient(srv.URL, "", "m", 0, 5*time.Second, jsonMode, "")
	ag := New(&cfg, client, tool.NewRegistry(root), ui.New())

	if err := ag.Run(context.Background(), "do some work"); err != nil {
		t.Fatal(err)
	}
	if !sawFollowUp {
		t.Fatal("expected a record_thoughts follow-up request")
	}
	var sawRecorded bool
	for _, m := range ag.messages {
		if m.Role == llm.RoleTool && m.Name == "record_thoughts" && strings.Contains(m.Content, "thoughts recorded") {
			sawRecorded = true
		}
	}
	if !sawRecorded {
		t.Error("record_thoughts tool result not found in transcript")
	}
}

func TestAutoApprovedCategories(t *testing.T) {
	cfg := config.Config{}
	if autoApproved(&cfg, "write_file") || autoApproved(&cfg, "edit_file") || autoApproved(&cfg, "apply_patch") {
		t.Error("write tools should require approval by default")
	}
	if autoApproved(&cfg, "git_init") || autoApproved(&cfg, "git_commit") || autoApproved(&cfg, "git_rollback") {
		t.Error("git tools should require approval by default")
	}
	if autoApproved(&cfg, "run_command") {
		t.Error("run_command should require approval by default")
	}

	cfg.AutoWrite = true
	if !autoApproved(&cfg, "write_file") || !autoApproved(&cfg, "edit_file") || !autoApproved(&cfg, "apply_patch") {
		t.Error("AutoWrite should approve write tools")
	}
	if autoApproved(&cfg, "run_command") {
		t.Error("AutoWrite should not approve run_command")
	}

	cfg = config.Config{AutoGit: true}
	if !autoApproved(&cfg, "git_commit") {
		t.Error("AutoGit should approve git_commit")
	}
	if autoApproved(&cfg, "write_file") {
		t.Error("AutoGit should not approve write_file")
	}

	cfg = config.Config{AutoRun: true}
	if !autoApproved(&cfg, "run_command") {
		t.Error("AutoRun should approve run_command")
	}
	if autoApproved(&cfg, "write_file") {
		t.Error("AutoRun should not approve write_file")
	}

	cfg = config.Config{AutoApprove: true}
	for _, name := range []string{"write_file", "edit_file", "apply_patch", "git_init", "git_commit", "git_rollback", "run_command"} {
		if !autoApproved(&cfg, name) {
			t.Errorf("AutoApprove master switch should approve %s", name)
		}
	}
}

// TestAutoApproveSeesConfigUpdates verifies that live changes to the config
// (e.g. from an interactive /auto command) are visible to the agent without
// recreating it.
func TestAutoApproveSeesConfigUpdates(t *testing.T) {
	cfg := config.Config{}
	ag := New(&cfg, nil, tool.NewRegistry(t.TempDir()), ui.New())
	if autoApproved(ag.cfg, "write_file") {
		t.Error("write_file should require approval initially")
	}
	cfg.AutoWrite = true
	if !autoApproved(ag.cfg, "write_file") {
		t.Error("agent should see live AutoWrite update")
	}
}

func TestCommandPolicyBlocksDeniedCommand(t *testing.T) {
	root := t.TempDir()
	marker := filepath.Join(root, "created.txt")

	var turn int
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		turn++
		if turn == 1 {
			// Ask to run a command the deny policy forbids.
			fmt.Fprint(w, sse(
				`{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"c1","type":"function","function":{"name":"run_command","arguments":"{\"command\":\"touch `+"created.txt"+`\"}"}}]}}]}`,
				`{"choices":[{"delta":{},"finish_reason":"tool_calls"}]}`,
			))
			return
		}
		fmt.Fprint(w, sse(
			`{"choices":[{"delta":{"content":"ok"}}]}`,
			`{"choices":[{"delta":{},"finish_reason":"stop"}]}`,
		))
	}))
	defer srv.Close()

	cfg := config.Defaults(config.Saved{})
	cfg.Workdir = root
	cfg.AutoApprove = true // even with auto-approve, deny must win
	cfg.Persist = false
	cfg.DenyCommands = []string{`\btouch\b`}
	if err := cfg.Normalize(); err != nil {
		t.Fatal(err)
	}
	client := llm.NewClient(srv.URL, "", "m", 0, 5*time.Second, false, "")
	ag := New(&cfg, client, tool.NewRegistry(root), ui.New())

	if err := ag.Run(context.Background(), "touch a file"); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(marker); !os.IsNotExist(err) {
		t.Fatalf("denied command was executed (marker exists): %v", err)
	}
	var sawBlock bool
	for _, m := range ag.messages {
		if m.Role == llm.RoleTool && strings.Contains(m.Content, "blocked by the configured command policy") {
			sawBlock = true
		}
	}
	if !sawBlock {
		t.Error("policy-deny feedback not recorded in conversation")
	}
}

func TestShortErrStripsControl(t *testing.T) {
	err := fmt.Errorf("endpoint returned 429: \x1b[31malert\x1b[0m\x07rate limited")
	got := shortErr(err)
	if strings.ContainsAny(got, "\x1b\x07") {
		t.Errorf("shortErr left control characters: %q", got)
	}
	if !strings.Contains(got, "alert") || !strings.Contains(got, "rate limited") {
		t.Errorf("shortErr dropped visible text: %q", got)
	}
}

func sse(frames ...string) string {
	var b strings.Builder
	for _, f := range frames {
		fmt.Fprintf(&b, "data: %s\n\n", f)
	}
	b.WriteString("data: [DONE]\n\n")
	return b.String()
}

func TestUsageStringIncludesCacheTokens(t *testing.T) {
	u := Usage{PromptTokens: 100, CompletionTokens: 10, Requests: 1, PromptCacheHitTokens: 90, PromptCacheMissTokens: 10}
	s := u.String()
	if !strings.Contains(s, "90 cached") || !strings.Contains(s, "10 uncached") {
		t.Errorf("usage string missing cache breakdown: %q", s)
	}
}

// TestPromptCachingMultiTurn drives a full multi-turn conversation with a
// scripted endpoint that reports a cache hit when the current request's
// `messages` array starts with a previously seen messages array. It verifies
// that:
//   - the first turn is a miss (new prefix);
//   - subsequent turns under the context budget hit the cached prefix;
//   - after context compaction rewrites older history, the next turn misses.
func TestPromptCachingMultiTurn(t *testing.T) {
	var prevTaskMessages [][]json.RawMessage
	var statuses []string
	turn := 0
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		var req struct {
			Messages []json.RawMessage `json:"messages"`
		}
		if err := json.Unmarshal(body, &req); err != nil {
			t.Errorf("turn %d: invalid request body: %v", turn+1, err)
		}

		// Compaction sends its own summary request; ignore it for status tracking.
		isTask := false
		for _, raw := range req.Messages {
			var m struct {
				Role    string `json:"role"`
				Content string `json:"content"`
			}
			_ = json.Unmarshal(raw, &m)
			if m.Role == "user" && strings.HasPrefix(m.Content, "task ") {
				isTask = true
				break
			}
		}

		status := "MISS"
		if isTask {
			for _, prev := range prevTaskMessages {
				if messagesStartWith(req.Messages, prev) {
					status = "HIT"
					break
				}
			}
			prevTaskMessages = append(prevTaskMessages, req.Messages)
			statuses = append(statuses, status)
		}
		turn++

		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("X-Cache-Status", status)
		fmt.Fprint(w, sse(
			fmt.Sprintf(`{"choices":[{"delta":{"content":"reply %d"},"finish_reason":"stop"}]}`, turn),
		))
	}))
	defer srv.Close()

	cfg := config.Defaults(config.Saved{})
	cfg.Workdir = t.TempDir()
	cfg.AutoApprove = true
	cfg.Persist = false
	cfg.ContextTokens = 100000 // large enough that compaction does not run
	cfg.PromptCacheKey = "test-session"
	if err := cfg.Normalize(); err != nil {
		t.Fatal(err)
	}

	client := llm.NewClient(srv.URL, "", "m", 0, 5*time.Second, false, "")
	client.SetPromptCacheKey("test-session")
	ag := New(&cfg, client, tool.NewRegistry(cfg.Workdir), ui.New())
	ctx := context.Background()

	// Phase 1: stable prefix across several turns. Six turns create enough
	// history that the next (seventh) turn will trigger compaction.
	for i := 0; i < 6; i++ {
		if err := ag.Run(ctx, fmt.Sprintf("task %d", i)); err != nil {
			t.Fatalf("turn %d: %v", i+1, err)
		}
	}

	if len(statuses) < 6 {
		t.Fatalf("expected 6 task turns, got %d", len(statuses))
	}
	if statuses[0] != "MISS" {
		t.Errorf("turn 1 cache status = %q, want MISS", statuses[0])
	}
	for i := 1; i < 6; i++ {
		if statuses[i] != "HIT" {
			t.Errorf("turn %d cache status = %q, want HIT (messages prefix should match turn %d)", i+1, statuses[i], i)
		}
	}

	// Phase 2: force compaction and verify the rewritten prefix causes a miss.
	ag.cfg.ContextTokens = 1
	if err := ag.Run(ctx, "task after compaction"); err != nil {
		t.Fatalf("post-compaction turn: %v", err)
	}
	if len(statuses) < 7 {
		t.Fatalf("expected a 7th task turn, got %d", len(statuses))
	}
	if statuses[6] != "MISS" {
		t.Errorf("post-compaction cache status = %q, want MISS (messages prefix was rewritten)", statuses[6])
	}
}

// messagesStartWith reports whether a begins with the same sequence of messages
// as prefix. This mirrors how a provider matches a cached prompt prefix.
func messagesStartWith(a, prefix []json.RawMessage) bool {
	if len(a) < len(prefix) {
		return false
	}
	for i := range prefix {
		if string(a[i]) != string(prefix[i]) {
			return false
		}
	}
	return true
}
