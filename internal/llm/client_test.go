package llm

import (
	"context"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

// responseWithBody wraps a string body in a minimal *http.Response for tests
// that call consume directly.
func responseWithBody(body string) *http.Response {
	return &http.Response{
		StatusCode: 200,
		Header:     make(http.Header),
		Body:       io.NopCloser(strings.NewReader(body)),
	}
}

// TestConsumeAccumulatesToolCalls verifies that content and index-addressed
// tool-call fragments split across many SSE frames are merged correctly.
func TestConsumeAccumulatesToolCalls(t *testing.T) {
	sse := strings.Join([]string{
		`data: {"choices":[{"delta":{"content":"Hel"}}]}`,
		`data: {"choices":[{"delta":{"content":"lo"}}]}`,
		`data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"read_","arguments":"{\"pa"}}]}}]}`,
		`data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"name":"file","arguments":"th\":\"x\"}"}}]}}]}`,
		`data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}]}`,
		`data: {"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}}`,
		"data: [DONE]",
		"",
	}, "\n\n")

	c := &Client{}
	var streamed strings.Builder
	res, err := c.consume(responseWithBody(sse), Handlers{OnContent: func(s string) { streamed.WriteString(s) }}, false)
	if err != nil {
		t.Fatal(err)
	}
	if res.Content != "Hello" {
		t.Errorf("content = %q, want %q", res.Content, "Hello")
	}
	if streamed.String() != "Hello" {
		t.Errorf("streamed = %q, want %q", streamed.String(), "Hello")
	}
	if len(res.ToolCalls) != 1 {
		t.Fatalf("got %d tool calls, want 1", len(res.ToolCalls))
	}
	tc := res.ToolCalls[0]
	if tc.ID != "call_1" || tc.Function.Name != "read_file" {
		t.Errorf("tool call = %+v", tc)
	}
	if tc.Function.Arguments != `{"path":"x"}` {
		t.Errorf("arguments = %q", tc.Function.Arguments)
	}
	if res.FinishReason != "tool_calls" {
		t.Errorf("finish = %q", res.FinishReason)
	}
	if res.Usage.TotalTokens != 15 {
		t.Errorf("usage = %+v", res.Usage)
	}
}

// TestConsumeStreamsReasoning verifies that reasoning_content deltas are
// accumulated on the result and surfaced through the OnReasoning handler, so
// a model's chain-of-thought can be shown live.
func TestConsumeStreamsReasoning(t *testing.T) {
	sse := strings.Join([]string{
		`data: {"choices":[{"delta":{"reasoning_content":"First, I "}}]}`,
		`data: {"choices":[{"delta":{"reasoning_content":"need to read the file."}}]}`,
		`data: {"choices":[{"delta":{"content":"Done"},"finish_reason":"stop"}]}`,
		"data: [DONE]",
		"",
	}, "\n\n")

	c := &Client{}
	var streamed strings.Builder
	res, err := c.consume(responseWithBody(sse), Handlers{OnReasoning: func(s string) { streamed.WriteString(s) }}, false)
	if err != nil {
		t.Fatal(err)
	}
	if res.ReasoningContent != "First, I need to read the file." {
		t.Errorf("reasoning = %q", res.ReasoningContent)
	}
	if streamed.String() != "First, I need to read the file." {
		t.Errorf("streamed reasoning = %q", streamed.String())
	}
	if res.Content != "Done" {
		t.Errorf("content = %q", res.Content)
	}
}

// TestStreamHTTPError surfaces non-2xx responses with a helpful message.
func TestStreamHTTPError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "bad key", http.StatusUnauthorized)
	}))
	defer srv.Close()

	c := NewClient(srv.URL, "", "m", 0, 5*time.Second, false, "")
	_, err := c.Stream(context.Background(), []Message{{Role: RoleUser, Content: "hi"}}, nil, Handlers{})
	if err == nil || !strings.Contains(err.Error(), "401") {
		t.Fatalf("expected 401 error, got %v", err)
	}
}

const okStream = "data: {\"choices\":[{\"delta\":{\"content\":\"ok\"},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n"

// TestJSONModeSendsResponseFormat verifies that enabling JSON mode adds a
// strict response_format schema to the request body.
func TestJSONModeSendsResponseFormat(t *testing.T) {
	var got string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		got = string(b)
		w.Header().Set("Content-Type", "text/event-stream")
		io.WriteString(w, okStream)
	}))
	defer srv.Close()

	c := NewClient(srv.URL, "", "m", 0, 5*time.Second, true, "")
	if _, err := c.Stream(context.Background(), []Message{{Role: RoleUser, Content: "hi"}}, nil, Handlers{}); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(got, `"response_format"`) {
		t.Fatalf("request body missing response_format: %s", got)
	}
	if !strings.Contains(got, `"type":"json_schema"`) {
		t.Fatalf("request body missing json_schema type: %s", got)
	}
	if !strings.Contains(got, `"name":"tool_calls"`) {
		t.Fatalf("request body missing tool_calls schema: %s", got)
	}
}

// TestJSONFormatVariants verifies each supported response_format variant is
// sent correctly.
func TestJSONFormatVariants(t *testing.T) {
	cases := []struct {
		name     string
		format   string
		wantType string
		wantKey  string
	}{
		{"json_schema", "json_schema", `"type":"json_schema"`, `"json_schema"`},
		{"json_object", "json_object", `"type":"json_object"`, `"schema"`},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			var got string
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				b, _ := io.ReadAll(r.Body)
				got = string(b)
				w.Header().Set("Content-Type", "text/event-stream")
				io.WriteString(w, okStream)
			}))
			defer srv.Close()

			c := NewClient(srv.URL, "", "m", 0, 5*time.Second, true, tc.format)
			if _, err := c.Stream(context.Background(), []Message{{Role: RoleUser, Content: "hi"}}, nil, Handlers{}); err != nil {
				t.Fatal(err)
			}
			if !strings.Contains(got, `"response_format"`) {
				t.Fatalf("request body missing response_format: %s", got)
			}
			if !strings.Contains(got, tc.wantType) {
				t.Fatalf("request body missing %s: %s", tc.wantType, got)
			}
			if !strings.Contains(got, tc.wantKey) {
				t.Fatalf("request body missing %s: %s", tc.wantKey, got)
			}
		})
	}
}

// TestSchemaRejectedError surfaces a helpful hint when the endpoint rejects
// the strict JSON schema.
func TestSchemaRejectedError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, `{"error":{"message":"'strict' is not supported"}}`, http.StatusBadRequest)
	}))
	defer srv.Close()

	c := NewClient(srv.URL, "", "m", 0, 5*time.Second, true, "")
	_, err := c.Stream(context.Background(), []Message{{Role: RoleUser, Content: "hi"}}, nil, Handlers{})
	if err == nil {
		t.Fatal("expected error")
	}
	if !errors.Is(err, ErrSchemaRejected) {
		t.Fatalf("expected ErrSchemaRejected, got %v", err)
	}
	if !strings.Contains(err.Error(), "json_object") {
		t.Fatalf("error should hint at json_object: %v", err)
	}
}

// TestJSONModeContentAsToolCalls verifies that endpoints which stream JSON-mode
// tool calls as plain content deltas (without a finish_reason) are parsed into
// native tool calls.
func TestJSONModeContentAsToolCalls(t *testing.T) {
	sse := strings.Join([]string{
		`data: {"choices":[{"delta":{"content":"[{\"name\":\"list_dir\",\"arguments\":{\"path\":\".\"}}]"}}]}`,
		"data: [DONE]",
		"",
	}, "\n\n")

	c := &Client{jsonMode: true}
	res, err := c.consume(responseWithBody(sse), Handlers{}, true)
	if err != nil {
		t.Fatal(err)
	}
	if len(res.ToolCalls) != 1 {
		t.Fatalf("got %d tool calls, want 1", len(res.ToolCalls))
	}
	if res.ToolCalls[0].Function.Name != "list_dir" {
		t.Errorf("name = %q, want list_dir", res.ToolCalls[0].Function.Name)
	}
	if res.Content != "" {
		t.Errorf("content should be cleared in JSON mode after parsing tool calls, got %q", res.Content)
	}
}

// TestJSONModeReplyArray verifies that JSON-mode reply pseudo-tools are kept as
// tool calls (so the agent can recognise reply-only turns) while their prose is
// still streamed live.
func TestJSONModeReplyArray(t *testing.T) {
	sse := strings.Join([]string{
		`data: {"choices":[{"delta":{"content":"[{\"name\":\"reply\",\"arguments\":{\"message\":\"Hello\"}},{\"name\":\"reply\",\"arguments\":{\"message\":\"world\"}}]"}}]}`,
		`data: {"choices":[{"delta":{},"finish_reason":"stop"}]}`,
		"data: [DONE]",
		"",
	}, "\n\n")

	var streamed strings.Builder
	c := &Client{jsonMode: true}
	res, err := c.consume(responseWithBody(sse), Handlers{OnContent: func(s string) { streamed.WriteString(s) }}, true)
	if err != nil {
		t.Fatal(err)
	}
	if len(res.ToolCalls) != 2 {
		t.Fatalf("got %d tool calls, want 2", len(res.ToolCalls))
	}
	if !IsReplyCall(res.ToolCalls[0]) || !IsReplyCall(res.ToolCalls[1]) {
		t.Errorf("expected reply pseudo-tools, got %v", res.ToolCalls)
	}
	want := "Hello\nworld"
	if ReplyMessage(res.ToolCalls[0])+"\n"+ReplyMessage(res.ToolCalls[1]) != want {
		t.Errorf("reply messages = %q / %q, want %q", ReplyMessage(res.ToolCalls[0]), ReplyMessage(res.ToolCalls[1]), want)
	}
	if streamed.String() != want {
		t.Errorf("streamed = %q, want %q", streamed.String(), want)
	}
}

// TestJSONModeReplyArrayNotDeliveredTwice verifies that reply pseudo-tools are
// released through OnContent exactly once, not once during streaming and again
// at the end, while the tool call itself is preserved for the agent.
func TestJSONModeReplyArrayNotDeliveredTwice(t *testing.T) {
	sse := strings.Join([]string{
		`data: {"choices":[{"delta":{"content":"[{\"name\":\"reply\",\"arguments\":{\"message\":\"Hello\"}}]"}}]}`,
		`data: {"choices":[{"delta":{},"finish_reason":"stop"}]}`,
		"data: [DONE]",
		"",
	}, "\n\n")

	var count int
	c := &Client{jsonMode: true}
	res, err := c.consume(responseWithBody(sse), Handlers{OnContent: func(string) { count++ }}, true)
	if err != nil {
		t.Fatal(err)
	}
	if len(res.ToolCalls) != 1 || !IsReplyCall(res.ToolCalls[0]) {
		t.Errorf("expected one reply tool, got %v", res.ToolCalls)
	}
	if ReplyMessage(res.ToolCalls[0]) != "Hello" {
		t.Errorf("reply message = %q, want Hello", ReplyMessage(res.ToolCalls[0]))
	}
	if count != 1 {
		t.Errorf("OnContent called %d times, want 1", count)
	}
}

// TestJSONModeEmptyArrayIsNotToolCalls verifies that an empty JSON array is
// treated as ordinary prose (or at least not parsed as zero tool calls that
// would leave the assistant message empty). With minItems:1 in the schema the
// endpoint should reject this, but the client also defends against it.
func TestJSONModeEmptyArrayIsNotToolCalls(t *testing.T) {
	sse := strings.Join([]string{
		`data: {"choices":[{"delta":{"content":"[]"}}]}`,
		`data: {"choices":[{"delta":{},"finish_reason":"stop"}]}`,
		"data: [DONE]",
		"",
	}, "\n\n")

	var streamed strings.Builder
	c := &Client{jsonMode: true}
	res, err := c.consume(responseWithBody(sse), Handlers{OnContent: func(s string) { streamed.WriteString(s) }}, true)
	if err != nil {
		t.Fatal(err)
	}
	if len(res.ToolCalls) != 0 {
		t.Errorf("got %d tool calls, want 0", len(res.ToolCalls))
	}
	if res.Content != "[]" {
		t.Errorf("content = %q, want %q", res.Content, "[]")
	}
	if streamed.String() != "[]" {
		t.Errorf("streamed = %q, want %q", streamed.String(), "[]")
	}
}

// TestJSONModeEmptyReplyIsNotFinal verifies that a reply pseudo-tool with a
// blank message is preserved as a tool call with an empty message. The agent
// recognises it as a vacuous reply and nudges for a real action instead of
// treating it as a final assistant reply.
func TestJSONModeEmptyReplyIsNotFinal(t *testing.T) {
	sse := strings.Join([]string{
		`data: {"choices":[{"delta":{"content":"[{\"name\":\"reply\",\"arguments\":{\"message\":\"\"}}]"}}]}`,
		`data: {"choices":[{"delta":{},"finish_reason":"stop"}]}`,
		"data: [DONE]",
		"",
	}, "\n\n")

	var streamed strings.Builder
	c := &Client{jsonMode: true}
	res, err := c.consume(responseWithBody(sse), Handlers{OnContent: func(s string) { streamed.WriteString(s) }}, true)
	if err != nil {
		t.Fatal(err)
	}
	if len(res.ToolCalls) != 1 || !IsReplyCall(res.ToolCalls[0]) {
		t.Errorf("expected one reply tool, got %v", res.ToolCalls)
	}
	if ReplyMessage(res.ToolCalls[0]) != "" {
		t.Errorf("reply message = %q, want empty", ReplyMessage(res.ToolCalls[0]))
	}
	if streamed.String() != "" {
		t.Errorf("streamed = %q, want empty", streamed.String())
	}
}

// TestJSONModeProseFallback verifies that JSON-mode content which is not a
// valid tool-call serialization is treated as ordinary assistant prose.
func TestJSONModeProseFallback(t *testing.T) {
	sse := strings.Join([]string{
		`data: {"choices":[{"delta":{"content":"Hello"}}]}`,
		`data: {"choices":[{"delta":{"content":" world"}}]}`,
		`data: {"choices":[{"delta":{},"finish_reason":"stop"}]}`,
		"data: [DONE]",
		"",
	}, "\n\n")

	c := &Client{jsonMode: true}
	var streamed strings.Builder
	res, err := c.consume(responseWithBody(sse), Handlers{OnContent: func(s string) { streamed.WriteString(s) }}, true)
	if err != nil {
		t.Fatal(err)
	}
	if res.Content != "Hello world" {
		t.Errorf("content = %q, want %q", res.Content, "Hello world")
	}
	if streamed.String() != "Hello world" {
		t.Errorf("streamed = %q, want %q", streamed.String(), "Hello world")
	}
	if len(res.ToolCalls) != 0 {
		t.Errorf("got %d tool calls, want 0", len(res.ToolCalls))
	}
}

// TestJSONModeReplyStreamsIncrementally verifies that a JSON-mode reply
// pseudo-tool's message is delivered through OnContent as it arrives, not as
// a single block at the end, while the tool call is preserved for the agent.
func TestJSONModeReplyStreamsIncrementally(t *testing.T) {
	sse := strings.Join([]string{
		`data: {"choices":[{"delta":{"content":"[{"}}]}`,
		`data: {"choices":[{"delta":{"content":"\"name\":\"reply\",\"arguments\":{\"message\":\"Hello"}}]}`,
		`data: {"choices":[{"delta":{"content":" world\"}}]"}}]}`,
		`data: {"choices":[{"delta":{},"finish_reason":"stop"}]}`,
		"data: [DONE]",
		"",
	}, "\n\n")

	var chunks []string
	c := &Client{jsonMode: true}
	res, err := c.consume(responseWithBody(sse), Handlers{OnContent: func(s string) { chunks = append(chunks, s) }}, true)
	if err != nil {
		t.Fatal(err)
	}
	if len(res.ToolCalls) != 1 || !IsReplyCall(res.ToolCalls[0]) {
		t.Errorf("expected one reply tool, got %v", res.ToolCalls)
	}
	if ReplyMessage(res.ToolCalls[0]) != "Hello world" {
		t.Errorf("reply message = %q, want Hello world", ReplyMessage(res.ToolCalls[0]))
	}
	got := strings.Join(chunks, "")
	if got != "Hello world" {
		t.Errorf("streamed = %q, want %q", got, "Hello world")
	}
	if len(chunks) < 2 {
		t.Errorf("expected reply to be split across multiple OnContent calls, got %d", len(chunks))
	}
}

// TestJSONModeRealToolCallsAreNotStreamedAsProse verifies that JSON-mode tool
// call serializations are never passed to OnContent live, so the user does not
// see partial JSON scrolling by.
func TestJSONModeRealToolCallsAreNotStreamedAsProse(t *testing.T) {
	sse := strings.Join([]string{
		`data: {"choices":[{"delta":{"content":"[{"}}]}`,
		`data: {"choices":[{"delta":{"content":"\"name\":\"list_dir\",\"arguments\":{\"path\":\".\"}}]"}}]}`,
		`data: {"choices":[{"delta":{},"finish_reason":"stop"}]}`,
		"data: [DONE]",
		"",
	}, "\n\n")

	var streamed strings.Builder
	c := &Client{jsonMode: true}
	res, err := c.consume(responseWithBody(sse), Handlers{OnContent: func(s string) { streamed.WriteString(s) }}, true)
	if err != nil {
		t.Fatal(err)
	}
	if len(res.ToolCalls) != 1 {
		t.Fatalf("got %d tool calls, want 1", len(res.ToolCalls))
	}
	if res.ToolCalls[0].Function.Name != "list_dir" {
		t.Errorf("name = %q, want list_dir", res.ToolCalls[0].Function.Name)
	}
	if streamed.Len() != 0 {
		t.Errorf("tool-call JSON was streamed as prose: %q", streamed.String())
	}
	if res.Content != "" {
		t.Errorf("content should be cleared for tool-call JSON, got %q", res.Content)
	}
}

// TestNoJSONModeOmitsResponseFormat verifies the default request has no
// response_format field.
func TestNoJSONModeOmitsResponseFormat(t *testing.T) {
	var got string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		got = string(b)
		w.Header().Set("Content-Type", "text/event-stream")
		io.WriteString(w, okStream)
	}))
	defer srv.Close()

	c := NewClient(srv.URL, "", "m", 0, 5*time.Second, false, "")
	if _, err := c.Stream(context.Background(), []Message{{Role: RoleUser, Content: "hi"}}, nil, Handlers{}); err != nil {
		t.Fatal(err)
	}
	if strings.Contains(got, "response_format") {
		t.Fatalf("request body should not contain response_format: %s", got)
	}
}

// TestTopPForwarded verifies that top_p is always forwarded.
func TestTopPForwarded(t *testing.T) {
	var got string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		got = string(b)
		w.Header().Set("Content-Type", "text/event-stream")
		io.WriteString(w, okStream)
	}))
	defer srv.Close()

	c := NewClient(srv.URL, "", "m", 0, 5*time.Second, false, "")
	if _, err := c.Stream(context.Background(), []Message{{Role: RoleUser, Content: "hi"}}, nil, Handlers{}); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(got, `"top_p":0.95`) {
		t.Fatalf("request body missing default top_p: %s", got)
	}
}

// TestReasoningEffortForwarded verifies that a non-empty reasoning_effort is
// sent and an empty one is omitted.
func TestReasoningEffortForwarded(t *testing.T) {
	var got string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		got = string(b)
		w.Header().Set("Content-Type", "text/event-stream")
		io.WriteString(w, okStream)
	}))
	defer srv.Close()

	c := NewClientWithReasoning(srv.URL, "", "m", 0, 0.95, "medium", 5*time.Second, false, "")
	if _, err := c.Stream(context.Background(), []Message{{Role: RoleUser, Content: "hi"}}, nil, Handlers{}); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(got, `"reasoning_effort":"medium"`) {
		t.Fatalf("request body missing reasoning_effort: %s", got)
	}

	got = ""
	c.SetReasoningEffort("")
	if _, err := c.Stream(context.Background(), []Message{{Role: RoleUser, Content: "hi"}}, nil, Handlers{}); err != nil {
		t.Fatal(err)
	}
	if strings.Contains(got, "reasoning_effort") {
		t.Fatalf("request body should not contain reasoning_effort: %s", got)
	}
}

func testClient(url string) *Client {
	c := NewClient(url, "", "m", 0, 5*time.Second, false, "")
	c.baseDelay = time.Millisecond // keep backoff tiny in tests
	return c
}

// TestStreamRetriesThenSucceeds verifies a 429 is retried and the following
// successful stream is returned, with handlers fired only once.
func TestStreamRetriesThenSucceeds(t *testing.T) {
	var calls int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if atomic.AddInt32(&calls, 1) == 1 {
			http.Error(w, "<html>rate limited</html>", http.StatusTooManyRequests)
			return
		}
		w.Header().Set("Content-Type", "text/event-stream")
		io.WriteString(w, okStream)
	}))
	defer srv.Close()

	var retries, content int
	var retryErr string
	res, err := testClient(srv.URL).Stream(context.Background(),
		[]Message{{Role: RoleUser, Content: "hi"}}, nil, Handlers{
			OnContent: func(string) { content++ },
			OnRetry:   func(_ int, _ time.Duration, err error) { retries++; retryErr = err.Error() },
		})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if res.Content != "ok" {
		t.Errorf("content = %q, want %q", res.Content, "ok")
	}
	if got := atomic.LoadInt32(&calls); got != 2 {
		t.Errorf("server calls = %d, want 2", got)
	}
	if retries != 1 {
		t.Errorf("OnRetry called %d times, want 1", retries)
	}
	if content != 1 {
		t.Errorf("OnContent called %d times, want 1 (no double delivery)", content)
	}
	if !strings.Contains(retryErr, "Drea's connection to the AI API endpoint is temporarily overloaded") {
		t.Errorf("OnRetry error should be explanatory, got %q", retryErr)
	}
	if strings.Contains(retryErr, "<html>") || strings.Contains(retryErr, "endpoint returned") {
		t.Errorf("OnRetry error should not expose raw provider response, got %q", retryErr)
	}
}

// TestStreamHonorsRetryAfter checks the Retry-After header sets the delay.
func TestStreamHonorsRetryAfter(t *testing.T) {
	var calls int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if atomic.AddInt32(&calls, 1) == 1 {
			w.Header().Set("Retry-After", "1")
			http.Error(w, "slow down", http.StatusTooManyRequests)
			return
		}
		w.Header().Set("Content-Type", "text/event-stream")
		io.WriteString(w, okStream)
	}))
	defer srv.Close()

	var gotDelay time.Duration
	start := time.Now()
	_, err := testClient(srv.URL).Stream(context.Background(),
		[]Message{{Role: RoleUser, Content: "hi"}}, nil, Handlers{
			OnRetry: func(_ int, d time.Duration, _ error) { gotDelay = d },
		})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if gotDelay != time.Second {
		t.Errorf("retry delay = %s, want 1s", gotDelay)
	}
	if elapsed := time.Since(start); elapsed < time.Second {
		t.Errorf("waited %s, expected >= 1s from Retry-After", elapsed)
	}
}

// TestStreamRetryExhaustion returns a useful error after all attempts fail.
func TestStreamRetryExhaustion(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "still busy", http.StatusServiceUnavailable)
	}))
	defer srv.Close()

	c := testClient(srv.URL)
	c.maxRetries = 2
	_, err := c.Stream(context.Background(),
		[]Message{{Role: RoleUser, Content: "hi"}}, nil, Handlers{})
	if err == nil {
		t.Fatal("expected exhaustion error, got nil")
	}
	if !strings.Contains(err.Error(), "Drea's connection to the AI API endpoint is still unavailable") {
		t.Fatalf("expected explanatory exhaustion error, got %v", err)
	}
	if strings.Contains(err.Error(), "503") || strings.Contains(err.Error(), "still busy") {
		t.Errorf("exhaustion error should not expose raw status or body, got %v", err)
	}
}

// TestStreamNoRetryOn400 verifies non-transient statuses fail immediately.
func TestStreamNoRetryOn400(t *testing.T) {
	var calls int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt32(&calls, 1)
		http.Error(w, "bad request", http.StatusBadRequest)
	}))
	defer srv.Close()

	_, err := testClient(srv.URL).Stream(context.Background(),
		[]Message{{Role: RoleUser, Content: "hi"}}, nil, Handlers{})
	if err == nil || !strings.Contains(err.Error(), "400") {
		t.Fatalf("expected 400 error, got %v", err)
	}
	if strings.Contains(err.Error(), "giving up") {
		t.Errorf("non-retryable error should not be wrapped as exhaustion: %v", err)
	}
	if got := atomic.LoadInt32(&calls); got != 1 {
		t.Errorf("server calls = %d, want 1 (no retry on 400)", got)
	}
}

// TestStreamCancelDuringBackoff ensures a cancelled context aborts the wait.
func TestStreamCancelDuringBackoff(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Retry-After", "60")
		http.Error(w, "busy", http.StatusTooManyRequests)
	}))
	defer srv.Close()

	ctx, cancel := context.WithCancel(context.Background())
	go func() {
		time.Sleep(50 * time.Millisecond)
		cancel()
	}()

	start := time.Now()
	_, err := testClient(srv.URL).Stream(ctx,
		[]Message{{Role: RoleUser, Content: "hi"}}, nil, Handlers{})
	if err == nil {
		t.Fatal("expected context cancellation error")
	}
	if time.Since(start) > 5*time.Second {
		t.Errorf("backoff was not interrupted by cancellation")
	}
}

func TestParseRetryAfter(t *testing.T) {
	if d := parseRetryAfter("5"); d != 5*time.Second {
		t.Errorf("seconds: got %s", d)
	}
	if d := parseRetryAfter(""); d != 0 {
		t.Errorf("empty: got %s", d)
	}
	if d := parseRetryAfter("-3"); d != 0 {
		t.Errorf("negative: got %s", d)
	}
	if d := parseRetryAfter("garbage"); d != 0 {
		t.Errorf("garbage: got %s", d)
	}
	future := time.Now().Add(2 * time.Second).UTC().Format(http.TimeFormat)
	if d := parseRetryAfter(future); d <= 0 || d > 2*time.Second {
		t.Errorf("http-date: got %s", d)
	}
}

func TestRetriableStatus(t *testing.T) {
	for _, code := range []int{429, 500, 502, 503, 504} {
		if !retriableStatus(code) {
			t.Errorf("%d should be retriable", code)
		}
	}
	for _, code := range []int{400, 401, 403, 404, 200} {
		if retriableStatus(code) {
			t.Errorf("%d should not be retriable", code)
		}
	}
}

// TestPromptCacheKeyForwarded verifies that a configured prompt_cache_key is
// sent in the request body.
func TestPromptCacheKeyForwarded(t *testing.T) {
	var got string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		got = string(b)
		w.Header().Set("Content-Type", "text/event-stream")
		io.WriteString(w, okStream)
	}))
	defer srv.Close()

	c := NewClient(srv.URL, "", "m", 0, 5*time.Second, false, "")
	c.SetPromptCacheKey("session-42")
	if _, err := c.Stream(context.Background(), []Message{{Role: RoleUser, Content: "hi"}}, nil, Handlers{}); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(got, `"prompt_cache_key":"session-42"`) {
		t.Fatalf("request body missing prompt_cache_key: %s", got)
	}
}

// TestCacheStatusFromHeaderHit verifies cache hit detection from response headers.
func TestCacheStatusFromHeaderHit(t *testing.T) {
	sse := "data: {\"choices\":[{\"delta\":{\"content\":\"ok\"},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n"
	resp := responseWithBody(sse)
	resp.Header.Set("X-Cache-Status", "HIT")

	c := &Client{}
	res, err := c.consume(resp, Handlers{}, false)
	if err != nil {
		t.Fatal(err)
	}
	if res.CacheStatus != "hit" {
		t.Errorf("cache status = %q, want hit", res.CacheStatus)
	}
	if !strings.Contains(res.CacheDetail, "X-Cache-Status: HIT") {
		t.Errorf("cache detail = %q", res.CacheDetail)
	}
}

// TestCacheStatusFromHeaderMiss verifies cache miss detection from response headers.
func TestCacheStatusFromHeaderMiss(t *testing.T) {
	sse := "data: {\"choices\":[{\"delta\":{\"content\":\"ok\"},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n"
	resp := responseWithBody(sse)
	resp.Header.Set("X-Prompt-Cache-Hit", "false")

	c := &Client{}
	res, err := c.consume(resp, Handlers{}, false)
	if err != nil {
		t.Fatal(err)
	}
	if res.CacheStatus != "miss" {
		t.Errorf("cache status = %q, want miss", res.CacheStatus)
	}
}

// TestCacheStatusFromUsage verifies cache details are extracted from usage fields.
func TestCacheStatusFromUsage(t *testing.T) {
	sse := strings.Join([]string{
		`data: {"choices":[{"delta":{"content":"ok"},"finish_reason":"stop"}]}`,
		`data: {"usage":{"prompt_tokens":100,"completion_tokens":10,"total_tokens":110,"prompt_cache_hit_tokens":90,"prompt_cache_miss_tokens":10}}`,
		"data: [DONE]",
		"",
	}, "\n\n")

	c := &Client{}
	res, err := c.consume(responseWithBody(sse), Handlers{}, false)
	if err != nil {
		t.Fatal(err)
	}
	if res.CacheStatus != "hit" {
		t.Errorf("cache status = %q, want hit", res.CacheStatus)
	}
	if !strings.Contains(res.CacheDetail, "prompt_cache_hit_tokens=90") {
		t.Errorf("cache detail = %q", res.CacheDetail)
	}
}
