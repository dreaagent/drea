package llm

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math/rand"
	"net/http"
	"os"
	"regexp"
	"strconv"
	"strings"
	"time"
)

// ErrSchemaRejected is returned when the endpoint explicitly rejects the
// response_format schema, usually because it does not support the strict flag.
var ErrSchemaRejected = errors.New("endpoint rejected the strict JSON schema")

// Client talks to an OpenAI-compatible chat completions endpoint over HTTP.
type Client struct {
	url    string
	apiKey string
	model  string
	temp   float64
	topP   float64
	http   *http.Client

	// maxRetries is how many additional attempts are made after the first
	// when the endpoint returns a transient error (429/5xx) or the request
	// fails at the network level. baseDelay seeds the exponential backoff.
	maxRetries int
	baseDelay  time.Duration
	// jsonMode asks the endpoint to constrain output to a strict JSON schema.
	jsonMode bool
	// jsonFormat selects the response_format variant (json_schema or json_object).
	jsonFormat string
	// reasoningEffort is the reasoning effort level forwarded to the endpoint.
	// Empty means omit from the request so the endpoint default is used.
	reasoningEffort string
	// promptCacheKey is a non-standard extension sent as "prompt_cache_key" in
	// the request body. Empty means omit the field.
	promptCacheKey string
	// debug, when non-empty, is the path to append raw request/response traffic
	// to for inspection.
	debug string
}

// NewClient constructs a client. timeout bounds a single request (including
// the full streamed body). jsonMode, when set, asks the endpoint to constrain
// output to a strict JSON schema describing a tool call. jsonFormat selects
// the response_format variant to send (json_schema or json_object).
func NewClient(chatURL, apiKey, model string, temperature float64, timeout time.Duration, jsonMode bool, jsonFormat string) *Client {
	return NewClientWithTopP(chatURL, apiKey, model, temperature, 0.95, timeout, jsonMode, jsonFormat)
}

// NewClientWithTopP constructs a client with explicit top-p nucleus sampling.
// The supplied topP value is always forwarded in requests.
func NewClientWithTopP(chatURL, apiKey, model string, temperature, topP float64, timeout time.Duration, jsonMode bool, jsonFormat string) *Client {
	return NewClientWithReasoning(chatURL, apiKey, model, temperature, topP, "", timeout, jsonMode, jsonFormat)
}

// NewClientWithReasoning constructs a client with explicit top-p and reasoning
// effort. The supplied reasoningEffort value is forwarded only when non-empty.
func NewClientWithReasoning(chatURL, apiKey, model string, temperature, topP float64, reasoningEffort string, timeout time.Duration, jsonMode bool, jsonFormat string) *Client {
	if jsonFormat == "" {
		jsonFormat = "json_schema"
	}
	return &Client{
		url:             chatURL,
		apiKey:          apiKey,
		model:           model,
		temp:            temperature,
		topP:            topP,
		reasoningEffort: reasoningEffort,
		jsonMode:        jsonMode,
		jsonFormat:      jsonFormat,
		http:            &http.Client{Timeout: timeout},
		maxRetries:      5,
		baseDelay:       time.Second,
	}
}

// Handlers receives streaming callbacks during a completion. All fields are
// optional; a nil func is simply not called.
type Handlers struct {
	// OnContent is called with each assistant text delta as it arrives.
	OnContent func(delta string)
	// OnReasoning is called with each chain-of-thought delta as it arrives
	// (reasoning_content). It is display-only and never sent back to the model.
	OnReasoning func(delta string)
	// OnToolName is called once per tool call the first time its name is
	// known, with the call's stream index and resolved tool name.
	OnToolName func(index int, name string)
	// OnToolArgs is called with each argument fragment for a tool call,
	// identified by its stream index, as the arguments stream in.
	OnToolArgs func(index int, fragment string)
	// OnRetry is called before a transient failure is retried, with the
	// upcoming (1-based) attempt number, the delay before it, and the error
	// that triggered the retry. Useful for keeping the UI informative.
	OnRetry func(attempt int, delay time.Duration, err error)
}

// Model returns the model name currently in use.
func (c *Client) Model() string { return c.model }

// JSONMode reports whether the client constrains output to a JSON schema.
func (c *Client) JSONMode() bool { return c.jsonMode }

// SetModel changes the model used for subsequent requests.
func (c *Client) SetModel(model string) { c.model = model }

// SetChatURL changes the chat completions endpoint for subsequent requests.
func (c *Client) SetChatURL(url string) { c.url = url }

// SetReasoningEffort changes the reasoning effort level for subsequent requests.
func (c *Client) SetReasoningEffort(level string) { c.reasoningEffort = level }

// SetAPIKey changes the API key used for subsequent requests.
func (c *Client) SetAPIKey(key string) { c.apiKey = key }

// SetDebug enables raw request/response dumping to the given path.
func (c *Client) SetDebug(path string) { c.debug = path }

// SetPromptCacheKey sets the prompt cache key sent with each request. Empty
// disables it.
func (c *Client) SetPromptCacheKey(key string) { c.promptCacheKey = key }

// Stream sends the conversation and tool definitions, invoking h's callbacks
// as content and tool-call fragments arrive. Tool-call fragments are also
// accumulated internally and returned whole in the Result.
//
// Transient failures — a 429 (rate limit), a 5xx, or a network-level error
// before the stream begins — are retried with exponential backoff (honouring a
// Retry-After header when present) up to maxRetries times, so a rate limit no
// longer aborts the task. Retry details are explained in calm, model-facing
// language so the agent does not mistake an API capacity issue for a problem
// with the workspace or its own actions. Handlers are invoked only once the
// chosen attempt starts streaming, so retries never double up output.
func (c *Client) Stream(ctx context.Context, messages []Message, tools []Tool, h Handlers) (*Result, error) {
	return c.stream(ctx, messages, tools, h, c.jsonMode)
}

// StreamPlain streams a plain-text completion without constraining the output
// to a JSON schema. It is used for internal helper prompts (reply
// classification, context compaction) that expect prose answers even when the
// main agent loop runs in JSON mode.
func (c *Client) StreamPlain(ctx context.Context, messages []Message, h Handlers) (*Result, error) {
	return c.stream(ctx, messages, nil, h, false)
}

func (c *Client) stream(ctx context.Context, messages []Message, tools []Tool, h Handlers, jsonMode bool) (*Result, error) {
	body := request{
		Model:           c.model,
		Messages:        messages,
		Tools:           tools,
		Temperature:     c.temp,
		ReasoningEffort: c.reasoningEffort,
		Stream:          true,
		StreamOptions:   &streamOptions{IncludeUsage: true},
		PromptCacheKey:  c.promptCacheKey,
	}
	body.TopP = c.topP
	if jsonMode {
		body.ResponseFormat = toolCallResponseFormat(c.jsonFormat)
	}
	raw, err := json.Marshal(body)
	if err != nil {
		return nil, err
	}

	c.debugWrite("REQUEST\n" + string(raw) + "\n\n")
	for attempt := 0; ; attempt++ {
		res, transient, ra, err := c.attempt(ctx, raw, h, jsonMode)
		if err == nil {
			return res, nil
		}
		if !transient || ctx.Err() != nil {
			return nil, err
		}
		if attempt == c.maxRetries {
			return nil, fmt.Errorf("Drea's connection to the AI API endpoint is still unavailable after %d retries. This is an external provider issue, not caused by anything you did, any command you ran, or any action you took on this project. Please wait a moment and try again.", c.maxRetries+1)
		}
		delay := c.backoff(attempt+1, ra)
		if h.OnRetry != nil {
			h.OnRetry(attempt+1, delay, retryError())
		}
		if err := sleep(ctx, delay); err != nil {
			return nil, err
		}
	}
}

// retryError returns a calm, explanatory message for transient API failures.
// It addresses the model directly and makes clear the issue is Drea's
// connection to the provider, not the workspace or the model's actions.
func retryError() error {
	return errors.New("Drea's connection to the AI API endpoint is temporarily overloaded or rate-limiting (HTTP 429/5xx). This is not caused by anything you did, any command you ran, or any action you took on this project; Drea will retry automatically.")
}

// attempt performs a single request. transient reports whether a returned
// error is worth retrying; ra carries any Retry-After hint from the server.
func (c *Client) attempt(ctx context.Context, raw []byte, h Handlers, jsonMode bool) (res *Result, transient bool, ra time.Duration, err error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.url, bytes.NewReader(raw))
	if err != nil {
		return nil, false, 0, err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "text/event-stream")
	if c.apiKey != "" {
		req.Header.Set("Authorization", "Bearer "+c.apiKey)
	}

	resp, err := c.http.Do(req)
	if err != nil {
		// A cancelled context is terminal; any other transport error is
		// worth retrying (connection reset, temporary DNS failure, etc.).
		return nil, ctx.Err() == nil, 0, err
	}

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		snippet, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
		ra := parseRetryAfter(resp.Header.Get("Retry-After"))
		status := resp.Status
		resp.Body.Close()
		err := fmt.Errorf("endpoint returned %s: %s", status, strings.TrimSpace(string(snippet)))
		if resp.StatusCode == http.StatusBadRequest && jsonMode && schemaError(string(snippet)) {
			err = fmt.Errorf("%w; try --json-format json_object or --no-json-mode (%v)", ErrSchemaRejected, err)
		}
		return nil, retriableStatus(resp.StatusCode), ra, err
	}

	// Success: stream the body. A mid-stream failure is not retried because
	// partial output has already been delivered to the handlers.
	res, err = c.consume(resp, h, jsonMode)
	resp.Body.Close()
	return res, false, 0, err
}

// schemaError reports whether an endpoint error message is about the JSON
// schema / response_format being rejected.
func schemaError(msg string) bool {
	m := strings.ToLower(msg)
	return strings.Contains(m, "strict") ||
		strings.Contains(m, "response_format") ||
		strings.Contains(m, "json_schema") ||
		strings.Contains(m, "json_object") ||
		strings.Contains(m, "schema")
}

// retriableStatus reports whether an HTTP status warrants a retry.
func retriableStatus(code int) bool {
	switch code {
	case http.StatusTooManyRequests,
		http.StatusInternalServerError,
		http.StatusBadGateway,
		http.StatusServiceUnavailable,
		http.StatusGatewayTimeout:
		return true
	}
	return false
}

// backoff returns the delay before the given attempt (1-based). A server
// Retry-After hint wins; otherwise it is exponential with full jitter, capped.
func (c *Client) backoff(attempt int, retryAfter time.Duration) time.Duration {
	const maxDelay = 30 * time.Second
	if retryAfter > 0 {
		if retryAfter > maxDelay {
			return maxDelay
		}
		return retryAfter
	}
	d := c.baseDelay * (1 << (attempt - 1))
	if d > maxDelay || d <= 0 {
		d = maxDelay
	}
	return d/2 + time.Duration(rand.Int63n(int64(d/2)+1))
}

// parseRetryAfter interprets a Retry-After header, which may be either a number
// of seconds or an HTTP date. It returns 0 when the value is absent or invalid.
func parseRetryAfter(v string) time.Duration {
	v = strings.TrimSpace(v)
	if v == "" {
		return 0
	}
	if secs, err := strconv.Atoi(v); err == nil {
		if secs < 0 {
			return 0
		}
		return time.Duration(secs) * time.Second
	}
	if t, err := http.ParseTime(v); err == nil {
		if d := time.Until(t); d > 0 {
			return d
		}
	}
	return 0
}

// sleep waits for d or until ctx is cancelled, whichever comes first.

// cacheStatusFromHeaders interprets common provider-specific headers that report
// whether the prompt prefix was served from cache. It returns a normalized
// status ("hit", "miss", or empty) and a human-readable detail string.
func cacheStatusFromHeaders(h http.Header) (status, detail string) {
	// Headers ordered from most specific to least specific.
	candidates := []string{
		"X-Prompt-Cache-Hit",
		"X-Prompt-Cache",
		"X-Prompt-Cache-Status",
		"X-Cache-Status",
		"X-Cache",
		"X-Cache-Lookup",
		"Cf-Cache-Status",
	}
	for _, key := range candidates {
		if v := h.Get(key); v != "" {
			return normalizeCacheStatus(v), key + ": " + v
		}
	}
	return "", ""
}

func normalizeCacheStatus(v string) string {
	v = strings.ToLower(strings.TrimSpace(v))
	switch {
	case strings.Contains(v, "hit"), v == "true":
		return "hit"
	case strings.Contains(v, "miss"), v == "false":
		return "miss"
	}
	return ""
}

// debugWrite appends text to the debug log when debugging is enabled.
func (c *Client) debugWrite(text string) {
	if c.debug == "" {
		return
	}
	f, err := os.OpenFile(c.debug, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o600)
	if err != nil {
		return
	}
	defer f.Close()
	f.WriteString(text)
}

// teeReader returns a reader that copies everything read into the debug log.
func (c *Client) teeReader(r io.Reader) io.Reader {
	pr, pw := io.Pipe()
	go func() {
		defer pw.Close()
		f, err := os.OpenFile(c.debug, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o600)
		if err != nil {
			io.Copy(pw, r)
			return
		}
		defer f.Close()
		f.WriteString("RESPONSE\n")
		mw := io.MultiWriter(pw, f)
		io.Copy(mw, r)
		f.WriteString("\n\n")
	}()
	return pr
}

func sleep(ctx context.Context, d time.Duration) error {
	if d <= 0 {
		return ctx.Err()
	}
	t := time.NewTimer(d)
	defer t.Stop()
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-t.C:
		return nil
	}
}

// consume reads the SSE stream line-by-line, merging content and tool-call
// deltas into a single Result. It is tolerant of blank lines, comments and
// non-JSON frames so no token is ever dropped on a well-formed stream.
func (c *Client) consume(resp *http.Response, h Handlers, jsonMode bool) (*Result, error) {
	r := io.Reader(resp.Body)
	if c.debug != "" {
		r = c.teeReader(r)
	}
	res := &Result{}
	res.CacheStatus, res.CacheDetail = cacheStatusFromHeaders(resp.Header)
	// tool calls accumulated by their stream index.
	acc := map[int]*ToolCall{}
	var order []int
	// In JSON mode content may be ordinary prose (fallback) or a JSON
	// serialization of tool calls. We stream prose live; reply pseudo-tools
	// are also streamed by decoding their message field as the JSON arrives.
	streamedContent := false
	var replyEx *replyExtractor
	if jsonMode && h.OnContent != nil {
		replyEx = newReplyExtractor(func(s string) {
			if s != "" {
				h.OnContent(s)
			}
		})
	}

	scanner := bufio.NewScanner(r)
	scanner.Buffer(make([]byte, 0, 64*1024), 8*1024*1024)

	for scanner.Scan() {
		line := scanner.Text()
		if !strings.HasPrefix(line, "data:") {
			continue // skip comments / event: lines / blanks
		}
		data := strings.TrimSpace(line[len("data:"):])
		if data == "" {
			continue
		}
		if data == "[DONE]" {
			break
		}

		var chunk streamChunk
		if err := json.Unmarshal([]byte(data), &chunk); err != nil {
			continue // ignore malformed frames rather than aborting
		}
		if chunk.Usage != nil {
			res.Usage = *chunk.Usage
		}
		if len(chunk.Choices) == 0 {
			continue
		}
		choice := chunk.Choices[0]
		if choice.Delta.Content != "" {
			res.Content += choice.Delta.Content
			if h.OnContent != nil {
				if !jsonMode {
					// Non-JSON mode: content is ordinary prose, stream it live.
					h.OnContent(choice.Delta.Content)
					streamedContent = true
				} else {
					// JSON mode: content is either a JSON tool-call serialization
					// or plain prose that failed to conform to the schema. Stream
					// it live only when it does not look like JSON, so the user
					// never sees partial tool-call JSON scroll by. If it is a
					// reply pseudo-tool, decode and stream its message field.
					trimmed := strings.TrimSpace(res.Content)
					if !strings.HasPrefix(trimmed, "[") && !strings.HasPrefix(trimmed, "{") {
						h.OnContent(choice.Delta.Content)
						streamedContent = true
					} else if replyEx != nil {
						before := replyEx.emitted
						replyEx.feed(choice.Delta.Content)
						if replyEx.emitted > before {
							streamedContent = true
						}
					}
				}
			}
		}
		// Chain-of-thought is streamed separately from the answer and is
		// always display-only, so it is surfaced live regardless of JSON mode.
		if choice.Delta.ReasoningContent != "" {
			res.ReasoningContent += choice.Delta.ReasoningContent
			if h.OnReasoning != nil {
				h.OnReasoning(choice.Delta.ReasoningContent)
			}
		}
		// Some endpoints (e.g. llama.cpp) stream JSON-mode tool calls as
		// content deltas but never set a finish_reason on the same frame.
		// Reconstruct tool calls incrementally so the caller can dispatch them
		// as soon as the stream ends, rather than waiting for a finish_reason
		// that may never arrive.
		if jsonMode && res.Content != "" && len(res.ToolCalls) == 0 {
			if tcs, ok := parseJSONModeToolCalls(res.Content); ok {
				res.ToolCalls = tcs
			}
		}
		for _, tc := range choice.Delta.ToolCalls {
			cur, ok := acc[tc.Index]
			if !ok {
				cur = &ToolCall{Type: "function"}
				acc[tc.Index] = cur
				order = append(order, tc.Index)
			}
			if tc.ID != "" {
				cur.ID = tc.ID
			}
			if tc.Type != "" {
				cur.Type = tc.Type
			}
			if tc.Function.Name != "" {
				named := cur.Function.Name == ""
				cur.Function.Name += tc.Function.Name
				if named && h.OnToolName != nil {
					h.OnToolName(tc.Index, cur.Function.Name)
				}
			}
			if tc.Function.Arguments != "" {
				cur.Function.Arguments += tc.Function.Arguments
				if h.OnToolArgs != nil {
					h.OnToolArgs(tc.Index, tc.Function.Arguments)
				}
			}
		}
		// In JSON mode some endpoints stream the constrained JSON array as
		// plain content deltas with no tool_calls array. Reconstruct the tool
		// calls from the accumulated content so downstream code always sees
		// native tool calls.
		if jsonMode && res.Content != "" && len(res.ToolCalls) == 0 && choice.FinishReason != "" {
			if tcs, ok := parseJSONModeToolCalls(res.Content); ok {
				res.ToolCalls = tcs
			}
		}
		if choice.FinishReason != "" {
			res.FinishReason = choice.FinishReason
		}
	}
	if err := scanner.Err(); err != nil {
		return nil, err
	}

	for _, idx := range order {
		res.ToolCalls = append(res.ToolCalls, *acc[idx])
	}
	// If the endpoint streamed JSON-mode content as plain deltas, the
	// per-chunk reconstruction above may already have populated ToolCalls.
	// Make sure it is also reflected in the accumulated result.
	if jsonMode && res.Content != "" && len(res.ToolCalls) == 0 {
		if tcs, ok := parseJSONModeToolCalls(res.Content); ok {
			res.ToolCalls = tcs
		}
	}
	// If every parsed tool call is a non-empty "reply" pseudo-tool, treat the
	// whole turn as plain assistant prose so the agent stops and shows it to the
	// user. Empty replies are dropped so a confused model cannot stall the task
	// by emitting vacuous "[]" turns.
	released := false
	if jsonMode && len(res.ToolCalls) > 0 {
		allReply := true
		for _, tc := range res.ToolCalls {
			if !IsReplyCall(tc) {
				allReply = false
				break
			}
		}
		if allReply {
			var b strings.Builder
			for _, tc := range res.ToolCalls {
				msg := ReplyMessage(tc)
				if msg == "" {
					continue
				}
				if b.Len() > 0 {
					b.WriteByte('\n')
				}
				b.WriteString(msg)
			}
			replyContent := b.String()
			// Keep the original tool calls in res.ToolCalls so the agent can
			// recognise a reply-only turn consistently in both JSON and non-JSON
			// modes. The joined prose is used only for live streaming/release.
			res.ReplyStreamed = streamedContent
			released = replyContent != "" || streamedContent
			if h.OnContent != nil && replyContent != "" && !streamedContent {
				h.OnContent(replyContent)
			}
		}
	}
	// JSON mode content that could not be parsed as tool calls is ordinary
	// assistant prose; release it now so the caller can display it.
	if jsonMode && res.Content != "" && len(res.ToolCalls) == 0 && !released && !streamedContent {
		if h.OnContent != nil {
			h.OnContent(res.Content)
		}
	}
	if jsonMode && !released && len(res.ToolCalls) > 0 {
		// The content was a tool-call serialization; do not leave it in the
		// assistant message or the model will see its own JSON schema output.
		res.Content = ""
	}
	// If the provider reported cached prompt tokens in usage, reflect that in
	// the cache detail even when no cache header was present.
	if res.Usage.PromptCacheHitTokens+res.Usage.PromptCacheMissTokens > 0 {
		if res.Usage.PromptCacheHitTokens > 0 && res.CacheStatus == "" {
			res.CacheStatus = "hit"
		}
		res.CacheDetail = fmt.Sprintf("prompt_cache_hit_tokens=%d, prompt_cache_miss_tokens=%d",
			res.Usage.PromptCacheHitTokens, res.Usage.PromptCacheMissTokens)
	}
	return res, nil
}

// replyExtractor streams the message text from a JSON-mode "reply" pseudo-tool
// as the JSON object streams in. It is narrow on purpose: it only recognises
// the schema-constrained shape produced for a reply tool call, so ordinary
// prose and real tool calls are left alone.
type replyExtractor struct {
	onFragment func(string)
	buf        strings.Builder
	// offset is the byte position in buf where the current message string
	// value begins, or -1 while we are still searching for a reply marker.
	offset int
	// searchFrom is the earliest byte position worth scanning for the next
	// reply marker; it advances past messages we have already consumed so we
	// cannot loop over the same reply indefinitely.
	searchFrom int
	// escaped is true when the next byte follows a backslash inside a message
	// string.
	escaped bool
	// needsSeparator is true when the previous non-empty reply message ended
	// and the next non-empty reply should be preceded by a newline, matching
	// the end-of-turn formatting (concatenated with "\n").
	needsSeparator bool
	// pendingSep is a separator that has not been emitted yet because the
	// reply that follows it is still being decoded or was empty.
	pendingSep bool
	// emitted counts how many non-empty fragments have been delivered, so the
	// caller knows live streaming happened.
	emitted int
}

func newReplyExtractor(onFragment func(string)) *replyExtractor {
	return &replyExtractor{onFragment: onFragment, offset: -1}
}

// feed appends a content delta and emits any newly decoded reply message
// fragments via onFragment.
func (r *replyExtractor) feed(delta string) {
	r.buf.WriteString(delta)
	for {
		if r.offset < 0 {
			off, ok := findReplyMessageOffset(r.buf.String(), r.searchFrom)
			if !ok {
				return
			}
			if r.needsSeparator {
				r.pendingSep = true
				r.needsSeparator = false
			}
			r.offset = off
		}
		s := r.buf.String()
		fragment, done := r.decodeStringFragment(s)
		if fragment != "" {
			if r.pendingSep {
				r.onFragment("\n")
				r.pendingSep = false
			}
			r.onFragment(fragment)
			r.emitted++
		}
		if !done {
			return
		}
		// Closing quote reached; next search starts after it so we never
		// re-match a reply we have already streamed.
		r.searchFrom = r.offset
		r.offset = -1
		if fragment != "" {
			r.needsSeparator = true
		}
	}
}

// decodeStringFragment decodes the JSON string starting at r.offset. It returns
// the decoded text that is fully available and whether the closing quote has
// been reached.
func (r *replyExtractor) decodeStringFragment(s string) (string, bool) {
	var out strings.Builder
	for r.offset < len(s) {
		c := s[r.offset]
		if r.escaped {
			switch c {
			case '"', '\\', '/':
				out.WriteByte(c)
			case 'n':
				out.WriteByte('\n')
			case 't':
				out.WriteByte('\t')
			case 'r':
				out.WriteByte('\r')
			case 'b':
				out.WriteByte('\b')
			case 'f':
				out.WriteByte('\f')
			case 'u':
				if r.offset+4 >= len(s) {
					return out.String(), false
				}
				code, err := strconv.ParseUint(s[r.offset+1:r.offset+5], 16, 32)
				if err != nil {
					out.WriteByte(c)
				} else {
					out.WriteRune(rune(code))
				}
				r.offset += 5
				r.escaped = false
				continue
			default:
				out.WriteByte(c)
			}
			r.offset++
			r.escaped = false
			continue
		}
		if c == '\\' {
			r.escaped = true
			r.offset++
			continue
		}
		if c == '"' {
			r.offset++
			return out.String(), true
		}
		out.WriteByte(c)
		r.offset++
	}
	return out.String(), false
}

// replyNameRE matches the "name":"reply" marker; replyMessageRE matches the
// "message":" marker that must follow it inside the same reply object.
var (
	replyNameRE    = regexp.MustCompile(`"name"\s*:\s*"reply"`)
	replyMessageRE = regexp.MustCompile(`"message"\s*:\s*"`)
)

// findReplyMessageOffset returns the byte offset immediately after the opening
// quote of a reply tool's message string value, or ok=false when no such marker
// is present yet.
func findReplyMessageOffset(s string, start int) (int, bool) {
	for {
		loc := replyNameRE.FindStringIndex(s[start:])
		if loc == nil {
			return 0, false
		}
		nameEnd := start + loc[1]
		msgLoc := replyMessageRE.FindStringIndex(s[nameEnd:])
		if msgLoc == nil {
			// This name:reply has no following message marker yet; keep looking.
			start = nameEnd
			continue
		}
		return nameEnd + msgLoc[1], true
	}
}
