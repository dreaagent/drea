// Package llm is a minimal client for OpenAI-compatible chat completion APIs.
// It depends only on the Go standard library.
package llm

import (
	"encoding/json"
	"strings"
)

// Role constants for chat messages.
const (
	RoleSystem    = "system"
	RoleUser      = "user"
	RoleAssistant = "assistant"
	RoleTool      = "tool"
)

// Message is a single entry in the conversation.
type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
	// Name optionally identifies the author (unused for most roles).
	Name string `json:"name,omitempty"`
	// ToolCalls is populated on assistant messages that request tool use.
	ToolCalls []ToolCall `json:"tool_calls,omitempty"`
	// ToolCallID links a tool result back to the assistant's request.
	ToolCallID string `json:"tool_call_id,omitempty"`
}

// ToolCall is a single function call requested by the model.
type ToolCall struct {
	ID       string       `json:"id"`
	Type     string       `json:"type"`
	Function FunctionCall `json:"function"`
}

// FunctionCall carries the function name and raw JSON argument string.
type FunctionCall struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

// Tool is a function definition advertised to the model.
type Tool struct {
	Type     string       `json:"type"`
	Function FunctionSpec `json:"function"`
}

// FunctionSpec describes a callable function and its JSON-schema parameters.
type FunctionSpec struct {
	Name        string          `json:"name"`
	Description string          `json:"description"`
	Parameters  json.RawMessage `json:"parameters"`
}

// Usage reports token accounting when the endpoint provides it. Some providers
// also break prompt tokens into cached and uncached portions.
type Usage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
	// PromptCacheHitTokens is the number of prompt tokens served from cache.
	PromptCacheHitTokens int `json:"prompt_cache_hit_tokens,omitempty"`
	// PromptCacheMissTokens is the number of prompt tokens not served from cache.
	PromptCacheMissTokens int `json:"prompt_cache_miss_tokens,omitempty"`
}

// Result is the accumulated outcome of one streamed completion.
type Result struct {
	Content string
	// ReasoningContent is the model's chain-of-thought tokens, streamed by
	// reasoning-capable models (e.g. via reasoning_content). It is display-only
	// and never included in the conversation messages.
	ReasoningContent string
	ToolCalls        []ToolCall
	FinishReason     string
	Usage            Usage
	// CacheStatus is a short label for whether the prompt was served from cache:
	// "hit", "miss", or empty when the provider did not say.
	CacheStatus string
	// CacheDetail carries the raw header/value that determined CacheStatus, so
	// the user can see *how* the provider reported it.
	CacheDetail string
	// ReplyStreamed is true when a JSON-mode reply pseudo-tool was streamed to
	// the UI as it arrived. The agent should not emit the same prose again when
	// recording a reply-only turn.
	ReplyStreamed bool
}

// request is the JSON body sent to /chat/completions.
type request struct {
	Model           string          `json:"model"`
	Messages        []Message       `json:"messages"`
	Tools           []Tool          `json:"tools,omitempty"`
	Temperature     float64         `json:"temperature"`
	TopP            float64         `json:"top_p"`
	ReasoningEffort string          `json:"reasoning_effort,omitempty"`
	Stream          bool            `json:"stream"`
	StreamOptions   *streamOptions  `json:"stream_options,omitempty"`
	ResponseFormat  *responseFormat `json:"response_format,omitempty"`
	// PromptCacheKey is a non-standard extension used by some gateways to
	// identify a prompt whose prefix should be cached across requests.
	PromptCacheKey string `json:"prompt_cache_key,omitempty"`
}

type streamOptions struct {
	IncludeUsage bool `json:"include_usage"`
}

// responseFormat constrains the model's output to a strict JSON schema. It is
// sent only when JSON mode is enabled; not every OpenAI-compatible endpoint
// supports it, so it is opt-in.
type responseFormat struct {
	Type       string          `json:"type"`
	JSONSchema json.RawMessage `json:"json_schema,omitempty"`
	Schema     json.RawMessage `json:"schema,omitempty"`
}

// toolCallSchema is the JSON schema describing one or more action items. Each
// item is either a real tool call or a "reply" pseudo-tool that carries plain
// assistant prose. Keeping both shapes in the same constrained schema lets the
// model talk to the user without breaking out of JSON mode.
//
// minItems: 1 prevents the model from emitting an empty array "[]" as a
// no-op turn, which the harness would otherwise mistake for a finished task
// and leave the user staring at a bare "[]".
const toolCallSchema = `{
	"type": "array",
	"minItems": 1,
	"items": {
		"type": "object",
		"properties": {
			"name": {"type": "string", "description": "Name of the tool to call, or 'reply' to send a plain-text message to the user."},
			"arguments": {
				"anyOf": [
					{"type": "object", "description": "Arguments for the named tool call."},
					{"type": "object", "properties": {"message": {"type": "string", "minLength": 1}}, "required": ["message"], "additionalProperties": false}
				]
			}
		},
		"required": ["name", "arguments"],
		"additionalProperties": false
	}
}`

// toolCallSchemaStrict is the same schema with OpenAI's strict flag embedded,
// used by response_format variants that place strict inside the schema itself.
const toolCallSchemaStrict = `{
	"strict": true,
	"type": "array",
	"minItems": 1,
	"items": {
		"type": "object",
		"properties": {
			"name": {"type": "string", "description": "Name of the tool to call, or 'reply' to send a plain-text message to the user."},
			"arguments": {
				"anyOf": [
					{"type": "object", "description": "Arguments for the named tool call."},
					{"type": "object", "properties": {"message": {"type": "string", "minLength": 1}}, "required": ["message"], "additionalProperties": false}
				]
			}
		},
		"required": ["name", "arguments"],
		"additionalProperties": false
	}
}`

// toolCallResponseFormat returns a strict JSON schema that forces the model to
// emit one or more tool calls. Small models without reliable native tool-calling
// can emit a JSON array of calls; the harness parses it back into native
// tool_calls so the rest of the loop treats them normally.
//
// format selects the wrapper expected by the endpoint:
//   - "json_schema" (default): OpenAI structured outputs.
//   - "json_object":           json_object with a top-level schema; preferred
//     for llama.cpp and older servers.
func toolCallResponseFormat(format string) *responseFormat {
	switch format {
	case "json_object":
		return &responseFormat{
			Type:   "json_object",
			Schema: json.RawMessage(toolCallSchemaStrict),
		}
	default:
		return &responseFormat{
			Type: "json_schema",
			JSONSchema: json.RawMessage(`{
				"name": "tool_calls",
				"strict": true,
				"schema": ` + toolCallSchema + `
			}`),
		}
	}
}

// parseJSONModeToolCalls parses the JSON array (or single object, for backward
// compatibility) emitted when response_format constrains the model to the
// tool_calls schema. It normalises the result into native ToolCall values.
// Items whose name is "reply" are returned as tool calls with that name; the
// agent treats them as a plain-text assistant reply.
func parseJSONModeToolCalls(content string) ([]ToolCall, bool) {
	s := strings.TrimSpace(content)
	if s == "" {
		return nil, false
	}

	var calls []ToolCall
	if strings.HasPrefix(s, "[") {
		var arr []struct {
			Name      string          `json:"name"`
			Arguments json.RawMessage `json:"arguments"`
		}
		if err := json.Unmarshal([]byte(s), &arr); err != nil {
			return nil, false
		}
		for _, obj := range arr {
			if obj.Name == "" {
				continue
			}
			args := string(obj.Arguments)
			if args == "" {
				args = "{}"
			}
			calls = append(calls, ToolCall{Type: "function", Function: FunctionCall{Name: obj.Name, Arguments: args}})
		}
	} else if strings.HasPrefix(s, "{") {
		var obj struct {
			Name      string          `json:"name"`
			Arguments json.RawMessage `json:"arguments"`
		}
		if err := json.Unmarshal([]byte(s), &obj); err != nil {
			return nil, false
		}
		if obj.Name == "" {
			return nil, false
		}
		args := string(obj.Arguments)
		if args == "" {
			args = "{}"
		}
		calls = append(calls, ToolCall{Type: "function", Function: FunctionCall{Name: obj.Name, Arguments: args}})
	}
	return calls, len(calls) > 0
}

// IsReplyCall reports whether a tool call is the special "reply" pseudo-tool
// used in JSON mode to carry plain assistant prose.
func IsReplyCall(tc ToolCall) bool {
	return tc.Function.Name == "reply"
}

// ReplyMessage extracts the plain-text message from a "reply" pseudo-tool call.
// It returns the empty string when the message is missing or blank, which lets
// the harness distinguish a real reply from a vacuous one.
func ReplyMessage(tc ToolCall) string {
	var a struct {
		Message string `json:"message"`
	}
	_ = json.Unmarshal([]byte(tc.Function.Arguments), &a)
	return strings.TrimSpace(a.Message)
}

// streamChunk is one SSE frame from the endpoint.
type streamChunk struct {
	Choices []struct {
		Delta struct {
			Content          string          `json:"content"`
			ReasoningContent string          `json:"reasoning_content"`
			ToolCalls        []toolCallDelta `json:"tool_calls"`
		} `json:"delta"`
		FinishReason string `json:"finish_reason"`
	} `json:"choices"`
	Usage *Usage `json:"usage"`
}

// toolCallDelta is the incremental, index-addressed tool-call fragment that
// streaming responses emit; fragments are merged by Index.
type toolCallDelta struct {
	Index    int    `json:"index"`
	ID       string `json:"id"`
	Type     string `json:"type"`
	Function struct {
		Name      string `json:"name"`
		Arguments string `json:"arguments"`
	} `json:"function"`
}
