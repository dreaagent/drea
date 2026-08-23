package tool

import (
	"context"
	"encoding/json"
	"errors"
	"strings"

	"github.com/dreaagent/drea/internal/llm"
)

// RecordThoughtsTool is the LLM-facing spec for the record_thoughts tool. It is
// exported so the agent can offer the tool in internal follow-up requests.
var RecordThoughtsTool = llm.Tool{
	Type: "function",
	Function: llm.FunctionSpec{
		Name:        "record_thoughts",
		Description: "Record your reasoning, plan, or intermediate conclusions into the conversation context. Use this when you need to think step-by-step or write down a strategy before acting. This is not visible to the user as a reply; it is context for your own later use.",
		Parameters: json.RawMessage(`{
  "type":"object",
  "properties":{
    "thoughts":{"type":"string","description":"Your reasoning, plan, or conclusions. Be concise but complete enough to continue from in the next turn."}
  },
  "required":["thoughts"],
  "additionalProperties":false
}`),
	},
}

// recordThoughts lets the model persist reasoning or a plan into the chat
// transcript without mutating the workspace. It is not a user-facing reply;
// it is context the model writes for itself so later turns can build on it.
// Because it is read-only it never requires approval.
type recordThoughts struct{}

func (t *recordThoughts) Name() string   { return RecordThoughtsTool.Function.Name }
func (t *recordThoughts) Mutating() bool { return false }
func (t *recordThoughts) Description() string {
	return RecordThoughtsTool.Function.Description
}
func (t *recordThoughts) Schema() json.RawMessage {
	return RecordThoughtsTool.Function.Parameters
}
func (t *recordThoughts) Summary(args json.RawMessage) string {
	var a struct {
		Thoughts string `json:"thoughts"`
	}
	_ = decode(args, &a)
	s := strings.TrimSpace(a.Thoughts)
	if len(s) > 80 {
		s = s[:77] + "..."
	}
	return s
}
func (t *recordThoughts) Run(_ context.Context, args json.RawMessage) (string, error) {
	var a struct {
		Thoughts string `json:"thoughts"`
	}
	if err := decode(args, &a); err != nil {
		return "", err
	}
	if strings.TrimSpace(a.Thoughts) == "" {
		return "", errors.New("thoughts cannot be empty")
	}
	return "thoughts recorded", nil
}
