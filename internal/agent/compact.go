package agent

import (
	"context"
	"fmt"
	"strings"

	"github.com/dreaagent/drea/internal/llm"
)

const (
	// keepRecent is how many of the most recent messages are always retained
	// verbatim when older history is compacted.
	keepRecent = 8
	// minEvict is the smallest number of old messages worth summarizing; below
	// it, compaction is skipped to avoid churn.
	minEvict = 4
	// maxTranscriptChars bounds the transcript handed to the summarizer (and
	// the raw fallback), keeping the compaction request itself small.
	maxTranscriptChars = 48 * 1024
	// maxToolResultChars bounds a single tool result stored in history so one
	// large read or command output cannot dominate the context window.
	maxToolResultChars = 16 * 1024
)

// estimateTokens approximates the token count of a message list. The standard
// library has no tokenizer, so it uses the common ~4-characters-per-token rule
// plus a small per-message overhead. It is only a fallback: the accurate
// prompt-token count reported by the endpoint is preferred when available.
func estimateTokens(msgs []llm.Message) int {
	total := 0
	for _, m := range msgs {
		n := len(m.Content)
		for _, tc := range m.ToolCalls {
			n += len(tc.Function.Name) + len(tc.Function.Arguments)
		}
		total += n/4 + 8
	}
	return total
}

// maybeCompact summarizes the older part of the conversation into a single
// note when the prompt has grown past the configured token budget, so a long
// task cannot overflow the model's context window. The system prompt and the
// most recent messages are always kept verbatim, and an assistant tool-call is
// never separated from its tool results.
func (a *Agent) maybeCompact(ctx context.Context) {
	budget := a.cfg.ContextTokens
	if budget <= 0 {
		return
	}
	size := a.lastPromptTokens
	if size == 0 {
		size = estimateTokens(a.messages)
	}
	if size <= budget {
		return
	}

	cut := len(a.messages) - keepRecent
	// Back the cut up so the first retained message is not an orphaned tool
	// result whose originating assistant message would be evicted.
	for cut > 1 && a.messages[cut].Role == llm.RoleTool {
		cut--
	}
	if cut < 2 {
		return // only the system prompt would remain before the tail
	}
	evicted := a.messages[1:cut]
	if len(evicted) < minEvict {
		return
	}

	summary := a.summarize(ctx, evicted)
	compacted := make([]llm.Message, 0, len(a.messages)-len(evicted)+1)
	compacted = append(compacted, a.messages[0]) // original system prompt
	compacted = append(compacted, llm.Message{
		Role:    llm.RoleSystem,
		Content: "Summary of earlier work in this session (older messages were compacted to save context):\n" + summary,
	})
	compacted = append(compacted, a.messages[cut:]...)
	a.messages = compacted
	a.lastPromptTokens = estimateTokens(a.messages)
	a.ui.Info(fmt.Sprintf("  [compacted %d earlier messages to stay within the context budget]", len(evicted)))
}

// summarize asks the model to compress a slice of older messages into notes.
// If the call fails or returns nothing, it falls back to the raw (bounded)
// transcript so information is reduced rather than lost.
func (a *Agent) summarize(ctx context.Context, msgs []llm.Message) string {
	transcript := renderTranscript(msgs)
	req := []llm.Message{
		{Role: llm.RoleSystem, Content: "You compress an AI coding agent's prior context. Summarize the conversation below into concise but complete notes an engineer could use to continue the task: the goal, key decisions, files created or modified, commands run and their outcomes, the current state, and the next steps. Be factual and terse; preserve concrete names, paths and errors. Do not invent anything."},
		{Role: llm.RoleUser, Content: transcript},
	}
	res, err := a.client.StreamPlain(ctx, req, llm.Handlers{})
	if err != nil || strings.TrimSpace(res.Content) == "" {
		return transcript
	}
	return strings.TrimSpace(res.Content)
}

// renderTranscript flattens messages into a compact plain-text transcript for
// summarization, bounding the total size by dropping the middle if needed.
func renderTranscript(msgs []llm.Message) string {
	var b strings.Builder
	for _, m := range msgs {
		switch m.Role {
		case llm.RoleUser:
			fmt.Fprintf(&b, "USER: %s\n", strings.TrimSpace(m.Content))
		case llm.RoleAssistant:
			if c := strings.TrimSpace(m.Content); c != "" {
				fmt.Fprintf(&b, "ASSISTANT: %s\n", c)
			}
			for _, tc := range m.ToolCalls {
				fmt.Fprintf(&b, "ASSISTANT called %s(%s)\n", tc.Function.Name, oneLine(tc.Function.Arguments, 200))
			}
		case llm.RoleTool:
			fmt.Fprintf(&b, "TOOL[%s] -> %s\n", m.Name, oneLine(m.Content, 400))
		case llm.RoleSystem:
			fmt.Fprintf(&b, "SYSTEM: %s\n", oneLine(m.Content, 200))
		}
	}
	s := b.String()
	if len(s) > maxTranscriptChars {
		half := maxTranscriptChars / 2
		s = s[:half] + "\n…(transcript truncated)…\n" + s[len(s)-half:]
	}
	return s
}

// trimForModel bounds a single tool result stored in history, keeping the head
// and tail so both the start and the final status/errors survive.
func trimForModel(s string) string {
	if len(s) <= maxToolResultChars {
		return s
	}
	half := maxToolResultChars / 2
	return s[:half] + "\n… (output truncated to fit context) …\n" + s[len(s)-half:]
}

// oneLine collapses s to a single line no longer than max runes.
func oneLine(s string, max int) string {
	s = strings.Join(strings.Fields(s), " ")
	r := []rune(s)
	if len(r) > max {
		return string(r[:max]) + "…"
	}
	return s
}
