// Package agent implements the core reason-act loop: send the conversation and
// tool schemas to the model, execute the tool calls it requests (subject to
// approval), feed the results back, and repeat until the task is complete.
package agent

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"strings"
	"time"

	"github.com/dreaagent/drea/internal/config"
	"github.com/dreaagent/drea/internal/diff"
	"github.com/dreaagent/drea/internal/llm"
	"github.com/dreaagent/drea/internal/patch"
	"github.com/dreaagent/drea/internal/policy"
	"github.com/dreaagent/drea/internal/session"
	"github.com/dreaagent/drea/internal/tool"
	"github.com/dreaagent/drea/internal/ui"
)

// Agent ties together the model client, tool registry, UI and conversation
// state. It is safe to reuse across multiple user turns (interactive mode).
type Agent struct {
	cfg      *config.Config
	client   *llm.Client
	tools    *tool.Registry
	ui       *ui.UI
	policy   *policy.Policy
	messages []llm.Message

	// lastPromptTokens is the prompt-token count reported by the endpoint for
	// the most recent request (0 when unknown); it drives compaction.
	lastPromptTokens int
	// verifyRounds counts how many times the verification command has failed
	// and been fed back within the current Run.
	verifyRounds int

	// checkpoint is the commit taken before the current task, and baseline is
	// what the verification command reported at that point. Together they let
	// a task that regressed the goal be recognised and undone.
	checkpoint string
	baseline   Measure
	// objective is the latest state of the verification command, exposed so a
	// caller can tell work that met the goal from work that merely finished.
	objective Measure

	stall stallDetector
	// task and total accumulate reported token usage for the current task and
	// for the session, so a runaway loop is visible in what it costs.
	task  Usage
	total Usage

	// interactive is true when the agent is running inside the read-eval loop,
	// where reply continuation makes sense. One-shot tasks do not enable it.
	interactive bool
	// replyCount is how many consecutive assistant replies have been
	// auto-continued in the current Run; it is reset for each new user task.
	replyCount int
}

// Usage is cumulative token accounting as reported by the endpoint.
type Usage struct {
	PromptTokens     int
	CompletionTokens int
	Requests         int
	// PromptCacheHitTokens is the cumulative prompt tokens served from cache.
	PromptCacheHitTokens int
	// PromptCacheMissTokens is the cumulative prompt tokens not served from cache.
	PromptCacheMissTokens int
}

func (u Usage) add(v llm.Usage) Usage {
	u.PromptTokens += v.PromptTokens
	u.CompletionTokens += v.CompletionTokens
	u.PromptCacheHitTokens += v.PromptCacheHitTokens
	u.PromptCacheMissTokens += v.PromptCacheMissTokens
	u.Requests++
	return u
}

// String renders usage for display. An endpoint that reports no usage yields
// zeroes, which is honest: nothing is estimated here.
func (u Usage) String() string {
	s := fmt.Sprintf("%d prompt + %d completion tokens over %d request(s)",
		u.PromptTokens, u.CompletionTokens, u.Requests)
	if u.PromptCacheHitTokens+u.PromptCacheMissTokens > 0 {
		s += fmt.Sprintf(" (%d cached / %d uncached)", u.PromptCacheHitTokens, u.PromptCacheMissTokens)
	}
	return s
}

// Usage reports token accounting for the session so far.
func (a *Agent) Usage() Usage { return a.total }

// SetInteractive marks the agent as running inside the interactive read-eval
// loop, enabling reply continuation. One-shot tasks do not call this.
func (a *Agent) SetInteractive(v bool) { a.interactive = v }

// New constructs an agent with a seeded system prompt.
func New(cfg *config.Config, client *llm.Client, tools *tool.Registry, u *ui.UI) *Agent {
	// Config.Normalize already validated the patterns; on the off chance of an
	// error here, fall back to the built-in deny list only rather than failing.
	pol, err := policy.New(cfg.AllowCommands, cfg.DenyCommands)
	if err != nil {
		pol, _ = policy.New(nil, nil)
	}
	return &Agent{
		cfg:    cfg,
		client: client,
		tools:  tools,
		ui:     u,
		policy: pol,
		messages: []llm.Message{
			{Role: llm.RoleSystem, Content: systemPrompt(cfg.Workdir, tools.Names())},
		},
	}
}

// Run processes a single user instruction, looping through model turns and
// tool executions until the model stops requesting tools or MaxSteps is hit.
func (a *Agent) Run(ctx context.Context, userInput string) error {
	a.messages = append(a.messages, llm.Message{Role: llm.RoleUser, Content: userInput})
	a.verifyRounds = 0
	a.task = Usage{}
	a.stall.reset()
	a.replyCount = 0
	a.persist()
	a.beginTask(ctx, userInput)
	defer a.reportUsage()

	for step := 0; step < a.cfg.MaxSteps; step++ {
		a.maybeCompact(ctx)
		a.updateStatus()
		printedHeader := false
		contentOpen := false
		ensureNL := func() {
			if contentOpen {
				a.ui.Assistant("\n")
				contentOpen = false
			}
		}
		live := newLiveWrites(a.ui, a.tools, ensureNL)
		reasoning := newReasoningView(a.ui, ensureNL)
		a.ui.StartThinking()
		res, err := a.client.Stream(ctx, a.messages, a.tools.Specs(), llm.Handlers{
			OnReasoning: reasoning.onReasoning,
			OnContent: func(delta string) {
				reasoning.close()
				a.ui.StopThinking()
				if !printedHeader {
					a.ui.AssistantHeader()
					printedHeader = true
				}
				a.ui.Assistant(delta)
				contentOpen = !strings.HasSuffix(delta, "\n")
			},
			OnToolName: live.onName,
			OnToolArgs: live.onArgs,
			OnRetry: func(attempt int, delay time.Duration, err error) {
				ensureNL()
				a.ui.Warn(fmt.Sprintf("│  %s; retrying in %s (attempt %d)",
					shortErr(err), delay.Round(100*time.Millisecond), attempt))
			},
		})
		live.close()
		reasoning.close()
		a.ui.StopThinking()
		if err != nil {
			return err
		}
		ensureNL()
		if res.Usage.PromptTokens > 0 {
			a.lastPromptTokens = res.Usage.PromptTokens
		}
		a.task, a.total = a.task.add(res.Usage), a.total.add(res.Usage)
		a.updateStatus()
		if res.CacheStatus != "" {
			a.ui.Info(fmt.Sprintf("│  cache %s (%s)", res.CacheStatus, res.CacheDetail))
		}

		// Record the assistant turn (content and/or tool calls).
		// Keep the recorded message in sync with any post-processing below
		// (text-recovered tool calls, reply pseudo-tools, etc.) so the transcript
		// always matches what the agent actually dispatches.
		a.messages = append(a.messages, llm.Message{
			Role:      llm.RoleAssistant,
			Content:   res.Content,
			ToolCalls: res.ToolCalls,
		})
		recorded := &a.messages[len(a.messages)-1]

		if len(res.ToolCalls) == 0 {
			// A model without reliable native tool-calling may have printed tool
			// calls as JSON text instead of emitting a tool_calls block.
			// Recover them so the turn is executed rather than mistaken for done.
			if tcs, ok := parseToolCallsFromText(res.Content); ok {
				res.ToolCalls = tcs
				res.Content = ""
				recorded.ToolCalls = res.ToolCalls
				recorded.Content = res.Content
			}
		}
		// JSON-mode "reply" pseudo-tools were already converted to content by
		// the client; make sure the assistant message records that content.
		if a.client.JSONMode() && len(res.ToolCalls) == 0 && res.Content != "" {
			recorded.Content = res.Content
		}

		if len(res.ToolCalls) == 0 {
			// The model believes the task is done: run the verification
			// command (if any) and feed a failure back for self-correction.
			if fb, retry := a.verify(ctx); retry {
				a.messages = append(a.messages, llm.Message{Role: llm.RoleUser, Content: fb})
				a.persist()
				continue
			}
			// In JSON mode an empty assistant turn is not a meaningful final
			// reply; it usually means the model emitted a vacuous reply tool.
			// Nudge it to produce a real action rather than returning silence.
			if a.client.JSONMode() && recorded.Content == "" && len(recorded.ToolCalls) == 0 {
				a.ui.Warn("│  model returned an empty reply; asking for a real action.")
				a.messages = append(a.messages, llm.Message{Role: llm.RoleUser, Content: "Your last response was empty. Please continue the task by issuing a real tool call or a non-empty reply."})
				a.persist()
				continue
			}
			a.persist()
			return nil // task turn complete
		}

		// Separate reply pseudo-tools from real tools. Reply tools carry plain
		// assistant prose; real tools mutate state or run commands. The model may
		// emit a reply alone, or mix it with real tools (usually as a status
		// update before or after actions). When a reply appears as the last tool
		// call in the turn, it is a natural stopping point where we can ask the
		// model whether it wants to continue or wait for user input.
		replyMsg := joinReplyMessages(res.ToolCalls)
		lastIsReply := len(res.ToolCalls) > 0 && llm.IsReplyCall(res.ToolCalls[len(res.ToolCalls)-1])
		allReply := allReplies(res.ToolCalls)
		if replyMsg == "" && allReply {
			// Vacuous reply tools do not count as a meaningful turn; nudge for a
			// real action instead of returning to the prompt.
			a.ui.Warn("│  model returned an empty reply; asking for a real action.")
			a.messages = append(a.messages, llm.Message{Role: llm.RoleUser, Content: "Your last response was empty. Please continue the task by issuing a real tool call or a non-empty reply."})
			a.persist()
			continue
		}
		if replyMsg != "" {
			if allReply {
				// JSON-mode reply tools leave raw JSON in res.Content; replace it
				// with the extracted prose so the model does not see its own schema.
				res.Content = replyMsg
			} else if res.Content != "" {
				res.Content += "\n" + replyMsg
			} else {
				res.Content = replyMsg
			}
			recorded.Content = res.Content
			if !res.ReplyStreamed {
				if !printedHeader {
					a.ui.AssistantHeader()
					printedHeader = true
				}
				a.ui.Assistant(replyMsg)
				contentOpen = !strings.HasSuffix(replyMsg, "\n")
			}
		}
		filtered := res.ToolCalls[:0]
		for _, tc := range res.ToolCalls {
			if !llm.IsReplyCall(tc) {
				filtered = append(filtered, tc)
			}
		}
		res.ToolCalls = filtered
		recorded.ToolCalls = res.ToolCalls

		nudge, abort := a.stall.observe(res.ToolCalls)
		for _, tc := range res.ToolCalls {
			result := a.dispatch(ctx, tc)
			a.messages = append(a.messages, llm.Message{
				Role:       llm.RoleTool,
				ToolCallID: tc.ID,
				Name:       tc.Function.Name,
				Content:    trimForModel(result),
			})
		}
		// If the model produced reasoning but did not record its thoughts, ask it
		// to do so now. This persists the reasoning into the conversation context
		// so later turns can build on it.
		if res.ReasoningContent != "" && !hasToolCall(res.ToolCalls, "record_thoughts") {
			if err := a.recordThoughtsFollowUp(ctx, res.ReasoningContent); err != nil {
				a.ui.Warn("│  could not record thoughts: " + shortErr(err))
			}
		}
		switch {
		case abort:
			a.ui.Warn("│  the same action is being repeated with no progress; stopping this task.")
			a.persist()
			return nil
		case nudge:
			a.ui.Warn("│  repeated identical action; prompting for a different approach.")
			a.messages = append(a.messages, llm.Message{Role: llm.RoleUser, Content: a.stall.stallMessage()})
		}

		// If the reply was the last tool call, the model has finished its
		// actions and is now addressing the user. In interactive mode, ask it
		// whether this is just a status update (continue automatically) or a
		// real request for user input (pause).
		if a.interactive && replyMsg != "" && lastIsReply {
			if a.replyCount >= a.cfg.ReplyChain {
				a.ui.Warn(fmt.Sprintf("│  assistant has continued automatically %d times; pausing for your input", a.replyCount))
			} else {
				cont, err := a.shouldContinue(ctx, userInput, replyMsg)
				if err != nil {
					a.ui.Warn("│  could not decide whether to continue: " + shortErr(err))
				} else if cont {
					a.messages = append(a.messages, llm.Message{Role: llm.RoleSystem, Content: "Continue your work; do not ask the user for permission before proceeding."})
					a.replyCount++
					a.persist()
					continue
				}
				a.ui.Info("│  assistant reply did not look like a status update; waiting for your input")
			}
		}

		// A pure reply turn with no real tools may be the model declaring the
		// task done: run the verify command and feed failures back for self-
		// correction before returning to the user.
		if allReply {
			if fb, retry := a.verify(ctx); retry {
				a.messages = append(a.messages, llm.Message{Role: llm.RoleUser, Content: fb})
				a.persist()
				continue
			}
			a.persist()
			return nil
		}

		// A reply followed by real actions is not a stopping point; reset the
		// reply chain because the next assistant reply starts fresh.
		a.replyCount = 0
		a.persist()
	}

	a.ui.Warn(fmt.Sprintf("reached step limit (%d); stopping.", a.cfg.MaxSteps))
	return nil
}

// dispatch resolves and runs a single tool call, handling approval and
// returning the text to feed back to the model (errors are returned as text so
// the model can recover rather than aborting the whole run).
func (a *Agent) dispatch(ctx context.Context, tc llm.ToolCall) string {
	t, ok := a.tools.Get(tc.Function.Name)
	if !ok {
		a.ui.ToolCall(tc.Function.Name, "(unknown)")
		return toolError(fmt.Errorf("unknown tool %q", tc.Function.Name))
	}

	args := json.RawMessage(tc.Function.Arguments)
	summary := t.Summary(args)
	a.ui.ToolCall(t.Name(), summary)

	// For file edits, preview the change as a diff (before approval).
	if d, added, removed, ok := a.prospectiveDiff(t.Name(), args); ok {
		a.ui.Diff(d)
		a.ui.Info(fmt.Sprintf("│  (+%d/-%d)", added, removed))
	}

	requireApproval := t.Mutating() && !autoApproved(a.cfg, t.Name())
	// The command policy governs run_command specifically: a denied command is
	// refused regardless of auto-approve; an allowed one runs without a prompt.
	if t.Name() == "run_command" {
		switch a.policy.Decide(commandArg(args)) {
		case policy.Deny:
			a.ui.Warn("│  blocked by command policy (deny list)")
			return "This command is blocked by the configured command policy and was not run. Choose a safe alternative or ask the user to adjust the policy."
		case policy.Allow:
			requireApproval = false
			a.ui.Info("│  (auto-approved by command policy)")
		}
	}
	if requireApproval {
		if !a.ui.Confirm("│  approve this action?") {
			a.ui.Warn("│  denied by user")
			return "The user denied this action. Consider a different approach or ask what to do."
		}
	}

	out, err := t.Run(ctx, args)
	if err != nil {
		a.ui.ToolResult(err.Error(), true)
		return toolError(err)
	}
	a.ui.ToolResult(out, false)
	return out
}

// allReplies reports whether every tool call is a non-empty "reply" pseudo-tool.
// Vacuous reply tools are ignored so a confused model cannot end the turn with
// an empty message.
func allReplies(tcs []llm.ToolCall) bool {
	if len(tcs) == 0 {
		return false
	}
	hasReply := false
	for _, tc := range tcs {
		if !llm.IsReplyCall(tc) {
			return false
		}
		if llm.ReplyMessage(tc) != "" {
			hasReply = true
		}
	}
	return hasReply
}

// joinReplyMessages concatenates the messages from reply pseudo-tools,
// preserving order and dropping empty ones.
func joinReplyMessages(tcs []llm.ToolCall) string {
	var parts []string
	for _, tc := range tcs {
		if msg := llm.ReplyMessage(tc); msg != "" {
			parts = append(parts, msg)
		}
	}
	return strings.Join(parts, "\n")
}

// reportUsage prints what the task cost, when the endpoint reported anything.
func (a *Agent) reportUsage() {
	if a.task.PromptTokens+a.task.CompletionTokens == 0 {
		return
	}
	a.ui.Info("╰─ usage: " + a.task.String())
}

// Reset clears the conversation history, keeping only the system prompt.
func (a *Agent) Reset() {
	a.messages = a.messages[:1]
	a.lastPromptTokens = 0
	a.persist()
	a.updateStatus()
}

// updateStatus refreshes the bottom status bar with the current context usage.
func (a *Agent) updateStatus() {
	used := a.lastPromptTokens
	if used == 0 {
		used = estimateTokens(a.messages)
	}
	a.ui.SetStatus(used, a.cfg.ContextTokens)
	a.ui.ShowStatus()
}

// persist saves the current transcript for this workspace when persistence is
// enabled. It never writes the API key (only messages are stored) and failures
// are non-fatal — persistence must never break an in-progress task.
func (a *Agent) persist() {
	if !a.cfg.Persist {
		return
	}
	_ = session.Save(session.Session{
		Workdir:  a.cfg.Workdir,
		Model:    a.cfg.Model,
		Messages: a.messages,
	})
}

// Restore replaces the conversation with a previously saved transcript so a
// session can be resumed after exit or a crash. An empty transcript is ignored.
func (a *Agent) Restore(msgs []llm.Message) bool {
	if len(msgs) == 0 {
		return false
	}
	a.messages = append([]llm.Message(nil), msgs...)
	a.lastPromptTokens = estimateTokens(a.messages)
	return true
}

// MessageCount reports how many messages are in the current transcript.
func (a *Agent) MessageCount() int { return len(a.messages) }

// prospectiveDiff computes the diff a write_file/edit_file/apply_patch call
// would produce, so it can be previewed before the change is applied. Edits are
// replayed through the same patch engine the tools use, so the preview cannot
// diverge from what is written. ok is false for other tools or when the change
// cannot be determined (e.g. an ambiguous edit, which the tool itself rejects).
func (a *Agent) prospectiveDiff(name string, args json.RawMessage) (text string, added, removed int, ok bool) {
	var whole string // full content for write_file
	path, edits, isPatch := tool.ParseEdits(name, args)
	if !isPatch {
		var a2 struct {
			Path    string `json:"path"`
			Content string `json:"content"`
		}
		if name != "write_file" || json.Unmarshal(args, &a2) != nil || a2.Path == "" {
			return "", 0, 0, false
		}
		path, whole = a2.Path, a2.Content
	}
	p, err := a.tools.ResolvePath(path)
	if err != nil {
		return "", 0, 0, false
	}
	oldData, _ := os.ReadFile(p) // missing file => empty (a creation)
	oldText := string(oldData)

	newText := whole
	if isPatch {
		res, err := patch.Apply(oldText, edits)
		if err != nil {
			return "", 0, 0, false
		}
		newText = res.Text
	}
	if newText == oldText {
		return "", 0, 0, false
	}
	// Bound the O(n*m) line diff: skip the preview for very large files.
	const maxDiffBytes = 256 * 1024
	if len(oldText) > maxDiffBytes || len(newText) > maxDiffBytes {
		return "", 0, 0, false
	}
	added, removed = diff.Stat(oldText, newText)
	return diff.Unified(oldText, newText, 3), added, removed, true
}

// shortErr collapses an error to a single, length-bounded line so a retry
// notice stays readable even when the endpoint returns a large HTML body.
func shortErr(err error) string {
	s := strings.TrimSpace(err.Error())
	if i := strings.IndexAny(s, "\r\n"); i >= 0 {
		s = s[:i]
	}
	if i := strings.Index(s, "<"); i >= 0 {
		s = strings.TrimSpace(s[:i])
	}
	s = stripControl(s)
	const max = 120
	if len(s) > max {
		s = s[:max] + "…"
	}
	return s
}

// stripControl removes escape and other control characters from a single-line
// string so a provider-supplied message cannot corrupt the terminal.
func stripControl(s string) string {
	var b strings.Builder
	inEsc, inCSI := false, false
	for _, r := range s {
		switch {
		case inCSI:
			if r >= 0x40 && r <= 0x7e {
				inCSI = false
			}
		case inEsc:
			inEsc = false
			if r == '[' {
				inCSI = true
			}
		case r == 0x1b:
			inEsc = true
		case r < 0x20 || r == 0x7f:
			// drop control characters.
		default:
			b.WriteRune(r)
		}
	}
	return b.String()
}

// commandArg extracts the "command" field from a run_command tool call's
// arguments so the policy can classify it. It returns "" when absent.
func commandArg(args json.RawMessage) string {
	var a struct {
		Command string `json:"command"`
	}
	_ = json.Unmarshal(args, &a)
	return a.Command
}

// replyClassifierTool lets the model emit a structured classification even
// when the endpoint forces JSON output. It is only used for the internal reply
// continuation check, not advertised as a workspace tool.
var replyClassifierTool = llm.Tool{
	Type: "function",
	Function: llm.FunctionSpec{
		Name:        "classify_reply",
		Description: "Classify the assistant's latest reply to the user.",
		Parameters: json.RawMessage(`{
		  "type":"object",
		  "properties":{
		    "action":{"type":"string","enum":["CONTINUE_AUTONOMOUSLY","PAUSE_FOR_USER"],"description":"CONTINUE_AUTONOMOUSLY if the reply is only a status update, progress report, or statement of intent to keep working. PAUSE_FOR_USER if the reply asks a question, offers a choice, requests confirmation, says it is blocked, or needs more information before proceeding."}
		  },
		  "required":["action"],
		  "additionalProperties":false
		}`),
	},
}

// replyClassifierPrompt asks the model to classify its latest reply using the
// classify_reply tool. It includes the user's task so the classification is not
// made in a vacuum, and gives concrete examples so small models are less likely
// to misread statements of intent as requests for permission.
const replyClassifierPrompt = `You are classifying your latest reply to the user.

The user's task: %s
Your reply: "%s"

Use the classify_reply tool to indicate:
- CONTINUE_AUTONOMOUSLY: the reply is only a status update, progress report, or statement of intent to keep working. Examples: "I'll start by reading the files.", "Working on it.", "GTK3 is available; let me create the GUI calculator.", "I will now run the tests.", "X is done, building Y next."
- PAUSE_FOR_USER: the reply asks the user a question, offers a choice, requests confirmation, says it is blocked, or needs more information. Examples: "Should I use A or B?", "Do you want me to overwrite the file?", "What color should the button be?", "I cannot proceed without an API key."

Do not reply to the user; only call classify_reply.`

// shouldContinue asks the model whether its latest reply is a status update
// (continue automatically) or a request for user input (pause). It returns
// false on any ambiguity or error so the harness defaults to pausing.
func (a *Agent) shouldContinue(ctx context.Context, task, reply string) (bool, error) {
	taskSnippet := oneLine(task, 200)
	replySnippet := oneLine(reply, 280)
	req := []llm.Message{
		{Role: llm.RoleSystem, Content: fmt.Sprintf(replyClassifierPrompt, taskSnippet, replySnippet)},
	}
	a.ui.StartThinking()
	res, err := a.client.Stream(ctx, req, []llm.Tool{replyClassifierTool}, llm.Handlers{})
	a.ui.StopThinking()
	if err != nil {
		return false, err
	}
	a.task, a.total = a.task.add(res.Usage), a.total.add(res.Usage)
	return classifierAction(res) == "CONTINUE_AUTONOMOUSLY", nil
}

// hasToolCall reports whether calls contains a tool with the given name.
func hasToolCall(calls []llm.ToolCall, name string) bool {
	for _, tc := range calls {
		if tc.Function.Name == name {
			return true
		}
	}
	return false
}

// recordThoughtsFollowUpPrompt asks the model to turn its reasoning into a
// record_thoughts tool call so the reasoning is preserved in context.
const recordThoughtsFollowUpPrompt = `You just produced reasoning while working on the user's task:

%s

Use the record_thoughts tool now to summarize your key findings, intentions, and next steps. Do not reply to the user and do not call any other tool.`

// recordThoughtsFollowUp asks the model to record the reasoning it just
// produced. It is a safety net for when the model did not emit record_thoughts
// on its own. The resulting tool call and its result are appended to the
// transcript.
func (a *Agent) recordThoughtsFollowUp(ctx context.Context, reasoning string) error {
	if strings.TrimSpace(reasoning) == "" {
		return nil
	}
	req := []llm.Message{
		{Role: llm.RoleSystem, Content: fmt.Sprintf(recordThoughtsFollowUpPrompt, reasoning)},
	}
	a.ui.StartThinking()
	res, err := a.client.Stream(ctx, req, []llm.Tool{tool.RecordThoughtsTool}, llm.Handlers{})
	a.ui.StopThinking()
	if err != nil {
		return err
	}
	a.task, a.total = a.task.add(res.Usage), a.total.add(res.Usage)

	for _, tc := range res.ToolCalls {
		if tc.Function.Name != "record_thoughts" {
			continue
		}
		a.messages = append(a.messages, llm.Message{
			Role:      llm.RoleAssistant,
			Content:   res.Content,
			ToolCalls: []llm.ToolCall{tc},
		})
		result := a.dispatch(ctx, tc)
		a.messages = append(a.messages, llm.Message{
			Role:       llm.RoleTool,
			ToolCallID: tc.ID,
			Name:       tc.Function.Name,
			Content:    trimForModel(result),
		})
		a.persist()
		return nil
	}
	return nil
}

// classifierAction extracts the classification from a tool call, falling back
// to reply pseudo-tools or raw JSON-mode content so different endpoint styles
// are tolerated.
func classifierAction(res *llm.Result) string {
	actionFrom := func(tc llm.ToolCall) (string, bool) {
		if tc.Function.Name == "classify_reply" {
			var a struct {
				Action string `json:"action"`
			}
			if err := json.Unmarshal([]byte(tc.Function.Arguments), &a); err == nil {
				return strings.ToUpper(strings.TrimSpace(a.Action)), true
			}
		}
		if llm.IsReplyCall(tc) {
			msg := strings.ToUpper(llm.ReplyMessage(tc))
			if strings.Contains(msg, "CONTINUE_AUTONOMOUSLY") {
				return "CONTINUE_AUTONOMOUSLY", true
			}
			if strings.Contains(msg, "PAUSE_FOR_USER") {
				return "PAUSE_FOR_USER", true
			}
		}
		return "", false
	}

	for _, tc := range res.ToolCalls {
		if action, ok := actionFrom(tc); ok {
			return action
		}
	}
	if tcs, ok := parseToolCallsFromText(res.Content); ok {
		for _, tc := range tcs {
			if action, ok := actionFrom(tc); ok {
				return action
			}
		}
	}
	return ""
}

// autoApproved reports whether a tool should run without a prompt based on the
// per-category auto-approve flags. AutoApprove is the backward-compatible
// master switch that enables all categories.
func autoApproved(cfg *config.Config, name string) bool {
	if cfg.AutoApprove {
		return true
	}
	switch name {
	case "write_file", "edit_file", "apply_patch":
		return cfg.AutoWrite
	case "git_init", "git_commit", "git_rollback":
		return cfg.AutoGit
	case "run_command":
		return cfg.AutoRun
	}
	return false
}

func toolError(err error) string {
	if err == nil {
		err = errors.New("unknown error")
	}
	return "ERROR: " + err.Error()
}
