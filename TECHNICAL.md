# Technical Overview

This document describes what *drea* is and how it is built. It reflects the
current source of this repository, which may drift from `README.md`.

---

## What

**drea** is a minimalist, terminal-based **AI coding-agent harness** written in
Go with **zero third-party dependencies** (standard library only, Go 1.19). It
drives any OpenAI-compatible model through a "reason–act" loop — send the
conversation, execute the tool calls the model requests, feed the results back,
repeat — in order to carry out a software task inside a *workspace root* a Linux
terminal.

The core promise is **a single, auditable static binary**:

- No Node.js, package managers, runtimes, or anything to install at runtime.
- `go.mod` contains no `require` directives.
- Every file/search/shell/git capability is confined to one workspace root.
- Mutating actions require interactive confirmation unless `--auto` is given.
- It is deliberately general-purpose: nothing about drea itself is compiled in.
  Project rules come from the workspace's own instruction files.

### Capabilities

**Tool use** — the model is offered a fixed set of tools: `read_file`,
`record_thoughts`, `list_dir`, `search`, `git_inspect` (all read-only) and
`write_file`, `edit_file`, `apply_patch`, `run_command`, `git_init`,
`git_commit`, `git_rollback` (all mutating). A `reply` pseudo-tool lets the
model send plain prose to the user. `record_thoughts` is context recording:
it persists the model's reasoning into the chat transcript so later turns can
build on it, without prompting for approval.

**Model reliability** — works with small/local models that lack reliable native
tool-calling:

- By default the endpoint is asked to constrain output to a strict JSON
  "tool_calls" schema (`response_format`; tune with `--json-format`, disable
  with `--no-json-mode`).
- If a turn contains text but no tool call, the JSON is parsed out of the text
  and executed as a tool call (`parseToolCallsFromText`).
- Rate limits and transient 5xx errors are retried with exponential backoff
  (honouring `Retry-After`).

**Safety net / autonomy** — supports running unattended without blanket trust:

- `run_command` is governed by an allow/deny **command policy** (Go RE2
  regexes; deny wins; a built-in deny list always blocks catastrophic commands,
  even under `--auto`).
- `git_*` tools give the agent first-class, confined checkpoints and rollback
  so bad changes are undone rather than hand-reverted.

**Goal-driven iteration** — the `--verify` command is the only objective
measure. With `--checkpoint` it becomes a *goal*: the workspace is committed
before the task, verify is measured, failures are fed back for self-correction
(bounded by `--verify-attempts`), and a task that turned a *passing* measure
into a *failing* one is rolled back — the attempt preserved on a
`drea/attempt-*` branch.

**Isolated attempts** — `--worktree` runs in a scratch git worktree on its own
branch; `--promote` merges back only when the measure passes and the repo can
fast-forward.

**Long-running robustness**:

- **Context compaction** — once the prompt exceeds `--context-tokens` (default
  96000), older history is summarized into a single note; the system prompt and
  recent turns stay verbatim; tool outputs are size-capped.
- **Stall detection** — an identical tool call with identical arguments 3 times
  in a row is flagged; 5 times aborts the task.
- **Resumable sessions** — per-workspace transcripts are saved after every step
  (atomic write, no secrets) and can be resumed with `--resume` / `/resume`.

**Evaluation** — `drea eval <dir>` runs every `*.json` task spec in a directory
and scores each by its own verify command, for measurable self-improvement.

**Interactive session** — without a task argument, drea runs a read-eval loop
with `/`-slash commands (`/help`, `/config`, `/model`, `/host`, `/auto`,
`/verify`, `/checkpoint`, `/cache-key`, `/usage`, `/policy`, `/reasoning`,
`/key`, `/multiline`, `/save`, `/resume`, `/reset`, `/tools`, `/exit`, plus
`/quit`
and bare `exit`/`quit`).

---

## How

### Execution entry points (`cmd/drea`)

- `main.go` — resolves configuration (`config.Defaults` + flags), sets up
  isolation, builds the LLM client / tool registry / agent, and dispatches to
  either a one-shot task or the interactive loop. `--version` is handled before
  flag parsing so it works even with no config. Ctrl-C cancels the in-flight
  task via a fresh per-task context without killing the process.
- `eval.go` — the `eval <dir>` subcommand: loads specs, runs each in its own
  auto-approved, non-persistent agent, and scores it with the spec's verify
  command.
- `worktree.go` — `--worktree` setup (`enterWorktree`), teardown
  (`finishWorktree`: remove unchanged, promote on evidence, else report), and
  the evidence-gated `promote`.

### Configuration resolution (`internal/config`)

Precedence (highest first): **flags → environment → settings file → API key
file → built-in defaults**. `config.Defaults` folds env + saved settings into a
`Config`; flags override afterwards; `Config.Normalize` validates and
canonicalises (no empty base URL/model/workdir, `--top-p` in `[0,1]`,
`--reasoning-effort` in `{low,medium,high}`, `--json-format`, `--debug` path,
policy patterns compile, `--reply-chain` positive). Built-in defaults are
`temperature` 1, `top_p` 0.95 and `reply-chain` 15. Auto-approval can be set
per category (`--auto-write`, `--auto-git`, `--auto-run`) or with the backward-
compatible `--auto` master switch. Persisted preferences live in
`~/.config/drea/settings.json`; the API key is kept separately in
`~/.config/drea/key` (both `0600`, never in transcripts). The chat endpoint is
`BaseURL + "/chat/completions"`.

### The reason–act loop (`internal/agent`)

`Agent.Run` is the core. For each step (bounded by `MaxSteps`):

1. `maybeCompact` squeezes history if needed; status bar updated.
2. `client.Stream` sends messages + tool specs, calling handlers live:
   `OnContent` prints assistant prose, `OnReasoning` streams the model's
   chain-of-thought into a "reasoning" `LiveView` (last 10 lines, wrapping long
   lines so the full reasoning stays visible; display-only, never stored), and
   `OnToolName`/`OnToolArgs` feed the **live file-write tail**
   (`internal/agent/live.go`) which renders the last 10 lines of a file as the
   model streams it.
3. The assistant message (content + tool calls) is appended to the transcript,
   kept in sync with any post-processing below.
4. If the model emitted **no tool call**, `parseToolCallsFromText` recovers a
   JSON-encoded call from plain text; else if JSON-mode, an empty/vacuous reply
   is nudged to produce a real action; else the loop calls `verify` — running
   the verify command and feeding a failure back for self-correction.
5. Reply pseudo-tools carry plain assistant prose. The agent separates them
   from real tools. If every tool call is a reply, or if a reply is the **last**
   tool call in a mixed batch, the turn is a natural stopping point. In
   interactive mode `shouldContinue` classifies it: the harness calls
   `Client.Stream` with a dedicated internal `classify_reply` tool, so the
   decision is emitted as structured JSON even on endpoints that force JSON
   output. The tool takes an `action` of either `CONTINUE_AUTONOMOUSLY` (the
   reply is only a status update or statement of intent) or `PAUSE_FOR_USER`
   (it asks a question, offers a choice, or needs input). The classifier prompt
   includes the original user task and the full reply, with concrete examples
   to reduce ambiguity. On `CONTINUE_AUTONOMOUSLY` a system message tells the
   model to continue without asking for permission; on `PAUSE_FOR_USER` the
   harness returns to the interactive prompt. A configurable `--reply-chain`
   limit (default 15) prevents infinite auto-continuation.
6. Real tools from the batch are `dispatch`ed: looked up, summarized in the UI,
   diff-previewed if a file edit, run subject to approval (mutating tools
   prompt unless their category is auto-approved; `run_command` additionally
   goes through the command policy — deny always wins, allow skips the prompt),
   results appended as `tool` messages.
7. A reply followed by real actions resets the reply-continuation counter; a
   reply that ends the batch is classified after the actions complete.
8. Stall detection inspects the batch; a nudge or abort decides the next move.

Token usage reported by the endpoint is accumulated and printed after each task
(`/usage` for the session). Nothing is estimated.

### Model client and streaming (`internal/llm`)

`Client.Stream` POSTs a chat-completions request with `stream: true` and parses
SSE frames (`data:` lines, tolerant of comments/blanks/malformed frames, until
`[DONE]`). Content deltas and index-addressed tool-call fragments are merged.
`Client.StreamPlain` is the same machinery without the JSON schema, used for
internal prose prompts (reply classification and context compaction) that must
receive plain text even when the main loop runs in JSON mode.

In **JSON mode**, tool calls may arrive as constrained JSON in `content`; they
are reconstructed incrementally into native `ToolCall`s
(`parseJSONModeToolCalls`). Reply pseudo-tools are kept as `ToolCall`s in the
result so the agent can recognise reply-only and reply-last turns; their prose
is still streamed live so the user sees it as it arrives. Real tool-call JSON
is withheld until the stream ends so partial JSON does not scroll by. Transient
errors retry with exponential backoff. `--debug` dumps raw request/response
traffic to a file. A non-standard `prompt_cache_key` is forwarded when
configured, and cache hit/miss/status is read from common response headers
(`X-Cache-Status`, `X-Prompt-Cache-Hit`, etc.) and usage fields
(`prompt_cache_hit_tokens`). The JSON schema constrains output to a non-empty
array of `{name, arguments}` items, preventing a vacuous `[]` from being
mistaken for "done".

### Tools and confinement (`internal/tool`)

A `Tool` exposes a name, description, JSON-schema, `Mutating` flag, a
`Summary`, and `Run`. `Registry` holds all tools confined to `Workdir`.
`resolve` defends the workspace root: paths that escape lexically are rejected,
and symlinks are resolved (deepest existing ancestor, so about-to-be-created
files still check against the real root) and re-checked — defeating both
`../` traversal and symlink escapes.

- `run_command` uses `bash -c` from the workspace root, in its own process
  group (killed as a tree on timeout/cancel), bounded by a wall-clock timeout
  (default 120s, max 15m) and a 64 KiB combined-output cap. The hardened
  runner is shared with the verification loop (`tool.RunShell`).
- The **patch engine** (`internal/patch`, shared by `edit_file`/`apply_patch`
  and the diff preview) is multi-hunk and all-or-nothing: edits apply in memory
  and write only if every one succeeds. Exact match first, then
  whitespace-tolerant fallbacks (ignore trailing whitespace, then leading
  indentation); a fuzzy match is accepted only when **unique** — never guessed.
  File permissions are preserved.
- `search` is a pure-Go, regexp-based content search.
- Git tools run `git -C <root> …` with an explicit argument vector (never a
  shell string); refs/paths are validated and confined. It is named
  `git_inspect`, not `git`, so the model is not nudged toward `run_command`.

### Git bookkeeping (`internal/vcs` vs `internal/tool`)

`internal/vcs` is the harness's *own* git usage (checks, checkpoints,
rollback, worktrees), distinct from the model-facing git *tools*; both go
through git with explicit argv. Notable behaviours:

- `Checkpoint` commits everything, or returns the current `HEAD` when clean,
  so callers always get a rollback-able ref. `IdentityArgs` supplies a
  `github.com/dreaagent/drea` identity **for that commit only** when none is
  configured, never touching your git config.
- `Rollback` preserves the attempt on a branch first, then restores — if the
  attempt cannot be preserved it refuses to destroy anything.
- `Restore` does `reset --hard` **plus** `clean -fd`, so a real undo also
  removes files the task added.
- `AddWorktree` creates the scratch checkout; `Worktree.Changed` counts
  uncommitted changes **or** new commits (so committed work is never mistaken
  for "untouched" and deleted). `Remove` is deliberately unforceful
  (`remove` + `branch -d`).

The **verification/rollback decision** is in `internal/agent/verify.go`:
`beginTask` records the checkpoint and the *baseline* measure; `verify` iterates
up to `VerifyAttempts` self-correction rounds; `offerRollback` undoes the task
**only** when the baseline was `MeasurePassing` and the final measure is
`MeasureFailing` — the one regression judged objectively. The outcome is
exposed via `Agent.Objective()` for worktree promotion.

### Project instructions (`internal/conventions`)

On startup the system prompt appends the workspace's own instruction files,
your project's rules coming from the project rather than the harness.
Recognised most-specific-first (case-insensitive): `AGENTS.md`, `CLAUDE.md`,
`CONVENTIONS.md`, `CONTRIBUTING.md`, `README.md`, `README`. Loading is bounded
(8 KiB/file, 16 KiB total, truncated on line boundaries), only regular files
directly in the root are read, and symlinks are ignored (an `Lstat`), so a repo
can't point the agent at an arbitrary file.

### UI (`internal/ui`)

Dependency-free terminal output, all of it TTY-aware and no-op when stdout is
not a terminal:

- ANSI colour (auto-disabled off a TTY or with `NO_COLOR`), a startup `Banner`
  showing model/workdir/auto-approve categories, and dim/green/red/yellow status
  `Println` helpers (`Info`, `Done`, `Warn`, `Error`).
- A persistent bottom **status bar** (`SetStatus`/`ShowStatus`) rendering the
  context-used/budget bar and the workdir; `HideStatus`/`ensureLineStart` keep
  scrolling output from colliding with it.
- A "thinking" spinner (`StartThinking`/`StopThinking`) driven by a goroutine,
  guarded by mutex + an atomic active flag. It stays up only while the model
  is producing neither content nor reasoning; the first `reasoning_content`
  delta swaps it for the reasoning `LiveView`.
- A `LiveView` for the rolling file-write tail: redraws the last `maxLines` in
  place by emitting an exact cursor-up count (lines are preserved to one row
  each, so the count is never off by wrapping). No-op off a TTY; the final
  result is printed elsewhere.
- `Assistant` renders streamed model text through a `streamFilter` that strips
  escape/control sequences while preserving newlines/tabs, **carrying its
  in-escape state across chunk boundaries** so a sequence split mid-stream is
  still neutralised. Tool calls/summaries/results/diffs are shown through
  `sanitizeLine`, which also expands tabs to fixed stops.
- Interactive `Prompt`/`ReadInput` (single-line, or multi-line where Enter
  adds a line and Ctrl-D sends) and a `Confirm` prompt that defaults to "no"
  (`[y/N]`); the input helpers are used by the interactive loop and approval
  flow in `cmd/drea` and `internal/agent`.

### Evaluation (`internal/eval`)

Reads every `*.json` spec (name/prompt/workdir/setup/verify), runs each task in
a fresh auto-approved, ephemeral agent, then scores with the spec's verify
command. Prints a PASS/FAIL report and exits non-zero unless all pass, so it
can gate self-improvement in CI or a loop.

---

## Design invariants (safety properties)

- **No third-party dependencies; Go 1.19.** Standard library only; `go.mod` has
  no `require`s.
- **Workspace confinement.** No file/search/shell/git path may escape the root —
  lexical and symlink checks.
- **Approval by default.** Mutations prompt unless `--auto`; `run_command`'
  policy (deny > allow > ask) layers on top; built-in denies can't be
  overridden.
- **Never guess.** Ambiguous edits and ambiguous inputs fail with guidance
  rather than acting.
- **Only labelled evidence decides.** Verify-passing-then-failing gates the one
  automatic rollback; worktree promotion requires a passing measure and a clean
  fast-forward.
- **Cleanup never loses work.** Modified worktrees are kept and reported;
  rollback preserves attempts on branches; unforceful deletes.
- **Untrusted input is sanitized.** Provider text is stripped of terminal
  control codes before display; command output and transcripts are size-capped;
  no API key ever reaches a transcript or settings.
- **Fail open that favours safety.** Persistence failures are non-fatal;
  missing conventions / sessions are normal.

---

## Verification

The repository's own makefile / AGENTS.md verify steps are:
`gofmt -l .`, `go vet ./...`, `go test ./...`, `go test -race ./...`,
`go build ./...`. In this workspace `gofmt`, `go vet`, `go test`, and
`go test -race` all pass.