# drea

Official website: <https://drea.run>  
Source: <https://github.com/dreaagent/drea>

A minimalist, terminal-based AI **agent harness**, written in Go with **zero
third-party dependencies** (Go standard library only). It drives any
OpenAI-compatible model through a reason–act loop with a small, auditable set
of tools, enough to build a software project locally from start to finish.

It deliberately avoids Node.js, package managers, and external runtimes. The
result is a single static binary with a tiny attack surface: no `node_modules`,
no supply chain, nothing to install at runtime.

## Why this design

- **No dependencies.** HTTP, TLS, JSON and SSE streaming all come from Go's
  standard library. `go.mod` lists no `require`s.
- **Single static binary.** Nothing to install on the target machine; nothing
  interpreted.
- **Confined by default.** Every file/search/command tool is restricted to a
  single workspace root; path traversal (`../`) outside it is rejected.
- **Approval by default.** Commands and file writes require interactive
  confirmation unless you pass `--auto`. Writes and edits are shown as a
  coloured diff before they are applied.
- **Live file writes.** When the model writes or edits a file, the streamed
  content is rendered as a rolling tail of the last 10 lines.
- **Small and readable.** Go you can audit in one sitting.

## Versioning

drea follows [Semantic Versioning](https://semver.org/) (`MAJOR.MINOR.PATCH`).
The leading `v0.x` line is pre-stable: anything may still change while the
interface and behaviour settle. Releases carry stability suffixes:

- `v0.1.0-alpha.N` — early preview releases. The current release is
  **`v0.1.0-alpha.1`**.
- `-beta.N` — feature-complete previews for wider testing.
- `-rc.N` — release candidates.
- `v0.1.0` — first stable-enough minor release.

Use `drea --version` (or `drea -v`) to see the version of the binary you are
running.

## Build

Requires Go 1.19+ (builds with the Go version shipped in Debian 12).

```sh
make build        # produces ./drea
# or
go build -o drea ./cmd/drea
```

Run the tests (standard library only):

```sh
make test
```

### Debian 12 (bookworm), from official sources only

Debian 12 ships a new-enough Go in its official repositories, so no external
repos, PPAs, `go get`, npm or pip are needed. Install Go and Git, then build:

```sh
sudo apt update
sudo apt install -y golang-go git ca-certificates
go version                       # expect go1.19.x from Debian

git clone https://github.com/dreaagent/drea.git
cd drea
go build -o drea ./cmd/drea      # -> ./drea (make is optional)
```

That's it — `./drea` is a single static binary with no runtime dependencies.
Optionally put it on your PATH:

```sh
sudo install -m 0755 drea /usr/local/bin/drea
```

Then run against your model endpoint:

```sh
export DREA_API_KEY='your-key'          # keeps the key out of shell history
./drea --model gpt-4o "inspect this project and run its tests"
# or a local, keyless endpoint:
./drea --base-url http://127.0.0.1:8080/v1 --model my-local-model "…"
# or interactive:
./drea
```

## Configure

Configuration comes from flags, environment variables, an optional settings
file and an optional key file, in that order of precedence:

1. command-line flags
2. environment variables
3. the settings file (`~/.config/drea/settings.json`)
4. the key file (`~/.config/drea/key`)
5. built-in defaults

| Flag | Env | Default | Meaning |
|------|-----|---------|---------|
| `--base-url` | `DREA_BASE_URL` | `https://api.openai.com/v1` | OpenAI-compatible API base URL |
| `--key` | `DREA_API_KEY` / `OPENAI_API_KEY` | — | Bearer API key (blank for local endpoints) |
| `--model` | `DREA_MODEL` | `gpt-4o` | Model name |
| `--workdir` | `DREA_WORKDIR` | current dir | Workspace root the agent is confined to |
| `--auto` | `DREA_AUTO_APPROVE` | off | Skip confirmation for commands/writes |
| `--max-steps` | — | 50 | Max model turns per task |
| `--temperature` | — | 1 | Sampling temperature |
| `--top-p` | — | 0.95 | Nucleus-sampling probability |
| `--reasoning-effort` | `DREA_REASONING_EFFORT` | — (saved value) | Reasoning effort level: `low`, `medium` or `high`; empty means the endpoint default (persisted via `/save`) |
| `--no-json-mode` | — | off | Disable the strict JSON tool-call schema (`response_format`); only use with endpoints that do not support it |
| `--json-format` | `DREA_JSON_FORMAT` | `json_schema` | `response_format` variant: `json_schema` (OpenAI) or `json_object` (llama.cpp/older servers) |
| `--debug` | `DREA_DEBUG` | — | Dump raw request/response traffic to the given file |
| `--verify` | `DREA_VERIFY` | — | Command run when a task completes; failures are fed back for self-correction |
| `--verify-attempts` | `DREA_VERIFY_ATTEMPTS` | 3 | How many times a failing verify command is fed back before giving up |
| `--checkpoint` | `DREA_CHECKPOINT` | off | Commit the workspace before each task (and measure `--verify` first) so a task that regresses it can be rolled back |
| `--worktree` | — | off | Run in a scratch git worktree of the workspace, so an attempt cannot damage it |
| `--promote` | — | off | With `--worktree`, merge the work back when the verify command passes and it fast-forwards cleanly |
| `--context-tokens` | `DREA_CONTEXT_TOKENS` | 96000 | Prompt-size budget above which older history is compacted |
| `--persist` | `DREA_NO_PERSIST` (inverts) | on | Save the transcript so a session can be resumed |
| `--resume` | — | off | Resume the most recent saved session for this workspace |
| `--allow` (repeatable) | `DREA_ALLOW_COMMANDS` | — | Regex for a `run_command` command to auto-run without a prompt |
| `--deny` (repeatable) | `DREA_DENY_COMMANDS` | — | Regex for a `run_command` command to block outright (wins over allow) |

The `DREA_ALLOW_COMMANDS` / `DREA_DENY_COMMANDS` environment variables take a
newline-separated list of regexes; each `--allow` / `--deny` flag adds one
pattern.

Prefer the environment variable for the key so it does not land in your shell
history — or store it once with `/key <key>` + `/save` (see below) and it is
read back from disk on every start, so a bare `./drea` works.

### Settings file

The base URL, model, verify command, checkpointing, context budget, reasoning
effort and command policy can be persisted to `~/.config/drea/settings.json`
(follows `XDG_CONFIG_HOME`) with the `/save` command in an interactive session.
The file is written `0600`. The API key is never stored in it — `/save` writes
the key to its own file, `~/.config/drea/key` (also `0600`), and on startup the
key is read from there when neither `--key` nor the environment provides one.
Flags and environment variables still override the saved values.

### Context management, sessions and verification

- **Tool-call recovery.** The harness drives the model through the native
  `tool_calls` mechanism. Some small models (and local OpenAI-compatible shims)
  lack reliable tool-calling and instead print a tool call as JSON text, e.g.
  `{"name":"list_dir","arguments":{"path":"."}}`. When a turn produces text but
  no tool call, `drea` parses that JSON and executes it as a tool call rather
  than mistaking the turn for "done" and dropping back to the prompt. For models
  that still won't cooperate, `drea` asks the endpoint to constrain output to a
  strict JSON tool-call schema via `response_format` by default. Endpoints that
  do not support it can opt out with `--no-json-mode`. In JSON mode, plain prose
  replies and `reply` pseudo-tool messages are still streamed to the terminal as
  they arrive, so you do not watch reasoning scroll by only to have the final
  answer pop in all at once.
- **Context management.** Long tasks no longer overflow the model window: the
  prompt-token count is tracked (using the endpoint's reported usage, or a
  fallback estimate) and once it exceeds `--context-tokens`, older turns are
  summarized into a single note while the system prompt and recent turns are
  kept verbatim. Individual tool outputs stored in history are also size-capped.
- **Sessions.** With persistence on (the default), the transcript for each
  workspace is written atomically to `~/.config/drea/sessions/` after every
  step, so an exit or crash loses nothing. Resume with `drea --resume` or the
  `/resume` command. Transcripts contain only messages — never the API key.
- **Verification loop.** Set `--verify` (e.g. `--verify 'go build ./... && go
  test ./...'`) and, when the model believes a task is done, `drea` runs it
  automatically; a non-zero exit is fed back so the agent self-corrects, bounded
  by `--verify-attempts` so it cannot loop forever.

### Goal-driven iteration

The verify command is the only objective measure the harness has, so
`--checkpoint` turns it into a goal rather than a final check:

```
drea --checkpoint --verify 'go test ./...' --auto 'make the parser handle empty input'
```

(`--auto` can be replaced with `--auto-write`, `--auto-git` and/or
`--auto-run` to limit what is auto-approved.)

Before the task, `drea` commits the workspace (so there is a ref to return to)
and runs the verify command once to record where things stood. It then works,
verifies, and feeds failures back up to `--verify-attempts` times. If the
measure passed *before* the task and still fails after the budget is spent, the
task made things demonstrably worse, and `drea` rolls back to the checkpoint
(after confirmation, or straight away under `--auto`).

Rolling back does not destroy the attempt: it is committed to a
`drea/attempt-<n>` branch first, and only then is the work tree returned to the
checkpoint, so what was tried stays reviewable with `git diff <checkpoint>
<branch>` instead of having to be rediscovered. Files the task added are removed
from the work tree along with its edits — anything that existed beforehand is
inside the checkpoint commit and survives. If the attempt cannot be preserved,
nothing is rolled back.

This is the only case judged automatically. If the measure was already failing,
or there is none, the work is kept: only the user can say whether a partial fix
is progress. Requires a git repository; without one, checkpointing is skipped
with a warning rather than failing the task.

**Stall detection.** A model that is stuck repeats itself. When the same tool is
called with byte-identical arguments three times in a row, `drea` tells it so
and asks for a different approach; after five, the task is stopped rather than
spending the remaining step budget. Different arguments never count, so reading
many files or re-running a test after an edit is unaffected.

**Cost.** Token usage reported by the endpoint is accumulated and printed after
each task; `/usage` shows the session total. Nothing is estimated: an endpoint
that reports no usage shows zeroes.

### Prompt caching

Some OpenAI-compatible gateways cache the prompt prefix when a stable
`prompt_cache_key` is sent with each request. Enable it with
`--prompt-cache-key <key>` or `DREA_PROMPT_CACHE_KEY`, and set it in a session
with `/cache-key <key>`. The key is saved with `/save`.

For a cache to hit, the prefix of the request must stay byte-identical across
turns. `drea` keeps the conversation history stable, so caching works normally
until the context budget is reached and automatic compaction summarizes older
messages — after which a cache miss is expected. After each response, `drea`
reports the cache status when the provider returns it (via headers such as
`X-Cache-Status` or usage fields such as `prompt_cache_hit_tokens`).

### Isolated attempts (`--worktree`)

`drea --worktree …` creates a scratch `git worktree` of the workspace on its own
`drea/…` branch (under your cache dir) and works there, so a change that goes
wrong cannot damage the repository it was asked to improve. By default nothing
is merged: at the end, an untouched worktree is removed, and one containing
work is kept, with the exact commands to review, merge or discard it.

Adding `--promote` merges the work back, but only on terms that can be checked
rather than assumed:

```
drea --worktree --promote --verify 'go test ./...' --auto 'add retry to the fetch helper'
```

Promotion needs a verify command, that command passing, and the workspace being
able to fast-forward to the branch (so it is clean and has not moved on).
Anything else — no measure, a failing one, a diverged or dirty workspace — is a
judgement call, and `drea` says which one applies and leaves the work in the
worktree for you. Cleanup is never what loses work either: a worktree counts as
untouched only if it has no uncommitted changes *and* no commits of its own, and
its branch is deleted with `branch -d`, which refuses to drop unmerged commits.

A worktree starts from `HEAD`, so uncommitted changes in the original tree are
*not* carried over — `drea` warns when that is the case rather than quietly
working from a different state than you are looking at. It needs a repository
with at least one commit and refuses to start otherwise.

### Command policy (unattended, reversible autonomy)

The command policy classifies each `run_command` the model wants to run so the
harness can run unattended without blanket trust:

- **deny** — matching commands are refused outright and never executed, *even
  with `--auto-run` or `--auto`*. Deny always wins over allow.
- **allow** — matching commands run without a confirmation prompt (as if
  auto-approved), so routine safe commands (`go test`, `git status`, …) don't
  interrupt an unattended run.
- otherwise the normal approval behaviour applies (prompt unless `--auto-run`
  or `--auto`).

Auto-approval can also be split by tool category. `--auto` still enables all
mutating actions, but you can use `--auto-write`, `--auto-git` and `--auto-run`
separately. For example, `--auto-write` lets the model edit files without
interrupting you, while `run_command` still prompts (unless the command policy
allows it).

A built-in deny list always blocks clearly catastrophic commands (e.g.
`rm -rf /`, `mkfs`, `dd of=/dev/…`, `shutdown`/`reboot`, fork bombs), regardless
of configuration; user allow patterns cannot override it. Patterns are Go
(RE2) regexes matched against the command string, and invalid patterns are
rejected at startup.

```sh
# Run unattended: auto-run tests and git inspection, block network fetches.
drea --auto \
  --allow '^go (test|build|vet)\b' --allow '^git (status|diff|log|show)\b' \
  --deny '\b(curl|wget)\b' \
  "refactor the parser and keep the tests green"
```

```sh
# Auto-approve file edits only; commands still prompt (or obey the policy).
drea --auto-write "refactor the parser and keep the tests green"
```

Inspect the effective policy in an interactive session with `/policy`. This is
**defence in depth layered on the approval prompt — not a sandbox**: it governs
the harness's decision to run or prompt for a command, not what a running
process can do. Real isolation still relies on least privilege and OS-level
sandboxing.

### Git checkpoints and rollback

The agent has first-class, workspace-confined git tools so it can snapshot its
work and cleanly undo a bad change rather than hand-reverting it — the key
safety net for iterating on a codebase:

- `git_inspect` (read-only): `status`, `diff` (optional `path`/`staged`), `log`,
  `show`.
- `git_init`: create a repository when the workspace isn't under version control
  yet. Idempotent, and it never nests a second repository inside an existing one.
- `git_commit`: stage changes (all by default, or specific `paths`) and create
  a checkpoint commit; requires a `message`.
- `git_rollback`: `git reset --hard` back to a `ref` (default `HEAD`),
  optionally `clean=true` to also remove untracked files. Destructive, so it
  always requires approval.

Every git invocation uses an explicit argument vector (`git -C <root> …`), never
a shell string, and ref/path arguments are validated and workspace-confined, so
the model can't point git at another repository or smuggle in extra arguments.
The read tool is `git_inspect`, not `git`: a tool named exactly after a shell
command invites the model to conflate the two and reach for `run_command`, which
bypasses the approval prompt and this argument validation entirely.
A typical loop: `git_commit` a known-good checkpoint → make a risky change →
verify → `git_rollback` if it regressed.

A workspace that isn't a repository is a normal starting state, not an error:
the agent is instructed to `git_init` and commit a baseline first, so checkpoints
exist from the outset. Commits work even where git has no configured
`user.name`/`user.email` (the default on a fresh machine) — drea supplies a
`github.com/dreaagent/drea <github.com/dreaagent/drea@localhost>` identity for that commit only, and never touches
your git configuration or an identity you have already set. Read actions on a
repository with no commits yet report `(no commits yet)` rather than failing.

### Evaluation

`drea eval <dir>` runs every `*.json` task spec in a directory and scores each
by its own verify command, printing a pass/fail report and exiting non-zero if
any fail. This makes repeated self-improvement measurable. A spec looks like:

```json
{
  "name": "adds-health-endpoint",
  "prompt": "Add a /health endpoint returning 200 and a test for it.",
  "workdir": "fixtures/webapp",
  "setup": "git checkout -- .",
  "verify": "go test ./..."
}
```

## Interactive commands

In an interactive session, lines starting with `/` are commands rather than
tasks:

| Command | Purpose |
|---------|---------|
| `/help` | List the commands |
| `/config` | Show the current configuration (the key is masked) |
| `/model [name]` | Show or change the model |
| `/host [url]` | Show or change the API base URL |
| `/auto [on\|off\|write\|git\|run]` | Show or toggle auto-approve (e.g. `/auto write on`) |
| `/verify [cmd]` | Show or set the verify command (`/verify off` to clear) |
| `/checkpoint [on\|off]` | Show or toggle per-task checkpointing |
| `/usage` | Show token usage for this session |
| `/policy` | Show the `run_command` allow/deny policy (incl. built-in denies) |
| `/reasoning [low\|medium\|high\|off]` | Show or set the reasoning effort level |
| `/key [value\|off\|show]` | Show (masked) or set the API key; `/save` persists it |
| `/save` | Persist model + base URL + verify + policy + reasoning + API key |
| `/resume` | Reload the saved transcript for this workspace |
| `/reset` | Clear the conversation history |
| `/tools` | List the available tools |
| `/exit` | Quit (also `/quit`, `exit`, Ctrl-D) |
| `/multiline [on\|off]` | Toggle multi-line input (Enter starts a new line; Ctrl-D sends) |

### Interactive reply continuation

In an interactive session, when the assistant ends a turn with a `reply`
(either alone or as the last tool call after real work), `drea` asks the model
to classify that reply. It calls a small internal `classify_reply` tool with
the original task and the reply, and the model answers either
`CONTINUE_AUTONOMOUSLY` or `PAUSE_FOR_USER`.

If the model classifies the reply as a status update or statement of intent
(e.g. "GTK3 is available; let me create the GUI calculator."), `drea`
automatically continues the task with a system instruction instead of making
you press Enter after every progress report. If the reply asks a question,
offers a choice, requests confirmation, or the model's intent is ambiguous,
the chat pauses and waits for you as usual.

This is bounded by `--reply-chain` (default 15, configurable with
`DREA_REPLY_CHAIN` or `/save`), so a model that keeps chatting cannot loop
forever without your attention.

## Use

One-shot task (runs until done, then exits):

```sh
export DREA_API_KEY=sk-...
drea --model gpt-4o "add a /health endpoint and a test, then run the tests"
```

Interactive session (keeps conversation context across turns):

```sh
drea
› scaffold a small Go CLI that prints the current time
```

Point it at a local, keyless server (e.g. llama.cpp / Ollama's OpenAI shim):

```sh
drea --base-url http://localhost:8080/v1 --model my-local-model "…"
```

## Tools

The model is given thirteen tools, all confined to the workspace root:

| Tool | Mutating | Purpose |
|------|----------|---------|
| `read_file` | no | Read a text file (optional line offset/limit) |
| `record_thoughts` | no | Persist reasoning or a plan into the conversation context |
| `list_dir` | no | List a directory |
| `search` | no | Regexp search across files (pure Go, optional glob) |
| `git_inspect` | no | Read-only git inspection: status / diff / log / show |
| `write_file` | **yes** | Create/overwrite a file |
| `edit_file` | **yes** | Replace one string in a file |
| `apply_patch` | **yes** | Apply several edits to one file, all-or-nothing |
| `run_command` | **yes** | Run a shell command via `bash -c`, with timeout |
| `git_init` | **yes** | Create a repository so work can be checkpointed |
| `git_commit` | **yes** | Stage changes and create a checkpoint commit |
| `git_rollback` | **yes** | `git reset --hard` to a ref (optionally clean untracked) |
| `reply` | no | Send plain-text prose to the user (used to report status or finish a turn) |

Mutating tools prompt for approval unless their category is auto-approved
(`--auto-write`, `--auto-git`, `--auto-run`) or the `--auto` master switch is
set. `run_command` is additionally subject to the allow/deny command policy
(see above).

### Reliable edits

`edit_file` and `apply_patch` share one matching engine (`internal/patch`),
which is also what renders the diff preview — so what you approve is exactly
what gets written.

- **Multi-hunk.** `apply_patch` takes a list of edits applied in order to one
  file. They are applied in memory and written only if *every* edit succeeds, so
  a failure late in the list can never leave a half-patched file.
- **Whitespace-tolerant.** If no exact match exists, the text is matched
  line-by-line ignoring trailing whitespace, then ignoring indentation too. A
  model that re-indents a snippet no longer wastes a turn.
- **Never guesses.** A fuzzy match is accepted only when it is unique. Zero
  matches or several candidates is an error telling the model to add context;
  the file is left untouched. Whitespace *inside* a line is never normalised, as
  that could match a semantically different line.
- File permissions are preserved, so patching a script keeps it executable.

### Project instructions

On startup the agent reads the workspace's own instruction files and appends
them to its system prompt, so a project's rules come from the project rather
than from the harness. Recognised in the workspace root, most specific first:
`AGENTS.md`, `CLAUDE.md`, `CONVENTIONS.md`, `CONTRIBUTING.md`, `README.md`,
`README` (matched case-insensitively).

Loading is bounded — 8 KiB per file, 16 KiB in total, truncated on a line
boundary — so instructions cannot crowd out the actual work. Only regular files
directly in the root are read; symlinks are ignored, so a repository cannot use
one to make the agent read an arbitrary file. No instruction files is a normal
case, not an error.

This is how the harness learns *any* project's conventions, including its own:
nothing about drea itself is compiled into the binary.

## Layout

```
cmd/drea/          CLI entry: flags, banner, one-shot & interactive loop
internal/config/   configuration (flags + env, workspace validation)
internal/llm/      OpenAI-compatible client: SSE streaming + tool-call accrual
internal/tool/     tool registry and implementations (fs, search, shell)
internal/agent/    reason-act loop, system prompt, approval handling
internal/ui/       terminal output and prompts (ANSI, TTY-aware, live tail)
internal/diff/     standard-library unified line diff
internal/settings/ optional on-disk preferences (base URL + model, no secrets)
internal/session/  resumable per-workspace transcript persistence (no secrets)
internal/eval/     minimal evaluation scaffold (task specs + scoring)
internal/policy/   allow/deny command policy (regex, deny-wins, built-in denies)
internal/patch/    multi-hunk edit engine (exact, then whitespace-tolerant)
internal/vcs/      the harness's own git bookkeeping: checkpoints, worktrees
internal/conventions/ discovery of the workspace's own instruction files
```

## Security notes

- File tools cannot read or write outside `--workdir`.
- `run_command` is the one broad capability; it is gated by approval, bounded by
  a wall-clock timeout, and its output is size-capped.
- The only network egress is to the configured model endpoint. The API key is
  sent solely as a `Bearer` token to that endpoint and is never logged.
- The command policy (allow/deny) is defence in depth on top of approval, with
  a built-in deny list for catastrophic commands that user config cannot
  override. It is not a sandbox: it governs whether the harness runs or prompts
  for a command, not what a running process may do.
- Git tools are workspace-confined and run git with an explicit argument vector
  (never a shell string); refs and paths are validated and confined.
- Project instructions are read only from regular files in the workspace root,
  never through symlinks, and are size-bounded.

## Module path

`go.mod` declares `module github.com/dreaagent/drea`, matching the import paths
used throughout the source (`github.com/dreaagent/drea/internal/...`). The
project builds cleanly as-is; no local `replace` directive is needed.
