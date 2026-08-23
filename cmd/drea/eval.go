package main

import (
	"context"
	"errors"
	"fmt"
	"os"

	"github.com/dreaagent/drea/internal/agent"
	"github.com/dreaagent/drea/internal/config"
	"github.com/dreaagent/drea/internal/eval"
	"github.com/dreaagent/drea/internal/llm"
	"github.com/dreaagent/drea/internal/settings"
	"github.com/dreaagent/drea/internal/tool"
	"github.com/dreaagent/drea/internal/ui"
)

// runEval loads task specs from a directory, runs the agent (auto-approving,
// non-persistent) against each, scores it with the spec's verify command, and
// prints a report. It exits non-zero if any task fails, so it can gate
// self-improvement in CI or a loop.
func runEval(args []string) error {
	if len(args) != 1 || args[0] == "-h" || args[0] == "--help" {
		fmt.Fprintln(os.Stderr, "Usage: drea eval <dir>")
		fmt.Fprintln(os.Stderr, "\nRuns every .json task spec in <dir> and reports pass/fail.")
		if len(args) != 1 {
			return errors.New("eval requires exactly one directory argument")
		}
		return nil
	}

	specs, err := eval.Load(args[0])
	if err != nil {
		return err
	}

	saved, _, _ := settings.Load()
	key, _, _ := settings.LoadKey()
	base := config.Defaults(config.Saved{
		BaseURL:         saved.BaseURL,
		APIKey:          key,
		Model:           saved.Model,
		Verify:          saved.Verify,
		ContextTokens:   saved.ContextTokens,
		ReasoningEffort: saved.ReasoningEffort,
		PromptCacheKey:  saved.PromptCacheKey,
	})

	u := ui.New()
	u.Info(fmt.Sprintf("running %d eval task(s) with model %s", len(specs), base.Model))

	var results []eval.Result
	for _, s := range specs {
		results = append(results, runOne(base, s, u))
	}

	u.Info("\n" + eval.Summary(results))
	if !eval.Passed(results) {
		return errors.New("one or more eval tasks failed")
	}
	return nil
}

// runOne executes a single eval task in its own agent, unattended.
func runOne(base config.Config, s eval.Spec, u *ui.UI) eval.Result {
	u.Info(fmt.Sprintf("\n=== %s ===", s.Name))

	cfg := base
	cfg.Workdir = s.Workdir
	cfg.AutoApprove = true // unattended: enable all categories
	cfg.AutoWrite = true
	cfg.AutoGit = true
	cfg.AutoRun = true
	cfg.Persist = false // eval runs are ephemeral
	cfg.Verify = ""     // scored explicitly below, not inside the loop
	if err := cfg.Normalize(); err != nil {
		return eval.Result{Name: s.Name, Detail: "invalid workdir: " + err.Error()}
	}

	ctx := context.Background()

	if s.Setup != "" {
		out, code, _, err := tool.RunShell(ctx, cfg.Workdir, s.Setup, cfg.CommandTimeout)
		if err != nil || code != 0 {
			return eval.Result{Name: s.Name, Detail: "setup failed: " + firstLine(out)}
		}
	}

	client := llm.NewClientWithReasoning(cfg.ChatURL(), cfg.APIKey, cfg.Model, cfg.Temperature, cfg.TopP, cfg.ReasoningEffort, cfg.RequestTimeout, cfg.JSONMode, cfg.JSONFormat)
	client.SetPromptCacheKey(cfg.PromptCacheKey)
	tools := tool.NewRegistry(cfg.Workdir)
	ag := agent.New(&cfg, client, tools, u)
	if err := ag.Run(ctx, s.Prompt); err != nil {
		return eval.Result{Name: s.Name, Detail: "agent error: " + err.Error()}
	}

	out, code, timedOut, err := tool.RunShell(ctx, cfg.Workdir, s.Verify, cfg.CommandTimeout)
	if err != nil {
		return eval.Result{Name: s.Name, Detail: "verify could not run: " + err.Error()}
	}
	if code == 0 && !timedOut {
		return eval.Result{Name: s.Name, Passed: true}
	}
	return eval.Result{Name: s.Name, Detail: "verify failed: " + firstLine(out)}
}

func firstLine(s string) string {
	for i := 0; i < len(s); i++ {
		if s[i] == '\n' {
			return s[:i]
		}
	}
	if len(s) > 200 {
		return s[:200]
	}
	return s
}
