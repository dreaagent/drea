package tool

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestResolveConfinement(t *testing.T) {
	root := t.TempDir()
	cases := []struct {
		in      string
		wantErr bool
	}{
		{"a.txt", false},
		{"sub/b.txt", false},
		{".", false},
		{"../escape.txt", true},
		{"sub/../../escape.txt", true},
		{"/etc/passwd", true},
	}
	for _, tc := range cases {
		_, err := resolve(root, tc.in)
		if (err != nil) != tc.wantErr {
			t.Errorf("resolve(%q): err=%v, wantErr=%v", tc.in, err, tc.wantErr)
		}
	}
}

func TestResolveRejectsSymlinkEscape(t *testing.T) {
	root := t.TempDir()
	outside := t.TempDir()
	if err := os.WriteFile(filepath.Join(outside, "secret.txt"), []byte("top secret"), 0o644); err != nil {
		t.Fatal(err)
	}
	// A symlink inside the workspace pointing outside it must not be a way out.
	if err := os.Symlink(outside, filepath.Join(root, "link")); err != nil {
		t.Fatal(err)
	}
	if _, err := resolve(root, "link/secret.txt"); err == nil {
		t.Fatal("expected symlink escape to be rejected")
	}
	// A symlink that stays inside the workspace is still allowed.
	inside := filepath.Join(root, "real")
	if err := os.Mkdir(inside, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink(inside, filepath.Join(root, "innerlink")); err != nil {
		t.Fatal(err)
	}
	if _, err := resolve(root, "innerlink/ok.txt"); err != nil {
		t.Fatalf("in-workspace symlink should be allowed: %v", err)
	}
}

func TestRunCommandOutputCap(t *testing.T) {
	root := t.TempDir()
	rc := &runCommand{root: root}
	// Produce far more than maxOutputBytes; result must be bounded and flagged.
	out, err := run(t, rc, map[string]any{"command": "yes x | head -c 500000"})
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(out, "output truncated") {
		t.Fatalf("expected truncation marker, got %d bytes", len(out))
	}
	if len(out) > maxOutputBytes+256 {
		t.Fatalf("captured output not bounded: %d bytes", len(out))
	}
}

func TestRunCommandTimeout(t *testing.T) {
	root := t.TempDir()
	rc := &runCommand{root: root}
	out, err := run(t, rc, map[string]any{"command": "sleep 5", "timeout_seconds": 1})
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(out, "timed out") {
		t.Fatalf("expected timeout marker, got %q", out)
	}
}

func run(t *testing.T, tool Tool, args map[string]any) (string, error) {
	t.Helper()
	raw, _ := json.Marshal(args)
	return tool.Run(context.Background(), json.RawMessage(raw))
}

func TestWriteReadEdit(t *testing.T) {
	root := t.TempDir()
	w := &writeFile{root: root}
	if _, err := run(t, w, map[string]any{"path": "pkg/x.txt", "content": "one two two"}); err != nil {
		t.Fatal(err)
	}
	if data, _ := os.ReadFile(filepath.Join(root, "pkg/x.txt")); string(data) != "one two two" {
		t.Fatalf("write produced %q", data)
	}

	e := &editFile{root: root}
	// Non-unique without replace_all should error.
	if _, err := run(t, e, map[string]any{"path": "pkg/x.txt", "old_string": "two", "new_string": "2"}); err == nil {
		t.Error("expected error on ambiguous edit")
	}
	if _, err := run(t, e, map[string]any{"path": "pkg/x.txt", "old_string": "two", "new_string": "2", "replace_all": true}); err != nil {
		t.Fatal(err)
	}
	r := &readFile{root: root}
	out, err := run(t, r, map[string]any{"path": "pkg/x.txt"})
	if err != nil || !strings.Contains(out, "one 2 2") {
		t.Fatalf("read got %q err %v", out, err)
	}
}

func TestSearch(t *testing.T) {
	root := t.TempDir()
	os.WriteFile(filepath.Join(root, "a.go"), []byte("package main\nfunc Foo() {}\n"), 0o644)
	os.WriteFile(filepath.Join(root, "b.txt"), []byte("Foo in text\n"), 0o644)

	s := &search{root: root}
	out, err := run(t, s, map[string]any{"pattern": "Foo", "glob": "*.go"})
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(out, "a.go:2") || strings.Contains(out, "b.txt") {
		t.Fatalf("search glob failed: %q", out)
	}
}

func TestRunCommand(t *testing.T) {
	root := t.TempDir()
	rc := &runCommand{root: root}
	out, err := run(t, rc, map[string]any{"command": "echo hello && pwd"})
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(out, "hello") || !strings.Contains(out, "[exit code 0]") {
		t.Fatalf("run_command output: %q", out)
	}

	out, _ = run(t, rc, map[string]any{"command": "exit 3"})
	if !strings.Contains(out, "[exit code 3]") {
		t.Fatalf("expected exit code 3, got %q", out)
	}
}

func TestRegistrySpecs(t *testing.T) {
	r := NewRegistry(t.TempDir())
	specs := r.Specs()
	if len(specs) != 13 {
		t.Fatalf("got %d specs, want 13", len(specs))
	}
	for _, s := range specs {
		var js map[string]any
		if err := json.Unmarshal(s.Function.Parameters, &js); err != nil {
			t.Errorf("tool %s has invalid schema: %v", s.Function.Name, err)
		}
	}
}

func TestReplyTool(t *testing.T) {
	r := NewRegistry(t.TempDir())
	tool, ok := r.Get("reply")
	if !ok {
		t.Fatal("reply tool not registered")
	}
	if tool.Mutating() {
		t.Error("reply tool should not be mutating")
	}
	out, err := run(t, tool, map[string]any{"message": "hello"})
	if err != nil {
		t.Fatal(err)
	}
	if out != "" {
		t.Errorf("reply tool returned %q, want empty", out)
	}
	if got := tool.Summary(json.RawMessage(`{"message":"hello"}`)); got != "hello" {
		t.Errorf("summary = %q, want hello", got)
	}
	if got := tool.Summary(json.RawMessage(`{}`)); got != "(no message)" {
		t.Errorf("empty summary = %q, want (no message)", got)
	}
}
