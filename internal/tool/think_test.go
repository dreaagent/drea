package tool

import (
	"context"
	"testing"
)

func TestRecordThoughtsStoresReasoning(t *testing.T) {
	th := &recordThoughts{}
	if th.Mutating() {
		t.Error("record_thoughts must not be mutating")
	}

	out, err := th.Run(context.Background(), []byte(`{"thoughts":"I need to fix the negate logic before testing."}`))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if out != "thoughts recorded" {
		t.Fatalf("unexpected output: %q", out)
	}
}

func TestRecordThoughtsRejectsEmptyThoughts(t *testing.T) {
	th := &recordThoughts{}
	if _, err := th.Run(context.Background(), []byte(`{"thoughts":"   "}`)); err == nil {
		t.Fatal("expected error for empty thoughts")
	}
}

func TestRecordThoughtsSummaryTruncates(t *testing.T) {
	th := &recordThoughts{}
	long := ""
	for i := 0; i < 100; i++ {
		long += "x"
	}
	summary := th.Summary([]byte(`{"thoughts":"` + long + `"}`))
	if len(summary) > 85 {
		t.Fatalf("summary too long: %d", len(summary))
	}
}
