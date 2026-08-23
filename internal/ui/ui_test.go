package ui

import (
	"bufio"
	"bytes"
	"fmt"
	"strings"
	"testing"
)

func TestSanitizeLine(t *testing.T) {
	cases := []struct{ in, want string }{
		{"plain", "plain"},
		{"a\tb", "a   b"},            // tab -> next stop of 4
		{"\tx", "    x"},             // leading tab
		{"has\rcr", "hascr"},         // carriage return dropped
		{"esc\x1b[31mred", "escred"}, // ESC sequence stripped
		{"bell\x07", "bell"},         // other control dropped
		{"keeps ünïcode ☃", "keeps ünïcode ☃"},
	}
	for _, c := range cases {
		if got := sanitizeLine(c.in, 4); got != c.want {
			t.Errorf("sanitizeLine(%q) = %q, want %q", c.in, got, c.want)
		}
	}
}

func TestClip(t *testing.T) {
	if got := clip("hello", 10); got != "hello" {
		t.Errorf("no clip: got %q", got)
	}
	if got := clip("hello world", 5); got != "hell…" {
		t.Errorf("clip: got %q", got)
	}
	if got := clip("héllo wörld", 4); len([]rune(got)) != 4 {
		t.Errorf("clip rune count = %d, want 4 (%q)", len([]rune(got)), got)
	}
	if got := clip("anything", 0); got != "" {
		t.Errorf("clip 0: got %q", got)
	}
}

func TestLastLines(t *testing.T) {
	if got := lastLines("a\nb\nc\n", 2); len(got) != 2 || got[0] != "b" || got[1] != "c" {
		t.Errorf("tail = %v", got)
	}
	if got := lastLines("only", 5); len(got) != 1 || got[0] != "only" {
		t.Errorf("single = %v", got)
	}
}

// TestLiveViewRedrawCount verifies the cursor-up count on redraw equals the
// number of lines drawn in the previous frame — the invariant that was broken
// when wrapped lines caused duplicate output.
func TestLiveViewRedrawCount(t *testing.T) {
	var buf bytes.Buffer
	u := &UI{out: &buf, tty: true, colour: false}
	lv := u.LiveView("writing x", 10)

	lv.Update("l1\nl2\nl3")
	first := buf.String()
	if strings.Contains(first, "\x1b[") && strings.Contains(first, "A\r") {
		t.Fatalf("first frame should not move the cursor up: %q", first)
	}

	buf.Reset()
	lv.Update("l1\nl2\nl3\nl4")
	second := buf.String()
	if !strings.Contains(second, "\x1b[3A\r\x1b[J") {
		t.Fatalf("second frame should move up 3 lines (prev body), got %q", second)
	}
}

// TestLiveViewClipsLongLines ensures a very long line is truncated so it fits
// on one physical row (no wrapping, hence no duplication on redraw).
func TestLiveViewClipsLongLines(t *testing.T) {
	var buf bytes.Buffer
	u := &UI{out: &buf, tty: true, colour: false}
	lv := u.LiveView("writing x", 10)

	long := strings.Repeat("A", 500)
	lv.Update(long)

	for _, line := range strings.Split(strings.TrimRight(buf.String(), "\n"), "\n") {
		if strings.HasPrefix(line, livePrefix) {
			if n := len([]rune(line)); n > 80 {
				t.Errorf("rendered line width %d exceeds 80: %q", n, line)
			}
			if !strings.Contains(line, "…") {
				t.Errorf("long line should be clipped with ellipsis: %q", line)
			}
		}
	}
}

// TestLiveViewWrappedRedrawCount verifies the cursor-up count on redraw equals
// the number of physical rows drawn in the previous frame when lines wrap.
func TestLiveViewWrappedRedrawCount(t *testing.T) {
	var buf bytes.Buffer
	u := &UI{out: &buf, tty: true, colour: false}
	lv := u.LiveViewWrapped("reasoning", 10)

	termWidth := termWidth()
	if termWidth < 20 {
		t.Skip("terminal too narrow for meaningful wrap test")
	}

	lv.Update("l1\nl2\nl3")
	first := buf.String()
	if strings.Contains(first, "\x1b[") && strings.Contains(first, "A\r") {
		t.Fatalf("first frame should not move the cursor up: %q", first)
	}

	buf.Reset()
	lv.Update("l1\nl2\nl3\nl4")
	second := buf.String()
	want := fmt.Sprintf("\x1b[%dA\r\x1b[J", 3)
	if !strings.Contains(second, want) {
		t.Fatalf("second frame should move up 3 rows (prev body), got %q", second)
	}
}

// TestLiveViewWrapsLongLines ensures a very long line is wrapped onto multiple
// physical rows instead of being clipped.
func TestLiveViewWrapsLongLines(t *testing.T) {
	var buf bytes.Buffer
	u := &UI{out: &buf, tty: true, colour: false}
	lv := u.LiveViewWrapped("reasoning", 10)

	termWidth := termWidth()
	if termWidth < 20 {
		t.Skip("terminal too narrow for meaningful wrap test")
	}

	long := strings.Repeat("A", termWidth*3)
	lv.Update(long)

	prefixCount := 0
	for _, line := range strings.Split(strings.TrimRight(buf.String(), "\n"), "\n") {
		if strings.HasPrefix(line, livePrefix) {
			prefixCount++
			if strings.Contains(line, "…") {
				t.Errorf("wrapped line should not be clipped with ellipsis: %q", line)
			}
			if n := len([]rune(line)); n > termWidth {
				t.Errorf("rendered line width %d exceeds terminal width %d: %q", n, termWidth, line)
			}
		}
	}
	if prefixCount < 2 {
		t.Errorf("long line should wrap to at least 2 rows, got %d", prefixCount)
	}
}

// TestWrapLine verifies the wrapping helper.
func TestWrapLine(t *testing.T) {
	if got := wrapLine("hello", 10); len(got) != 1 || got[0] != "hello" {
		t.Errorf("short line: %v", got)
	}
	if got := wrapLine("hello world", 5); len(got) != 3 || got[0] != "hello" || got[1] != " worl" || got[2] != "d" {
		t.Errorf("wrap 5: %v", got)
	}
	if got := wrapLine("", 5); len(got) != 1 || got[0] != "" {
		t.Errorf("empty line: %v", got)
	}
}

// TestStreamFilter verifies that streamed assistant text is stripped of escape
// sequences and control characters while newlines and tabs survive.
func TestStreamFilter(t *testing.T) {
	var f streamFilter
	cases := []struct{ in, want string }{
		{"plain text", "plain text"},
		{"keep\nnewline", "keep\nnewline"},
		{"keep\ttab", "keep\ttab"},
		{"drop\rcr", "dropcr"},
		{"bell\x07here", "bellhere"},
		{"\x1b[31mred\x1b[0m", "red"},   // CSI colour sequences stripped
		{"move\x1b[2Kaway", "moveaway"}, // CSI erase stripped
		{"esc\x1bXchar", "escchar"},     // two-char escape: ESC + one byte
		{"keeps ünïcode ☃", "keeps ünïcode ☃"},
	}
	for _, c := range cases {
		f = streamFilter{}
		if got := f.feed(c.in); got != c.want {
			t.Errorf("feed(%q) = %q, want %q", c.in, got, c.want)
		}
	}
}

// TestStreamFilterSplitEscape ensures an escape sequence split across chunks is
// still neutralised (state carries between feed calls).
func TestStreamFilterSplitEscape(t *testing.T) {
	var f streamFilter
	var got strings.Builder
	got.WriteString(f.feed("hi\x1b"))   // ESC at end of a chunk
	got.WriteString(f.feed("[1;31m"))   // rest of the CSI sequence
	got.WriteString(f.feed("bold red")) // visible text
	if got.String() != "hibold red" {
		t.Errorf("split escape = %q, want %q", got.String(), "hibold red")
	}
}

// TestAssistantSanitises checks the full Assistant path strips injected escapes
// (the reported bug: a random-character test string rendered like AI text).
func TestAssistantSanitises(t *testing.T) {
	var buf bytes.Buffer
	u := &UI{out: &buf, colour: false}
	u.AssistantHeader()
	buf.Reset()
	u.Assistant("output: \x1b[31m$%^&*[]\x1b[0m done\n")
	if got := buf.String(); got != "output: $%^&*[] done\n" {
		t.Errorf("Assistant wrote %q", got)
	}
}

func TestReadInputSingleLine(t *testing.T) {
	u := &UI{in: bufio.NewReader(strings.NewReader("hello\n"))}
	got, err := u.ReadInput()
	if err != nil {
		t.Fatalf("ReadInput: %v", err)
	}
	if got != "hello" {
		t.Errorf("single-line ReadInput = %q, want %q", got, "hello")
	}
}

func TestReadInputMultiline(t *testing.T) {
	// Ctrl-D (EOF) after content sends the message; blank lines are content.
	u := &UI{in: bufio.NewReader(strings.NewReader("line one\n\nline two\n"))}
	u.SetMultiline(true)
	got, err := u.ReadInput()
	if err != nil {
		t.Fatalf("ReadInput: %v", err)
	}
	if got != "line one\n\nline two" {
		t.Errorf("multi-line ReadInput = %q", got)
	}
}

// TestReadInputPastedCode simulates pasting a 20-line block of code in
// multi-line mode. The whole block must come back as a single message, not as
// twenty one-line messages, even when it contains blank lines.
func TestReadInputPastedCode(t *testing.T) {
	var sb strings.Builder
	for i := 1; i <= 20; i++ {
		fmt.Fprintf(&sb, "line %02d of pasted code\n", i)
		if i%5 == 0 {
			sb.WriteString("\n") // blank line inside the block is ordinary content
		}
	}

	u := &UI{in: bufio.NewReader(strings.NewReader(sb.String()))}
	u.SetMultiline(true)
	got, err := u.ReadInput()
	if err != nil {
		t.Fatalf("ReadInput: %v", err)
	}
	if n := strings.Count(got, "\n"); n != 22 {
		t.Errorf("pasted block has %d newlines, want 22 (20 lines + 4 blank, trailing newline trimmed, one message)", n)
	}
	if !strings.HasPrefix(got, "line 01 of pasted code") || !strings.HasSuffix(got, "line 20 of pasted code") {
		t.Errorf("pasted block not preserved as one message: %q", got)
	}
}

func TestTermWidthFallback(t *testing.T) {
	// In the test harness stdout is not a terminal, so termWidth must fall
	// back to a sane default rather than 0.
	if w := termWidth(); w <= 0 {
		t.Errorf("termWidth = %d, want > 0", w)
	}
}
