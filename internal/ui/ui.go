// Package ui provides small, dependency-free terminal helpers: ANSI colour,
// section headers, streaming writers and interactive confirmation prompts.
package ui

import (
	"bufio"
	"errors"
	"fmt"
	"io"
	"os"
	"strings"
	"sync"
	"sync/atomic"
	"syscall"
	"time"
	"unsafe"
)

// ANSI colour codes. Colour is disabled automatically when stdout is not a
// TTY or when NO_COLOR is set (https://no-color.org/).
const (
	reset   = "\033[0m"
	dim     = "\033[2m"
	bold    = "\033[1m"
	red     = "\033[31m"
	green   = "\033[32m"
	yellow  = "\033[33m"
	blue    = "\033[34m"
	magenta = "\033[35m"
	cyan    = "\033[36m"
)

// UI writes formatted output and reads user input from a terminal.
type UI struct {
	out    io.Writer
	in     *bufio.Reader
	colour bool
	tty    bool
	afilt  streamFilter // sanitises streamed assistant text

	statusMu      sync.Mutex
	statusWorkdir string
	statusUsed    int
	statusTotal   int
	statusMsg     string
	statusShown   bool

	thinkMu     sync.Mutex
	thinkActive int32
	thinkStop   chan struct{}
	thinkDone   chan struct{}

	lineMu    sync.Mutex
	lineStart bool // true when the cursor is known to be at the start of a line

	multiline bool // when true, Ctrl-D (not Enter) ends ReadInput
}

// New builds a UI writing to stdout and reading from stdin.
func New() *UI {
	tty := isTTY()
	return &UI{
		out:       os.Stdout,
		in:        bufio.NewReader(os.Stdin),
		colour:    tty && !noColor(),
		tty:       tty,
		lineStart: true,
	}
}

// SetStatus configures the context-usage values shown in the bottom bar.
func (u *UI) SetStatus(used, total int) {
	u.statusMu.Lock()
	u.statusUsed = used
	if total > 0 {
		u.statusTotal = total
	}
	u.statusMu.Unlock()
}

// SetStatusWorkdir sets the working directory shown in the bottom bar.
func (u *UI) SetStatusWorkdir(wd string) {
	u.statusMu.Lock()
	u.statusWorkdir = wd
	u.statusMu.Unlock()
}

// ShowStatus draws (or redraws) the bottom status bar. On a non-TTY it does
// nothing. The bar is always the last line on screen; callers should ensure
// scrolling content leaves a blank row for it.
func (u *UI) ShowStatus() {
	if !u.tty {
		return
	}
	u.statusMu.Lock()
	used, total, wd, msg := u.statusUsed, u.statusTotal, u.statusWorkdir, u.statusMsg
	u.statusShown = true
	u.statusMu.Unlock()

	width := termWidth()
	if total <= 0 {
		total = 1
	}
	if used < 0 {
		used = 0
	}
	if used > total {
		used = total
	}
	pct := float64(used) / float64(total) * 100

	barWidth := width / 5
	if barWidth < 8 {
		barWidth = 8
	}
	filled := int(pct / 100 * float64(barWidth))
	if filled > barWidth {
		filled = barWidth
	}
	bar := strings.Repeat("█", filled) + strings.Repeat("░", barWidth-filled)

	left := fmt.Sprintf("context %s %5.1f%%", bar, pct)
	right := wd
	if msg != "" {
		right += " · " + msg
	}

	// Build the line with right-aligned workspace (and optional message).
	line := left
	if len([]rune(left))+1+len([]rune(right)) <= width {
		padding := width - len([]rune(left)) - len([]rune(right))
		if padding < 1 {
			padding = 1
		}
		line = left + strings.Repeat(" ", padding) + right
	}
	if len([]rune(line)) > width {
		line = string([]rune(line)[:width])
	}

	fmt.Fprintf(u.out, "\r\033[K%s", u.paint(dim, line))
	u.setLineStart(true)
}

// HideStatus clears the bottom status bar and moves the cursor to the start of
// that line so normal output can overwrite it.
func (u *UI) HideStatus() {
	if !u.tty {
		return
	}
	u.statusMu.Lock()
	u.statusShown = false
	u.statusMu.Unlock()
	fmt.Fprint(u.out, "\r\033[K")
	u.setLineStart(true)
}

// StatusShown reports whether the status bar is currently on screen.
func (u *UI) StatusShown() bool {
	u.statusMu.Lock()
	defer u.statusMu.Unlock()
	return u.statusShown
}

// SetStatusMessage sets an optional short message shown in the status bar.
func (u *UI) SetStatusMessage(msg string) {
	u.statusMu.Lock()
	u.statusMsg = msg
	u.statusMu.Unlock()
}

// StartThinking shows an animated "thinking" indicator above the status bar.
// It is a no-op on non-TTY output or if already active. Call StopThinking when
// the model response begins streaming.
func (u *UI) StartThinking() {
	if !u.tty {
		return
	}
	u.thinkMu.Lock()
	if u.thinkStop != nil {
		u.thinkMu.Unlock()
		return
	}
	u.thinkStop = make(chan struct{})
	u.thinkDone = make(chan struct{})
	stop := u.thinkStop
	done := u.thinkDone
	u.thinkMu.Unlock()

	atomic.StoreInt32(&u.thinkActive, 1)
	go u.thinkLoop(stop, done)
}

// StopThinking hides the thinking indicator and stops the animation.
func (u *UI) StopThinking() {
	if !u.tty {
		return
	}
	u.thinkMu.Lock()
	stop := u.thinkStop
	u.thinkStop = nil
	u.thinkMu.Unlock()
	if stop == nil {
		return
	}
	close(stop)
	<-u.thinkDone
	atomic.StoreInt32(&u.thinkActive, 0)
}

// ThinkingActive reports whether the thinking indicator is running.
func (u *UI) ThinkingActive() bool { return atomic.LoadInt32(&u.thinkActive) == 1 }

var thinkFrames = []string{
	"Drea is thinking",
	"Drea is thinking ·",
	"Drea is thinking ··",
	"Drea is thinking ···",
}

func (u *UI) thinkLoop(stop, done chan struct{}) {
	defer close(done)
	ticker := time.NewTicker(250 * time.Millisecond)
	defer ticker.Stop()
	i := 0
	for {
		u.drawThinking(thinkFrames[i%len(thinkFrames)])
		i++
		select {
		case <-stop:
			u.clearThinking()
			return
		case <-ticker.C:
		}
	}
}

func (u *UI) drawThinking(text string) {
	if !u.tty {
		return
	}
	width := termWidth()
	line := u.paint(dim, text)
	if len([]rune(line)) > width-1 {
		line = string([]rune(line)[:width-1])
	}
	fmt.Fprintf(u.out, "\r\033[K%s", line)
	u.setLineStart(true)
}

func (u *UI) clearThinking() {
	if !u.tty {
		return
	}
	fmt.Fprint(u.out, "\r\033[K")
	u.setLineStart(true)
}

func noColor() bool {
	_, ok := os.LookupEnv("NO_COLOR")
	return ok
}

func isTTY() bool {
	fi, err := os.Stdout.Stat()
	if err != nil {
		return false
	}
	return fi.Mode()&os.ModeCharDevice != 0
}

func (u *UI) paint(code, s string) string {
	if !u.colour {
		return s
	}
	return code + s + reset
}

// Banner prints the startup banner with the active model, workdir and the
// current auto-approve categories.
func (u *UI) Banner(model, workdir string, autoWrite, autoGit, autoRun bool) {
	width := termWidth()
	if width < 40 {
		width = 40
	}
	title := " drea · minimalist agent harness "
	if len([]rune(title)) < width {
		pad := width - len([]rune(title))
		left := pad / 2
		right := pad - left
		title = strings.Repeat("─", left) + title + strings.Repeat("─", right)
	}
	fmt.Fprintln(u.out, u.paint(bold+magenta, title))
	fmt.Fprintln(u.out, u.paint(dim, "  model   ")+model)
	fmt.Fprintln(u.out, u.paint(dim, "  workdir ")+workdir)
	fmt.Fprintln(u.out, u.paint(dim, "  safety  ")+u.paint(modeColor(autoWrite, autoGit, autoRun), safetyMode(autoWrite, autoGit, autoRun)))
	fmt.Fprintln(u.out)
	u.SetStatusWorkdir(workdir)
	u.setLineStart(true)
}

// safetyMode renders a human-readable summary of which auto-approve categories
// are active.
func safetyMode(autoWrite, autoGit, autoRun bool) string {
	if !autoWrite && !autoGit && !autoRun {
		return "confirm before writes, git ops and commands"
	}
	var cats []string
	if autoWrite {
		cats = append(cats, "write")
	}
	if autoGit {
		cats = append(cats, "git")
	}
	if autoRun {
		cats = append(cats, "run")
	}
	return "auto-approve: " + strings.Join(cats, ", ")
}

// modeColor returns the colour used for the safety-mode line. When any
// category is auto-approved, use yellow as a cautionary signal.
func modeColor(autoWrite, autoGit, autoRun bool) string {
	if autoWrite || autoGit || autoRun {
		return yellow
	}
	return dim
}

// Assistant streams a chunk of assistant text. The text originates from the
// model and may contain control or escape sequences (for example when it
// echoes a test string of random bytes); those are stripped so model output
// can never move the cursor, inject colour or otherwise corrupt the terminal.
// Newlines and tabs are preserved. State is carried across chunks so an escape
// sequence split over a chunk boundary is still neutralised.
func (u *UI) Assistant(chunk string) {
	clean := u.afilt.feed(chunk)
	fmt.Fprint(u.out, clean)
	if strings.HasSuffix(chunk, "\n") {
		u.setLineStart(true)
	} else {
		u.setLineStart(false)
	}
}

// AssistantHeader prints the label shown before assistant output and resets the
// streaming filter for the new block of assistant text.
func (u *UI) AssistantHeader() {
	u.afilt = streamFilter{}
	u.ensureLineStart()
	fmt.Fprintln(u.out, u.paint(bold+cyan, "╭─ assistant"))
	u.setLineStart(true)
}

// Println writes a line of output, first hiding the status bar so the line is
// not appended to it, then restoring the bar.
func (u *UI) Println(s string) {
	if u.StatusShown() {
		u.HideStatus()
		fmt.Fprintln(u.out, s)
		u.ShowStatus()
		u.setLineStart(true)
		return
	}
	fmt.Fprintln(u.out, s)
	u.setLineStart(true)
}

// ToolCall reports a tool invocation the model requested. The summary comes
// from model-supplied arguments, so it is sanitised to a single safe line.
func (u *UI) ToolCall(name, summary string) {
	u.ensureLineStart()
	label := u.paint(bold+blue, "├─→ "+name)
	if summary != "" {
		label += " " + u.paint(dim, sanitizeLine(summary, 4))
	}
	fmt.Fprintln(u.out, label)
	u.setLineStart(true)
}

// ToolResult prints a short, indented preview of a tool result.
func (u *UI) ToolResult(text string, isErr bool) {
	preview := truncate(text, 12, 2000)
	code := dim
	if isErr {
		code = red
	}
	for _, line := range strings.Split(preview, "\n") {
		fmt.Fprintln(u.out, u.paint(code, "│  "+sanitizeLine(line, 4)))
	}
	u.setLineStart(true)
}

// Diff prints a unified diff, colouring added lines green and removed red.
func (u *UI) Diff(text string) {
	if text == "" {
		return
	}
	for _, line := range strings.Split(text, "\n") {
		code := dim
		if strings.HasPrefix(line, "+") {
			code = green
		} else if strings.HasPrefix(line, "-") {
			code = red
		}
		fmt.Fprintln(u.out, "│  "+u.paint(code, sanitizeLine(line, 4)))
	}
	u.setLineStart(true)
}

// LiveView renders a rolling, in-place tail of streaming text: only the last
// maxLines lines are shown and the block is redrawn as new text arrives. On a
// non-TTY writer it renders nothing (the final result is printed elsewhere).
type LiveView struct {
	u        *UI
	title    string
	maxLines int
	prev     int
	started  bool
	// wrap, when true, splits long logical lines across multiple physical rows
	// instead of clipping them. The caller must use the wrapped variant for
	// content where the full text must remain visible (e.g. reasoning).
	wrap bool
}

// LiveView begins a live tail view with the given title. Long lines are clipped
// so each logical line occupies exactly one physical row; this keeps the
// cursor-up redraw count exact for the file-write tail.
func (u *UI) LiveView(title string, maxLines int) *LiveView {
	return &LiveView{u: u, title: title, maxLines: maxLines}
}

// LiveViewWrapped begins a live tail view that wraps long logical lines onto
// multiple physical rows. Use this when the user needs to see the full text
// (e.g. chain-of-thought reasoning).
func (u *UI) LiveViewWrapped(title string, maxLines int) *LiveView {
	return &LiveView{u: u, title: title, maxLines: maxLines, wrap: true}
}

// livePrefix precedes each rendered tail line; livePrefixWidth is its width in
// terminal cells (box-drawing bar and two spaces).
const (
	livePrefix      = "│  "
	livePrefixWidth = 3
)

// Update redraws the view with the last maxLines lines of full.
//
// Each rendered line is expanded (tabs) and stripped of cursor-moving control
// characters. In the default (non-wrapped) mode long lines are clipped so each
// logical line occupies exactly one physical row, keeping the cursor-up count
// exact. In wrapped mode logical lines are split across as many physical rows
// as needed and the total row count is tracked for redraw.
func (lv *LiveView) Update(full string) {
	if !lv.u.tty {
		return
	}
	width := termWidth()
	body := lastLines(full, lv.maxLines)

	if !lv.started {
		lv.u.ensureLineStart()
		header := clip(sanitizeLine("┄ "+lv.title, 4), width-3)
		fmt.Fprintln(lv.u.out, lv.u.paint(dim, "  "+header))
		lv.started = true
	} else if lv.prev > 0 {
		fmt.Fprintf(lv.u.out, "\033[%dA\r\033[J", lv.prev)
	}

	// Leave a one-column margin so a full-width line cannot trigger the
	// terminal's phantom auto-wrap, which would break the redraw count.
	avail := width - livePrefixWidth - 1
	if avail < 1 {
		avail = 1
	}
	totalRows := 0
	for _, ln := range body {
		ln = sanitizeLine(ln, 4)
		if lv.wrap {
			for _, row := range wrapLine(ln, avail) {
				fmt.Fprintln(lv.u.out, lv.u.paint(dim, livePrefix+row))
				totalRows++
			}
		} else {
			fmt.Fprintln(lv.u.out, lv.u.paint(dim, livePrefix+clip(ln, avail)))
			totalRows++
		}
	}
	lv.prev = totalRows
	lv.u.setLineStart(true)
}

// Close finishes the view, leaving the last rendered tail in place.
func (lv *LiveView) Close() {}

func lastLines(s string, n int) []string {
	lines := strings.Split(strings.TrimRight(s, "\n"), "\n")
	if len(lines) > n {
		lines = lines[len(lines)-n:]
	}
	return lines
}

// sanitizeLine makes a single line safe to print: tabs are expanded to the next
// tab stop, and carriage returns, escape sequences and other control characters
// (which could move the cursor or inject colour) are dropped. Newlines are not
// expected — callers split on them first.
func sanitizeLine(s string, tabWidth int) string {
	if tabWidth < 1 {
		tabWidth = 4
	}
	var b strings.Builder
	col := 0
	rs := []rune(s)
	for i := 0; i < len(rs); i++ {
		r := rs[i]
		if r == 0x1b { // ESC: drop the whole escape sequence, not just ESC.
			if i+1 < len(rs) && rs[i+1] == '[' {
				i += 2
				for i < len(rs) && !(rs[i] >= 0x40 && rs[i] <= 0x7e) {
					i++
				}
			} else {
				i++ // two-character escape (ESC + one byte)
			}
			continue
		}
		switch {
		case r == '\t':
			n := tabWidth - col%tabWidth
			for k := 0; k < n; k++ {
				b.WriteByte(' ')
			}
			col += n
		case r < 0x20 || r == 0x7f:
			// drop CR/LF and other C0/DEL control characters.
		default:
			b.WriteRune(r)
			col++
		}
	}
	return b.String()
}

// streamFilter sanitises a stream of text delivered in arbitrary chunks. It
// drops escape sequences and other control characters (which could move the
// cursor or inject colour) while preserving newlines and tabs. Escape-sequence
// state is retained between feed calls so a sequence split across a chunk
// boundary is still neutralised.
type streamFilter struct {
	inEsc bool // seen ESC, awaiting the next byte
	inCSI bool // inside a CSI (ESC [ …) sequence, consuming until its final byte
}

func (f *streamFilter) feed(s string) string {
	var b strings.Builder
	for _, r := range s {
		switch {
		case f.inCSI:
			if r >= 0x40 && r <= 0x7e { // final byte ends the CSI sequence
				f.inCSI = false
			}
		case f.inEsc:
			f.inEsc = false
			if r == '[' {
				f.inCSI = true
			}
			// otherwise a two-character escape (ESC + one byte): drop both.
		case r == 0x1b: // ESC
			f.inEsc = true
		case r == '\n' || r == '\t':
			b.WriteRune(r)
		case r < 0x20 || r == 0x7f:
			// drop CR and other C0/DEL control characters.
		default:
			b.WriteRune(r)
		}
	}
	return b.String()
}

// wrapLine splits s into chunks of at most width runes. Empty strings produce
// one empty chunk so blank lines remain visible.
func wrapLine(s string, width int) []string {
	if width <= 0 {
		return []string{s}
	}
	rs := []rune(s)
	var out []string
	for len(rs) > width {
		out = append(out, string(rs[:width]))
		rs = rs[width:]
	}
	if len(rs) > 0 || len(out) == 0 {
		out = append(out, string(rs))
	}
	return out
}

// clip shortens s to at most max terminal cells (approximated by rune count),
// appending an ellipsis when it truncates so the result never exceeds max.
func clip(s string, max int) string {
	if max <= 0 {
		return ""
	}
	r := []rune(s)
	if len(r) <= max {
		return s
	}
	if max == 1 {
		return "…"
	}
	return string(r[:max-1]) + "…"
}

// termWidth returns the current terminal width in columns, falling back to 80
// when it cannot be determined (e.g. output is not a terminal).
func termWidth() int {
	var ws struct{ Row, Col, Xpixel, Ypixel uint16 }
	_, _, errno := syscall.Syscall(
		syscall.SYS_IOCTL,
		os.Stdout.Fd(),
		uintptr(syscall.TIOCGWINSZ),
		uintptr(unsafe.Pointer(&ws)),
	)
	if errno != 0 || ws.Col == 0 {
		return 80
	}
	return int(ws.Col)
}

// Info, Warn and Error print status lines.
func (u *UI) Info(s string)  { u.Println(u.paint(dim, s)) }
func (u *UI) Warn(s string)  { u.Println(u.paint(yellow, s)) }
func (u *UI) Error(s string) { u.Println(u.paint(red, "error: ") + s) }
func (u *UI) Done(s string)  { u.Println(u.paint(green, s)) }

// SetMultiline toggles the input mode. In single-line mode (the default) a
// single Enter sends the message; in multi-line mode Enter starts a new line
// and Ctrl-D (end of input) sends it, so blank lines are ordinary content and
// pasted code arrives as one message. The prompt glyph reflects the active
// mode and, in multi-line mode, carries a hint about how to send.
func (u *UI) SetMultiline(on bool) {
	u.multiline = on
}

// Multiline reports whether multi-line input mode is active.
func (u *UI) Multiline() bool { return u.multiline }

// Prompt writes the interactive input prompt (no newline). In multi-line mode
// the prompt carries a hint that Ctrl-D sends the message, so the user always
// sees how to submit multi-line input.
func (u *UI) Prompt() {
	u.ensureLineStart()
	if u.multiline {
		fmt.Fprint(u.out, u.paint(bold+green, "│─╮ ")+u.paint(dim, "(Ctrl-D to send) "))
	} else {
		fmt.Fprint(u.out, u.paint(bold+green, "│─› "))
	}
	u.setLineStart(false)
}

// ReadInput reads user input. In single-line mode (the default) the first Enter
// sends the message. In multi-line mode Enter starts a new line and Ctrl-D (end
// of input) sends the message, so blank lines are ordinary content and pasted
// code arrives as one message. io.EOF is returned when Ctrl-D is pressed on an
// empty buffer, so callers can exit cleanly.
func (u *UI) ReadInput() (string, error) {
	if !u.multiline {
		line, err := u.in.ReadString('\n')
		if err != nil {
			if errors.Is(err, io.EOF) && strings.TrimRight(line, "\r\n") != "" {
				return strings.TrimRight(line, "\r\n"), nil
			}
			return strings.TrimRight(line, "\r\n"), err
		}
		return strings.TrimRight(line, "\r\n"), nil
	}
	var b strings.Builder
	for {
		line, err := u.in.ReadString('\n')
		if err != nil {
			b.WriteString(line)
			if b.Len() > 0 {
				return strings.TrimRight(b.String(), "\r\n"), nil
			}
			return "", err
		}
		b.WriteString(line)
	}
}

// Confirm asks a yes/no question, defaulting to no. It returns true only on an
// explicit affirmative answer.
func (u *UI) Confirm(question string) bool {
	u.ensureLineStart()
	fmt.Fprint(u.out, u.paint(yellow, question)+u.paint(dim, " [y/N] "))
	u.setLineStart(false)
	line, err := u.in.ReadString('\n')
	if err != nil {
		return false
	}
	switch strings.ToLower(strings.TrimSpace(line)) {
	case "y", "yes":
		return true
	}
	return false
}

func (u *UI) setLineStart(v bool) {
	u.lineMu.Lock()
	u.lineStart = v
	u.lineMu.Unlock()
}

func (u *UI) ensureLineStart() {
	u.lineMu.Lock()
	start := u.lineStart
	u.lineMu.Unlock()
	if !start {
		fmt.Fprintln(u.out)
		u.setLineStart(true)
	}
}

func truncate(s string, maxLines, maxChars int) string {
	if len(s) > maxChars {
		s = s[:maxChars] + "\n… (truncated)"
	}
	lines := strings.Split(s, "\n")
	if len(lines) > maxLines {
		lines = append(lines[:maxLines], fmt.Sprintf("… (%d more lines)", len(lines)-maxLines))
	}
	return strings.Join(lines, "\n")
}
