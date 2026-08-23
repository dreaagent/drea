package agent

import (
	"strings"

	"github.com/dreaagent/drea/internal/ui"
)

// reasoningView streams the model's chain-of-thought tokens as a rolling tail
// (last 10 lines) so the user can watch progress while the model works instead
// of a static "thinking" spinner — if it is streaming, the user sees it happen.
//
// Reasoning tokens are display-only: they are never stored in the conversation
// (many models are trained without being handed their own chain-of-thought
// back), so only the screen is affected. When no reasoning content is streamed
// (most endpoints), the view never appears and the existing spinner still runs.
type reasoningView struct {
	ui       *ui.UI
	ensureNL func()
	view     *ui.LiveView
	buf      strings.Builder
	started  bool
}

func newReasoningView(u *ui.UI, ensureNL func()) *reasoningView {
	return &reasoningView{ui: u, ensureNL: ensureNL}
}

// onReasoning feeds one chain-of-thought delta into the live view, creating it
// (and stopping the spinner it replaces) on the first delta.
func (r *reasoningView) onReasoning(delta string) {
	if delta == "" {
		return
	}
	if !r.started {
		r.started = true
		r.ui.StopThinking()
		if r.ensureNL != nil {
			r.ensureNL()
		}
		r.view = r.ui.LiveViewWrapped("reasoning", 10)
	}
	r.buf.WriteString(delta)
	r.view.Update(r.buf.String())
}

// close finalises the view, leaving the last rendered tail visible.
func (r *reasoningView) close() {
	if r.view != nil {
		r.view.Close()
		r.view = nil
	}
}
