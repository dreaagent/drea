// Package settings persists preferences to a JSON file under the user's config
// directory. The API key is stored in its own 0600 file (never in settings.json
// or the session transcript) so `drea` works without exporting it every time.
package settings

import (
	"encoding/json"
	"io"
	"os"
	"path/filepath"
	"strings"
)

// Settings holds the persisted, non-secret preferences. The API key is kept
// separately (see KeyFile).
type Settings struct {
	BaseURL string `json:"base_url,omitempty"`
	Model   string `json:"model,omitempty"`
	// Verify is an optional command run to check the workspace after a task.
	Verify string `json:"verify,omitempty"`
	// VerifyAttempts caps the self-correction rounds after a failed verify.
	VerifyAttempts int `json:"verify_attempts,omitempty"`
	// Checkpoint commits the workspace before each task so it can be rolled back.
	Checkpoint bool `json:"checkpoint,omitempty"`
	// ContextTokens is the prompt-size budget above which history is compacted.
	ContextTokens int `json:"context_tokens,omitempty"`
	// JSONFormat selects the response_format variant sent to the endpoint.
	// Valid values: json_schema or json_object.
	JSONFormat string `json:"json_format,omitempty"`
	// TopP is the nucleus-sampling probability forwarded to the endpoint.
	TopP float64 `json:"top_p,omitempty"`
	// ReasoningEffort is the reasoning effort level forwarded to the endpoint.
	ReasoningEffort string `json:"reasoning_effort,omitempty"`
	// AllowCommands and DenyCommands are regex patterns for the run_command
	// policy (see the config package). No secrets are ever stored here.
	AllowCommands []string `json:"allow_commands,omitempty"`
	DenyCommands  []string `json:"deny_commands,omitempty"`
	// PromptCacheKey is a non-standard "prompt_cache_key" sent with requests.
	PromptCacheKey string `json:"prompt_cache_key,omitempty"`
	// AutoWrite skips confirmation for write_file, edit_file and apply_patch.
	AutoWrite bool `json:"auto_write,omitempty"`
	// AutoGit skips confirmation for git_init, git_commit and git_rollback.
	AutoGit bool `json:"auto_git,omitempty"`
	// AutoRun skips confirmation for run_command.
	AutoRun bool `json:"auto_run,omitempty"`
	// ReplyChain is the maximum number of consecutive assistant replies
	// auto-continued in interactive mode before pausing for user input.
	ReplyChain int `json:"reply_chain,omitempty"`
}

// Path returns the settings file location (…/drea/settings.json under the
// user's config dir, e.g. ~/.config/drea/settings.json on Linux).
func Path() (string, error) {
	dir, err := os.UserConfigDir()
	if err != nil {
		return "", err
	}
	return filepath.Join(dir, "drea", "settings.json"), nil
}

// Load reads the settings file. A missing file is not an error: it returns an
// empty Settings and ok=false so callers can tell "unset" from "set to empty".
func Load() (Settings, bool, error) {
	var s Settings
	p, err := Path()
	if err != nil {
		return s, false, err
	}
	f, err := os.Open(p)
	if err != nil {
		if os.IsNotExist(err) {
			return s, false, nil
		}
		return s, false, err
	}
	defer f.Close()

	// The settings file is a couple of short strings; refuse anything absurd
	// rather than reading an unbounded amount into memory.
	data, err := io.ReadAll(io.LimitReader(f, maxSize))
	if err != nil {
		return s, false, err
	}
	if err := json.Unmarshal(data, &s); err != nil {
		return s, false, err
	}
	return s, true, nil
}

// maxSize bounds how much of the settings file is read.
const maxSize = 64 * 1024

// Save writes the settings file, creating the config directory as needed. The
// file is written with 0600 permissions.
func Save(s Settings) (string, error) {
	p, err := Path()
	if err != nil {
		return "", err
	}
	if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
		return "", err
	}
	data, err := json.MarshalIndent(s, "", "  ")
	if err != nil {
		return "", err
	}
	data = append(data, '\n')
	if err := os.WriteFile(p, data, 0o600); err != nil {
		return "", err
	}
	return p, nil
}

// KeyFile is the file holding the API key (…/drea/key under the user's config
// dir, e.g. ~/.config/drea/key on Linux). It is stored separately from
// settings.json so secrets are never mixed with preferences or transcripts.
func KeyFile() (string, error) {
	dir, err := os.UserConfigDir()
	if err != nil {
		return "", err
	}
	return filepath.Join(dir, "drea", "key"), nil
}

// LoadKey reads the API key from the key file. A missing file is not an error:
// it returns ok=false so callers can tell "no stored key" from other failures.
// The key is trimmed of surrounding whitespace (a trailing newline written by
// echo is the normal case).
func LoadKey() (string, bool, error) {
	p, err := KeyFile()
	if err != nil {
		return "", false, err
	}
	data, err := os.ReadFile(p)
	if err != nil {
		if os.IsNotExist(err) {
			return "", false, nil
		}
		return "", false, err
	}
	k := strings.TrimSpace(string(data))
	return k, k != "", nil
}

// SaveKey writes the API key to the key file, creating the config directory as
// needed. The file is written with 0600 permissions, like the settings file.
// Saving an empty key removes the file.
func SaveKey(key string) (string, error) {
	p, err := KeyFile()
	if err != nil {
		return "", err
	}
	if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
		return "", err
	}
	key = strings.TrimSpace(key)
	if key == "" {
		if err := os.Remove(p); err != nil && !os.IsNotExist(err) {
			return "", err
		}
		return p, nil
	}
	if err := os.WriteFile(p, []byte(key+"\n"), 0o600); err != nil {
		return "", err
	}
	return p, nil
}
