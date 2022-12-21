package run

import (
	"os"
	"path/filepath"
	"time"

	"github.com/criblio/scope/libscope"
	"github.com/criblio/scope/loader"
	"github.com/criblio/scope/util"
)

// Config represents options to change how we run scope
type Config struct {
	WorkDir       string
	Verbosity     int
	Payloads      bool
	MetricsDest   string
	EventsDest    string
	MetricsFormat string
	CriblDest     string
	Subprocess    bool
	Loglevel      string
	LogDest       string
	LibraryPath   string
	NoBreaker     bool
	AuthToken     string
	UserConfig    string
	CommandDir    string

	now func() time.Time
	sc  *libscope.ScopeConfig
}

// Run executes a scoped command
func (rc *Config) Run(args []string) {
	env := os.Environ()

	// Disable detection of a scope filter file with this command
	env = append(env, "SCOPE_FILTER=false")

	// Disable cribl event breaker with this command
	if rc.NoBreaker {
		env = append(env, "SCOPE_CRIBL_NO_BREAKER=true")
	}

	// Normal operation, create a directory for this run.
	// Directory contains scope.yml which is configured to output to that
	// directory and has a command directory configured in that directory.
	rc.setupWorkDir(args, false)
	env = append(env, "SCOPE_CONF_PATH="+filepath.Join(rc.WorkDir, "scope.yml"))

	// Handle custom library path
	if len(rc.LibraryPath) > 0 {
		if !util.CheckDirExists(rc.LibraryPath) {
			util.ErrAndExit("Library Path does not exist: \"%s\"", rc.LibraryPath)
		}
		args = append([]string{"-f", rc.LibraryPath}, args...)
	}

	ld := loader.New()
	if !rc.Subprocess {
		ld.Passthrough(args, env)
	}
	ld.PassthroughSubProc(args, env)
}
