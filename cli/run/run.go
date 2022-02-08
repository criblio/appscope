package run

import (
	"bytes"
	"context"
	"os"
	"os/exec"
	"path/filepath"
	"syscall"
	"time"

	"github.com/criblio/scope/util"
	"github.com/rs/zerolog/log"
	"golang.org/x/sync/errgroup"
)

// Config represents options to change how we run scope
type Config struct {
	WorkDir       string
	Passthrough   bool
	Verbosity     int
	Payloads      bool
	MetricsDest   string
	EventsDest    string
	MetricsFormat string
	CriblDest     string
	Subprocess    bool
	Loglevel      string
	LibraryPath   string
	NoBreaker     bool
	AuthToken     string
	UserConfig    string

	now func() time.Time
	sc  *ScopeConfig
}

func (rc *Config) LiveRun(gctx context.Context, g *errgroup.Group, args []string) func() error {
	return func() error {
		if err := createLdscope(); err != nil {
			util.ErrAndExit("error creating ldscope: %v", err)
		}
		// Normal operational, not passthrough, create directory for this run
		// Directory contains scope.yml which is configured to output to that
		// directory and has a command directory configured in that directory.
		env := os.Environ()
		if rc.NoBreaker {
			env = append(env, "SCOPE_CRIBL_NO_BREAKER=true")
		}
		if !rc.Passthrough {
			rc.setupWorkDir(args, false)
			env = append(env, "SCOPE_CONF_PATH="+filepath.Join(rc.WorkDir, "scope.yml"))
			log.Info().Bool("passthrough", rc.Passthrough).Strs("args", args).Msg("calling syscall.Exec")
		}
		if len(rc.LibraryPath) > 0 {
			// Validate path exists
			if !util.CheckDirExists(rc.LibraryPath) {
				util.ErrAndExit("Library Path does not exist: \"%s\"", rc.LibraryPath)
			}
			// Prepend "-f" [PATH] to args
			args = append([]string{"-f", rc.LibraryPath}, args...)
		}
		if !rc.Subprocess {
			syscall.Exec(ldscopePath(), append([]string{"ldscope"}, args...), env)
		}
		cmd := exec.Command(ldscopePath(), args...)
		cmd.Env = env
		// TODO in live mode, consume stdout and stderr in web view instead?
		// cmd.Stdout = os.Stdout
		// cmd.Stderr = os.Stderr
		var execOut bytes.Buffer
		var execErr bytes.Buffer
		cmd.Stdout = &execOut
		cmd.Stderr = &execErr
		// cmd.Run() // OK to replace with start?
		cmd.Start()
		err := cmd.Process.Release()
		if err != nil {
			util.ErrAndExit("error releasing ldscope: %v", err)
		}
		return nil
	}
}

// Run executes a scoped command
func (rc *Config) Run(args []string) {
	if err := createLdscope(); err != nil {
		util.ErrAndExit("error creating ldscope: %v", err)
	}
	// Normal operational, not passthrough, create directory for this run
	// Directory contains scope.yml which is configured to output to that
	// directory and has a command directory configured in that directory.
	env := os.Environ()
	if rc.NoBreaker {
		env = append(env, "SCOPE_CRIBL_NO_BREAKER=true")
	}
	if !rc.Passthrough {
		rc.setupWorkDir(args, false)
		env = append(env, "SCOPE_CONF_PATH="+filepath.Join(rc.WorkDir, "scope.yml"))
		log.Info().Bool("passthrough", rc.Passthrough).Strs("args", args).Msg("calling syscall.Exec")
	}
	if len(rc.LibraryPath) > 0 {
		// Validate path exists
		if !util.CheckDirExists(rc.LibraryPath) {
			util.ErrAndExit("Library Path does not exist: \"%s\"", rc.LibraryPath)
		}
		// Prepend "-f" [PATH] to args
		args = append([]string{"-f", rc.LibraryPath}, args...)
	}
	if !rc.Subprocess {
		syscall.Exec(ldscopePath(), append([]string{"ldscope"}, args...), env)
	}
	cmd := exec.Command(ldscopePath(), args...)
	cmd.Env = env
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	cmd.Run()
}

// GetScopeConfig returns the ScopeConfig pointer
func (rc *Config) GetScopeConfig() *ScopeConfig {
	return rc.sc
}
