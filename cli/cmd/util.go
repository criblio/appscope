package cmd

import (
	"bufio"
	"fmt"
	"os"

	"github.com/criblio/scope/history"
	"github.com/criblio/scope/run"
	"github.com/criblio/scope/util"
	"github.com/spf13/cobra"
)

// sessionByID returns a session by ID, or if -1 (not set) returns last session
func sessionByID(id int) history.SessionList {
	var sessions history.SessionList
	if id == -1 {
		sessions = history.GetSessions().Last(1)
	} else {
		sessions = history.GetSessions().ID(id)
	}
	sessionCount := len(sessions)
	if sessionCount != 1 {
		util.ErrAndExit("Session not found. Session count: %d", sessionCount)
	}
	return sessions
}

func promptClean(sl history.SessionList) {
	fmt.Println("Invalid session, likely an invalid command was scoped / a session file was modified / a container session has ended. Would you like to delete this session? (y/yes)")
	in := bufio.NewScanner(os.Stdin)
	in.Scan()
	response := in.Text()
	if !(response == "y" || response == "yes") {
		fmt.Println("No action taken")
		os.Exit(0)
	}
	sl.Remove()
	fmt.Println("Deleted session.")
	os.Exit(0)
}

func helpErrAndExit(cmd *cobra.Command, errText string) {
	cmd.Help()
	fmt.Printf("\nerror: %s\n", errText)
	os.Exit(1)
}

func metricAndEventDestFlags(cmd *cobra.Command, rc *run.Config) {
	cmd.Flags().StringVarP(&rc.CriblDest, "cribldest", "c", "", "Set Cribl destination for metrics & events (host:port defaults to tls://)")
	cmd.Flags().StringVar(&rc.MetricsFormat, "metricformat", "ndjson", "Set format of metrics output (statsd|ndjson)")
	cmd.Flags().StringVar(&rc.MetricsPrefix, "metricprefix", "", "Set prefix for StatsD metrics, ignored if metric format isn't statsd")
	cmd.Flags().StringVarP(&rc.MetricsDest, "metricdest", "m", "", "Set destination for metrics (host:port defaults to tls://)")
	cmd.Flags().StringVarP(&rc.EventsDest, "eventdest", "e", "", "Set destination for events (host:port defaults to tls://)")
	cmd.Flags().BoolVarP(&rc.NoBreaker, "nobreaker", "n", false, "Set Cribl to not break streams into events.")
	cmd.Flags().StringVarP(&rc.AuthToken, "authtoken", "a", "", "Set AuthToken for Cribl")
}

func runCmdFlags(cmd *cobra.Command, rc *run.Config) {
	cmd.Flags().IntVarP(&rc.Verbosity, "verbosity", "v", 4, "Set scope metric verbosity")
	cmd.Flags().BoolVarP(&rc.Payloads, "payloads", "p", false, "Capture payloads of network transactions")
	cmd.Flags().StringVarP(&rc.PayloadsDest, "payloadsdest", "", "dir", "Set destination for payloads (dir|event)")
	cmd.Flags().StringVar(&rc.Loglevel, "loglevel", "", "Set scope library log level (debug, warning, info, error, none)")
	cmd.Flags().StringVarP(&rc.LibraryPath, "librarypath", "l", "", "Set path for dynamic libraries")
	cmd.Flags().StringVarP(&rc.UserConfig, "userconfig", "u", "", "Scope an application with a user specified config file; overrides all other settings.")
	cmd.Flags().BoolVarP(&rc.Coredump, "coredump", "d", false, "Enable core dump file generation when an application crashes.")
	cmd.Flags().BoolVarP(&rc.Backtrace, "backtrace", "b", false, "Enable backtrace file generation when an application crashes.")
	metricAndEventDestFlags(cmd, rc)
}
