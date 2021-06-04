package cmd

import (
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
		util.ErrAndExit("error expected a single session, saw: %d", sessionCount)
	}
	return sessions
}

func promptClean(sl history.SessionList) {
	fmt.Print("Invalid session, likely an invalid command was scoped. Would you like to delete this session? (default: yes) [y/n] ")
	var response string
	_, err := fmt.Scanf("%s", &response)
	util.CheckErrSprintf(err, "error reading response: %v", err)
	if !(response == "n" || response == "no") {
		sl.Remove()
	}
	os.Exit(0)
}

func helpErrAndExit(cmd *cobra.Command, errText string) {
	cmd.Help()
	fmt.Printf("\nerror: %s\n", errText)
	os.Exit(1)
}

func metricAndEventDestFlags(cmd *cobra.Command, rc *run.Config) {
	cmd.Flags().StringVarP(&rc.CriblDest, "cribldest", "c", "", "Set Cribl destination for metrics & events")
	cmd.Flags().StringVar(&rc.MetricsFormat, "metricformat", "ndjson", "Set format of metrics output (statsd|ndjson)")
	cmd.Flags().StringVarP(&rc.MetricsDest, "metricdest", "m", "", "Set destination for metrics")
	cmd.Flags().StringVarP(&rc.EventsDest, "eventdest", "e", "", "Set destination for events")
}

func runCmdFlags(cmd *cobra.Command, rc *run.Config) {
	cmd.Flags().BoolVar(&rc.Passthrough, "passthrough", false, "Runs ldscope with current environment & no config.")
	cmd.Flags().IntVarP(&rc.Verbosity, "verbosity", "v", 4, "Set scope metric verbosity")
	cmd.Flags().BoolVarP(&rc.Payloads, "payloads", "p", false, "Capture payloads of network transactions")
	cmd.Flags().StringVar(&rc.Loglevel, "loglevel", "", "Set ldscope log level (debug, warning, info, error, none)")
	cmd.Flags().StringVarP(&rc.LibraryPath, "librarypath", "l", "", "Set path for dynamic libraries")
	metricAndEventDestFlags(cmd, rc)
}
