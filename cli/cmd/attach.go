package cmd

import (
	"encoding/json"
	"fmt"
	"time"

	"github.com/criblio/scope/inspect"
	"github.com/criblio/scope/internal"
	"github.com/criblio/scope/ipc"
	"github.com/criblio/scope/util"
	"github.com/spf13/cobra"
)

/* Args Matrix (X disallows)
 *                 cribldest metricformat metricdest eventdest nobreaker authtoken verbosity payloads loglevel librarypath userconfig coredump backtrace
 * cribldest       -                      X          X                                                                     X
 * metricformat              -                                                                                             X
 * metricdest      X                      -                                                                                X
 * eventdest       X                                 -                                                                     X
 * nobreaker                                                   -                                                           X
 * authtoken                                                             -                                                 X
 * verbosity                                                                       -                                       X
 * payloads                                                                                  -                             X
 * loglevel                                                                                           -                    X
 * librarypath                                                                                                 -
 * userconfig      X         X            X          X         X         X         X         X        X                    -          X         X
 * coredump     																										   X		  -
 * backtrace  																											   X				    -
 */

// attachCmd represents the attach command
var attachCmd = &cobra.Command{
	Use:   "attach [flags] PID | <process_name>",
	Short: "Scope a currently-running process",
	Long: `Scopes a currently-running process identified by PID or <process_name>.

The --*dest flags accept file names like /tmp/scope.log or URLs like file:///tmp/scope.log. They may also
be set to sockets with unix:///var/run/mysock, tcp://hostname:port, udp://hostname:port, or tls://hostname:port.`,
	Example: `  scope attach 1000
  scope attach firefox 
  scope attach top < scope.yml
  scope attach --rootdir /path/to/host/root firefox 
  scope attach --rootdir /path/to/host/root/proc/<hostpid>/root 1000
  scope attach --payloads 2000`,
	Args: cobra.ExactArgs(1),
	Run: func(cmd *cobra.Command, args []string) {
		internal.InitConfig()
		rc.Rootdir, _ = cmd.Flags().GetString("rootdir")
		inspectFlag, _ := cmd.Flags().GetBool("inspect")
		jsonOut, _ := cmd.Flags().GetBool("json")

		// Disallow bad argument combinations (see Arg Matrix at top of file)
		if rc.CriblDest != "" && rc.MetricsDest != "" {
			helpErrAndExit(cmd, "Cannot specify --cribldest and --metricdest")
		} else if rc.CriblDest != "" && rc.EventsDest != "" {
			helpErrAndExit(cmd, "Cannot specify --cribldest and --eventdest")
		} else if rc.CriblDest != "" && rc.UserConfig != "" {
			helpErrAndExit(cmd, "Cannot specify --cribldest and --userconfig")
		} else if cmd.Flags().Lookup("metricformat").Changed && rc.UserConfig != "" {
			helpErrAndExit(cmd, "Cannot specify --metricformat and --userconfig")
		} else if rc.EventsDest != "" && rc.UserConfig != "" {
			helpErrAndExit(cmd, "Cannot specify --eventdest and --userconfig")
		} else if rc.NoBreaker && rc.UserConfig != "" {
			helpErrAndExit(cmd, "Cannot specify --nobreaker and --userconfig")
		} else if rc.AuthToken != "" && rc.UserConfig != "" {
			helpErrAndExit(cmd, "Cannot specify --authtoken and --userconfig")
		} else if cmd.Flags().Lookup("verbosity").Changed && rc.UserConfig != "" {
			helpErrAndExit(cmd, "Cannot specify --verbosity and --userconfig")
		} else if rc.Payloads && rc.UserConfig != "" {
			helpErrAndExit(cmd, "Cannot specify --payloads and --userconfig")
		} else if cmd.Flags().Lookup("payloadsdest").Changed && rc.UserConfig != "" {
			helpErrAndExit(cmd, "Cannot specify --payloadsdest and --userconfig")
		} else if rc.Loglevel != "" && rc.UserConfig != "" {
			helpErrAndExit(cmd, "Cannot specify --loglevel and --userconfig")
		} else if rc.Backtrace && rc.UserConfig != "" {
			helpErrAndExit(cmd, "Cannot specify --backtrace and --userconfig")
		} else if rc.Coredump && rc.UserConfig != "" {
			helpErrAndExit(cmd, "Cannot specify --coredump and --userconfig")
		}

		id := args[0] // The attach command ensures we have an argument

		procs, err := util.HandleInputArg(id, "", rc.Rootdir, true, true, true, false)
		if err != nil {
			util.ErrAndExit("Attach failure: %v", err)
		}

		if len(procs) == 0 {
			util.ErrAndExit("Attach failure: no matching processes found")
		}

		pid := procs[0].Pid // we told HandleInputArg above that we wanted to choose only one proc

		reattach, err := rc.Attach(pid, true)
		if err != nil {
			util.ErrAndExit("Attach failure: %v", err)
		}

		if rc.Rootdir != "" {
			util.Warn("Attaching to process %d", pid)
			util.Warn("It can take up to 1 minute to attach to a process in a parent namespace")
		}

		if inspectFlag {
			pidCtx := &ipc.IpcPidCtx{
				PrefixPath: rc.Rootdir,
				Pid:        pid,
			}

			// Simple approach for now
			// Later: check for attach then resume
			if rc.Rootdir != "" {
				time.Sleep(60 * time.Second) // parent attach uses cron (<60s)
			}
			if reattach {
				time.Sleep(10 * time.Second) // reattach uses dynamic config (<10s)
			}
			time.Sleep(2 * time.Second) // give the command time to run

			iout, _, err := inspect.InspectProcess(*pidCtx)
			if err != nil {
				util.ErrAndExit("Inspect PID fails: %v", err)
			}

			if jsonOut {
				// Print the json object without any pretty printing
				cfg, err := json.Marshal(iout)
				if err != nil {
					util.ErrAndExit("Error creating json object: %v", err)
				}
				fmt.Println(string(cfg))
			} else {
				// Print the json in a pretty format
				cfg, err := json.MarshalIndent(iout, "", "   ")
				if err != nil {
					util.ErrAndExit("Error creating json array: %v", err)
				}
				fmt.Println(string(cfg))
			}
		}
	},
}

func init() {
	runCmdFlags(attachCmd, rc)
	attachCmd.Flags().BoolP("inspect", "i", false, "Inspect the process after the attach is complete")
	attachCmd.Flags().StringP("rootdir", "R", "", "Path to root filesystem of target namespace")
	attachCmd.Flags().BoolP("json", "j", false, "Output as newline delimited JSON")
	RootCmd.AddCommand(attachCmd)
}
