package cmd

import (
	"encoding/json"
	"fmt"

	"github.com/criblio/scope/internal"
	"github.com/criblio/scope/rules"
	"github.com/criblio/scope/util"
	"github.com/rs/zerolog/log"
	"github.com/spf13/cobra"
)

/* Args Matrix (X disallows)
 *                 cribldest metricformat metricdest eventdest nobreaker authtoken verbosity payloads loglevel librarypath userconfig coredump backtrace rootdir add remove json sourceid arg
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
 * rootdir                                                                                                                                               -
 * add                                                                                                                                                           -   X
 * remove                                                                                                                                                        X   -
 * json                                                                                                                                                                      -
 * sourceid                                                                                                                                                                      -
 * arg                                                                                                                                                                                    -
 */

// rulesCmd represents the rules command
var rulesCmd = &cobra.Command{
	Use:   "rules [flags]",
	Short: "View or modify system-wide AppScope rules",
	Long:  `View or modify system-wide AppScope rules to automatically scope a set of processes. You can add or remove a single process at a time.`,
	Example: `  scope rules
  scope rules --rootdir /path/to/host/root --json
  scope rules --add nginx
  scope rules --add nginx < scope.yml
  scope rules --add java --arg myServer
  scope rules --add firefox --rootdir /path/to/host/root
  scope rules --remove chromium`,
	Args: cobra.NoArgs,
	Run: func(cmd *cobra.Command, args []string) {
		internal.InitConfig()
		rc.Rootdir, _ = cmd.Flags().GetString("rootdir")
		jsonOut, _ := cmd.Flags().GetBool("json")
		addProc, _ := cmd.Flags().GetString("add")
		remProc, _ := cmd.Flags().GetString("remove")
		sourceid, _ := cmd.Flags().GetString("sourceid")
		procArg, _ := cmd.Flags().GetString("arg")
		unixPath, _ := cmd.Flags().GetString("unixpath")

		// Disallow bad argument combinations (see Arg Matrix at top of file)
		if addProc == "" && remProc == "" {
			if rc.CriblDest != "" || rc.MetricsDest != "" || rc.EventsDest != "" || rc.UserConfig != "" ||
				cmd.Flags().Lookup("metricformat").Changed || rc.NoBreaker || rc.AuthToken != "" || rc.Payloads ||
				rc.Backtrace || rc.Loglevel != "" || rc.Coredump || cmd.Flags().Lookup("verbosity").Changed {
				helpErrAndExit(cmd, "The rules command without --add or --remove options, only supports the --json flag")
			}
		}
		if addProc != "" && remProc != "" {
			helpErrAndExit(cmd, "Cannot specify --add and --remove")
		}
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
		} else if rc.Loglevel != "" && rc.UserConfig != "" {
			helpErrAndExit(cmd, "Cannot specify --loglevel and --userconfig")
		} else if rc.Backtrace && rc.UserConfig != "" {
			helpErrAndExit(cmd, "Cannot specify --backtrace and --userconfig")
		} else if rc.Coredump && rc.UserConfig != "" {
			helpErrAndExit(cmd, "Cannot specify --coredump and --userconfig")
		}

		// Retrieve an existing rules file
		raw, rulesFile, err := rules.Retrieve(rc.Rootdir)
		if err != nil {
			util.ErrAndExit("Rules failure: %v", err)
		}

		// In the case that no --add argument or --remove argument was provided
		// just print the rules file
		if addProc == "" && remProc == "" {
			if len(raw) == 0 || len(rulesFile.Allow) == 0 {
				util.ErrAndExit("Empty rules file")
			}

			if jsonOut {
				// Marshal to json
				jsonData, err := json.Marshal(rulesFile)
				if err != nil {
					util.ErrAndExit("Error marshaling JSON: %v", err)
				}
				fmt.Println(string(jsonData))
			} else {
				content := string(raw)
				fmt.Println(content)
			}

			return
		}

		// Below operations require root
		// Validate user has root permissions
		if err := util.UserVerifyRootPerm(); err != nil {
			log.Error().Err(err)
			util.ErrAndExit("modifying the scope rules requires administrator privileges")
		}

		// Add a process to; or remove a process from the scope rules
		if addProc != "" {
			if err = rules.Add(rulesFile, addProc, procArg, sourceid, rc.Rootdir, rc, unixPath); err != nil {
				util.ErrAndExit("Rules failure: %v", err)
			}
		} else if remProc != "" {
			if err = rules.Remove(rulesFile, remProc, procArg, sourceid, rc.Rootdir, rc); err != nil {
				util.ErrAndExit("Rules failure: %v", err)
			}
		}

		return
	},
}

func init() {
	runCmdFlags(rulesCmd, rc)
	rulesCmd.Flags().StringP("rootdir", "R", "", "Path to root filesystem of target namespace")
	rulesCmd.Flags().String("add", "", "Add an entry to the global rules")
	rulesCmd.Flags().String("remove", "", "Remove an entry from the global rules")
	rulesCmd.Flags().BoolP("json", "j", false, "Output as newline delimited JSON")
	rulesCmd.Flags().String("sourceid", "", "Source identifier for a rules entry")
	rulesCmd.Flags().String("arg", "", "Argument to the command to be added to the rules")
	rulesCmd.Flags().String("unixpath", "", "Path to the Unix socket")
	RootCmd.AddCommand(rulesCmd)
}
