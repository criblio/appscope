package cmd

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"os"

	"github.com/criblio/scope/internal"
	"github.com/criblio/scope/libscope"
	"github.com/spf13/cobra"
	"gopkg.in/yaml.v2"
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

// filterCmd represents the filter command
var filterCmd = &cobra.Command{
	Use:   "filter [flags]",
	Short: "Automatically scope a set of processes",
	Long:  `Automatically scope a set of processes by modifying a system-wide AppScope filter. Overrides all other scope sessions.`,
	Example: `  scope filter
  scope filter --rootdir /path/to/host/root --json
  scope filter --add nginx
  scope filter --add nginx < scope.yml
  scope filter --add firefox --rootdir /path/to/host/root
  scope filter --remove chromium`,
	Args: cobra.NoArgs,
	RunE: func(cmd *cobra.Command, args []string) error {
		internal.InitConfig()
		rc.Rootdir, _ = cmd.Flags().GetString("rootdir")
		jsonOut, _ := cmd.Flags().GetBool("json")
		addProc, _ := cmd.Flags().GetString("add")
		remProc, _ := cmd.Flags().GetString("remove")
		// source, _ := cmd.Flags().GetString("source")

		// Disallow bad argument combinations (see Arg Matrix at top of file)
		if addProc == "" && remProc == "" {
			if rc.CriblDest != "" || rc.MetricsDest != "" || rc.EventsDest != "" || rc.UserConfig != "" ||
				cmd.Flags().Lookup("metricformat").Changed || rc.NoBreaker || rc.AuthToken != "" || rc.Payloads ||
				rc.Backtrace || rc.Loglevel != "" || rc.Coredump || cmd.Flags().Lookup("verbosity").Changed {
				helpErrAndExit(cmd, "The filter command without --add or --remove options, only supports the --json flag")
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

		// Add a process to the scope filter
		if addProc != "" {

			return nil
		}

		// Remove a process from the scope filter
		if remProc != "" {

			return nil
		}

		//pid, err := rc.Filter(args)
		//if err != nil {
		//	util.ErrAndExit("Filter failure: %v", err)
		//}

		//if rc.Rootdir != "" {
		//	util.Warn("Attaching to process %d", pid)
		//	util.Warn("It can take up to 1 minute to attach to a process in a parent namespace")
		//}

		// Retrieve an existing filter file
		filePath := "/usr/lib/appscope/scope_filter"
		if rc.Rootdir != "" {
			filePath = rc.Rootdir + filePath
		}

		file, err := os.Open(filePath)
		if err != nil {
			return err
		}
		defer file.Close()

		data, err := ioutil.ReadAll(file)
		if err != nil {
			return err
		}

		if jsonOut {
			// Read yaml in
			var filter libscope.Filter
			err = yaml.Unmarshal(data, &filter)
			if err != nil {
				fmt.Println("Error unmarshaling YAML:", err)
				return err
			}

			// Convert to json
			jsonData, err := json.Marshal(filter)
			if err != nil {
				fmt.Println("Error marshaling JSON:", err)
				return err
			}

			fmt.Println(string(jsonData))
		} else {
			content := string(data)
			fmt.Println(content)
		}

		return nil
	},
}

func init() {
	runCmdFlags(filterCmd, rc)
	filterCmd.Flags().StringP("rootdir", "R", "", "Path to root filesystem of target namespace")
	filterCmd.Flags().String("add", "", "Add an entry to the global filter")
	filterCmd.Flags().String("remove", "", "Remove an entry from the global filter")
	filterCmd.Flags().BoolP("json", "j", false, "Output as newline delimited JSON")
	filterCmd.Flags().String("source", "", "Source identifier for a filter entry")
	RootCmd.AddCommand(filterCmd)
}
