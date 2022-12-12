package cmd

import (
	"fmt"
	"io"
	"os"

	"github.com/criblio/scope/util"
	"github.com/spf13/cobra"
)

// logsCmd represents the logs command
var logsCmd = &cobra.Command{
	Use:     "logs",
	Short:   "Display scope logs",
	Long:    `Displays internal AppScope logs for troubleshooting AppScope itself.`,
	Example: `scope logs`,
	Args:    cobra.NoArgs,
	Run: func(cmd *cobra.Command, args []string) {
		id, _ := cmd.Flags().GetInt("id")
		lastN, _ := cmd.Flags().GetInt("last")
		scopeLog, _ := cmd.Flags().GetBool("scope")
		serviceLog, _ := cmd.Flags().GetString("service")

		var logPath string
		if len(serviceLog) > 0 {
			logPath = fmt.Sprintf("/var/log/scope/%s.log", serviceLog)
		} else {
			sessions := sessionByID(id)
			if scopeLog {
				logPath = sessions[0].ScopeLogPath
			} else {
				logPath = sessions[0].LdscopeLogPath
			}
		}

		// Open log file
		logFile, err := os.Open(logPath)
		if err != nil {
			if scopeLog {
				fmt.Println("No log file present for the CLI. If you want to see the ldscope logs, try running without -s")
			} else {
				fmt.Println("No log file present for ldscope. If you want to see the CLI logs, try running with -s")
			}
			os.Exit(1)
		}

		offset, err := util.FindReverseLineMatchOffset(lastN, logFile, util.MatchAny())
		if offset < 0 {
			offset = 0
		}
		if err == io.EOF {
			util.ErrAndExit("No logs generated")
		}
		util.CheckErrSprintf(err, "error seeking last N offset: %v", err)
		_, err = logFile.Seek(offset, io.SeekStart)
		util.CheckErrSprintf(err, "error seeking log file: %v", err)
		util.NewlineReader(logFile, util.MatchAlways, func(idx int, Offset int64, b []byte) error {
			fmt.Printf("%s\n", b)
			return nil
		})
	},
}

func init() {
	logsCmd.Flags().IntP("id", "i", -1, "Display logs from specific from session ID")
	logsCmd.Flags().IntP("last", "n", 20, "Show last <n> lines")
	logsCmd.Flags().BoolP("scope", "s", false, "Show scope.log (from CLI) instead of ldscope.log (from library)")
	logsCmd.Flags().StringP("service", "S", "", "Display logs from a Systemd service instead of a session)")
	RootCmd.AddCommand(logsCmd)
}
