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
	Long:    `Display internal scope logs for troubleshooting scope itself`,
	Example: `scope logs`,
	Args:    cobra.NoArgs,
	Run: func(cmd *cobra.Command, args []string) {
		id, _ := cmd.Flags().GetInt("id")
		lastN, _ := cmd.Flags().GetInt("last")
		scopeLog, _ := cmd.Flags().GetBool("scope")

		sessions := sessionByID(id)

		logPath := sessions[0].LdscopeLogPath
		if scopeLog {
			logPath = sessions[0].ScopeLogPath
		}

		// Open log file
		logFile, err := os.Open(logPath)
		util.CheckErrSprintf(err, "%v", err)

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
	RootCmd.AddCommand(logsCmd)
}
