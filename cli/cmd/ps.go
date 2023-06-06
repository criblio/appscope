package cmd

import (
	"errors"

	"github.com/criblio/scope/internal"
	"github.com/criblio/scope/util"
	"github.com/spf13/cobra"
)

/* Args Matrix (X disallows)
 */

// psCmd represents the ps command
var psCmd = &cobra.Command{
	Use:   "ps",
	Short: "List processes currently being scoped",
	Long:  `List processes currently being scoped.`,
	Example: `  scope ps
  scope ps --json
  scope ps --rootdir /path/to/host/mount
  scope ps --rootdir /path/to/host/mount/proc/<hostpid>/root`,
	Args: cobra.NoArgs,
	Run: func(cmd *cobra.Command, args []string) {
		internal.InitConfig()
		rootdir, _ := cmd.Flags().GetString("rootdir")

		// Nice message for non-adminstrators
		err := util.UserVerifyRootPerm()
		if errors.Is(err, util.ErrGetCurrentUser) {
			util.ErrAndExit("Unable to get current user: %v", err)
		}
		if errors.Is(err, util.ErrMissingAdmPriv) {
			util.Warn("INFO: Run as root (or via sudo) to see all scoped processes")
		}

		procs, err := util.ProcessesScoped(rootdir)
		if err != nil {
			util.ErrAndExit("Unable to retrieve scoped processes: %v", err)
		}
		util.PrintObj([]util.ObjField{
			{Name: "ID", Field: "id"},
			{Name: "Pid", Field: "pid"},
			{Name: "User", Field: "user"},
			{Name: "Command", Field: "command"},
		}, procs)
	},
}

func init() {
	psCmd.Flags().StringP("rootdir", "R", "", "Path to root filesystem of target namespace")
	psCmd.Flags().BoolP("json", "j", false, "Output as newline delimited JSON")
	RootCmd.AddCommand(psCmd)
}
