package cmd

import (
	"errors"
	"fmt"

	"github.com/criblio/scope/internal"
	"github.com/criblio/scope/util"
	"github.com/spf13/cobra"
)

/* Args Matrix (X disallows)
 */

// psCmd represents the run command
var psCmd = &cobra.Command{
	Use:   "ps",
	Short: "List processes currently being scoped",
	Long:  `Lists all processes where the libscope library is injected.`,
	Args:  cobra.NoArgs,
	Run: func(cmd *cobra.Command, args []string) {
		internal.InitConfig()
		// Nice message for non-adminstrators
		err := util.UserVerifyRootPerm()
		if errors.Is(err, util.ErrGetCurrentUser) {
			util.ErrAndExit("Unable to get current user: %v", err)
		}
		if errors.Is(err, util.ErrMissingAdmPriv) {
			fmt.Println("INFO: Run as root (or via sudo) to see all scoped processes")
		}

		procs, err := util.ProcessesScoped()
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
	RootCmd.AddCommand(psCmd)
}
