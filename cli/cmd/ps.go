package cmd

import (
	"fmt"
	"os/user"

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
		user, err := user.Current()
		if err != nil {
			util.ErrAndExit("Unable to get current user: %v", err)
		}
		if user.Uid != "0" {
			fmt.Println("INFO: Run as root (or via sudo) to see all scoped processes")
		}
		procs := util.ProcessesScoped()
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
