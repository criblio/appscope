package cmd

import (
	"fmt"
	"os"

	"github.com/criblio/scope/internal"
	"github.com/criblio/scope/util"
	"github.com/spf13/cobra"
)

// psCmd represents the run command
var psCmd = &cobra.Command{
	Use:   "ps",
	Short: "List processes currently being scoped",
	Long:  `List all processes where the libscope library is injected`,
	Args:  cobra.NoArgs,
	Run: func(cmd *cobra.Command, args []string) {
		internal.InitConfig()
		procs := util.ProcessesScoped()
		util.PrintObj([]util.ObjField{
			{Name: "ID", Field: "id"},
			{Name: "Pid", Field: "pid"},
			{Name: "User", Field: "user"},
			{Name: "Command", Field: "command"},
			{Name: "Scoped", Field: "scoped"},
		}, procs)
		if _, present := os.LookupEnv("SUDO_USER"); !present {
			fmt.Println("INFO: Run as root (or via sudo) to see all scoped processes")
		}
	},
}

func init() {
	RootCmd.AddCommand(psCmd)
}
