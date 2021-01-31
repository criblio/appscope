package cmd

import (
	"fmt"

	"github.com/criblio/scope/run"
	"github.com/criblio/scope/util"
	"github.com/spf13/cobra"
)

// excreteCmd represents the excrete command
var excreteCmd = &cobra.Command{
	Use:     "extract (<dir>)",
	Aliases: []string{"excrete", "expunge", "extricate", "exorcise"},
	Short:   "Output instrumentary library files to <dir>",
	Long: `Extract outputs ldscope, libscope.so, scope.yml and scope_protocol.yml to the provided directory. These files can configured
to instrument any application and output the data to any existing tool using simple TCP protocols. Libscope can easily be used
with any dynamic or static application, regardless of the runtime.`,
	Example: `scope extrete
scope excrete /opt/libscope`,
	Args: cobra.MaximumNArgs(1),
	Run: func(cmd *cobra.Command, args []string) {
		outPath := "./"
		if len(args) > 0 {
			if !util.CheckFileExists(args[0]) {
				util.ErrAndExit("%s does not exist", args[0])
			}
			outPath = args[0]
		}

		err := run.CreateAll(outPath)
		util.CheckErrSprintf(err, "error excreting files: %v", err)
		fmt.Printf("Successfully extracted to %s.\n", outPath)
	},
}

func init() {
	RootCmd.AddCommand(excreteCmd)
}
