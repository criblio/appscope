package cmd

import (
	"fmt"
	"os"
	"path"
	"path/filepath"

	"github.com/criblio/scope/run"
	"github.com/criblio/scope/util"
	"github.com/spf13/cobra"
)

// excreteCmd represents the excrete command
var excreteCmd = &cobra.Command{
	Use:     "extract [flags] <dir>",
	Aliases: []string{"excrete", "expunge", "extricate", "exorcise"},
	Short:   "Output instrumentary library files to <dir>",
	Long: `Extract outputs ldscope, libscope.so, scope.yml and scope_protocol.yml to the provided directory. These files can configured
to instrument any application and output the data to any existing tool using simple TCP protocols. With --metricdest and --eventdest, 
scope.yml can be preconfigured for outputting to specific destinations. 

The --*dest flags accept file names like /tmp/scope.log or URLs like file:///tmp/scope.log. They may also
be set to sockets with tcp://hostname:port, udp://hostname:port, or tls://hostname:port.

Libscope can easily be used with any dynamic or static application, regardless of the runtime.`,
	Example: `scope extract /opt/libscope
scope extract --metricdest tcp://some.host:8125 --eventdest tcp://other.host:10070 .
`,
	Args: cobra.ExactArgs(1),
	Run: func(cmd *cobra.Command, args []string) {
		outPath := "./"
		if len(args) > 0 {
			if !util.CheckDirExists(args[0]) {
				util.ErrAndExit("%s does not exist or is not a directory", args[0])
			}
			outPath = args[0]
		}
		newPath, err := filepath.Abs(outPath)
		util.CheckErrSprintf(err, "cannot resolve absolute path of %s: %v", outPath, err)
		outPath = newPath

		err = run.CreateAll(outPath)
		util.CheckErrSprintf(err, "error excreting files: %v", err)
		if rc.MetricsDest != "" || rc.EventsDest != "" || rc.CriblDest != "" {
			err = os.Rename(path.Join(outPath, "scope.yml"), path.Join(outPath, "scope_example.yml"))
			util.CheckErrSprintf(err, "error renaming scope.yml: %v", err)
			rc.WorkDir = outPath
			err = rc.WriteScopeConfig(path.Join(outPath, "scope.yml"))
			util.CheckErrSprintf(err, "error writing scope.yml: %v", err)
		}
		fmt.Printf("Successfully extracted to %s.\n", outPath)
	},
}

func init() {
	metricAndEventDestFlags(excreteCmd, rc)
	RootCmd.AddCommand(excreteCmd)
}
