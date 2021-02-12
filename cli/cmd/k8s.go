package cmd

import (
	"os"

	"github.com/criblio/scope/k8s"
	"github.com/spf13/cobra"
)

var opt k8s.Options

// k8sCmd represents the k8s command
var k8sCmd = &cobra.Command{
	Use:     "k8s",
	Short:   "Prints k8s configurations to install scope to automatically instrument pods",
	Long:    `Prints k8s configurations to install scope to automatically instrument pods`,
	Example: `scope k8s`,
	Args:    cobra.NoArgs,
	Hidden:  true,
	Run: func(cmd *cobra.Command, args []string) {
		k8s.PrintConfig(os.Stdout, opt)
	},
}

func init() {
	RootCmd.AddCommand(k8sCmd)
	k8sCmd.Flags().StringVar(&opt.App, "app", "scope", "Name of the app in Kubernetes")
	k8sCmd.Flags().StringVar(&opt.Namespace, "namespace", "default", "Name of the namespace to install in")
}
