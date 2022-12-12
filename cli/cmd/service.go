package cmd

import (
	"github.com/spf13/cobra"
)

/* Args Matrix (X disallows)
 *                 force user
 * force           -
 * user                  -
 */

// serviceCmd represents the service command
var serviceCmd = &cobra.Command{
	Use:     "service SERVICE [flags]",
	Short:   "Configure a systemd/OpenRC service to be scoped",
	Long:    `Configures the specified systemd/OpenRC service to be scoped upon starting.`,
	Example: `scope service cribl -c tls://in.my-instance.cribl.cloud:10090`,
	Args:    cobra.ExactArgs(1),
	Run: func(cmd *cobra.Command, args []string) {
		// service name is the first and only argument
		force, _ := cmd.Flags().GetBool("force")
		// TODO add handle user parameter
		user, _ := cmd.Flags().GetString("user")
		rc.Service(args[0], user, force)
	},
}

func init() {
	metricAndEventDestFlags(serviceCmd, rc)
	serviceCmd.Flags().Bool("force", false, "Bypass confirmation prompt")
	serviceCmd.Flags().StringP("user", "u", "", "Specify owner username")

	RootCmd.AddCommand(serviceCmd)
}
