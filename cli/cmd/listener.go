package cmd

import (
	"github.com/criblio/scope/listener"
	"github.com/spf13/cobra"
)

// listenerCmd represents the listener command
var listenerCmd = &cobra.Command{
	Use:   "listener [flags]",
	Short: "Run the scope listener",
	Long:  `Listen to the Appscope events.`,
	Example: `scope listener
	scope listener --addr localhost:9999 --notifytoken example_token`,
	Args: cobra.NoArgs,
	Run: func(cmd *cobra.Command, args []string) {
		notifyToken, _ := cmd.Flags().GetString("notifytoken")
		addr, _ := cmd.Flags().GetString("addr")
		listener.ListenAndServer(addr, notifyToken)
	},
}

func init() {
	listenerCmd.Flags().StringP("addr", "", "", "Set address to listen")
	listenerCmd.Flags().StringP("notifytoken", "", "", "Toke to notifier system (Slack)")
	listenerCmd.MarkFlagRequired("addr")
	listenerCmd.MarkFlagRequired("notifytoken")

	RootCmd.AddCommand(listenerCmd)
}
