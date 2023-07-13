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
	scope listener --listen localhost:9999 --slacktoken example_token`,
	Args: cobra.NoArgs,
	Run: func(cmd *cobra.Command, args []string) {
		slackToken, _ := cmd.Flags().GetString("slacktoken")
		listenDest, _ := cmd.Flags().GetString("listen")
		listener.Listener(listenDest, slackToken)
	},
}

func init() {
	listenerCmd.Flags().StringP("listen", "", "", "Set listening source (host:port defaults to tls://)")
	listenerCmd.Flags().StringP("slacktoken", "", "", "Token to Slack")
	listenerCmd.MarkFlagRequired("listen")
	listenerCmd.MarkFlagRequired("slacktoken")

	RootCmd.AddCommand(listenerCmd)
}
