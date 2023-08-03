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
	scope listener --addr 127.0.0.1:9999 --notifytoken example_token --channelid channel_id
	sopce listener --addr 127.0.0.1:10070`,
	Args: cobra.NoArgs,
	Run: func(cmd *cobra.Command, args []string) {
		notifyToken, _ := cmd.Flags().GetString("notifytoken")
		channelid, _ := cmd.Flags().GetString("channelid")
		addr, _ := cmd.Flags().GetString("addr")
		listener.ListenAndServer(addr, notifyToken, channelid)
	},
}

func init() {
	listenerCmd.Flags().StringP("addr", "", "", "Set address to listen")
	listenerCmd.Flags().StringP("notifytoken", "", "", "Token to notification system (Slack)")
	listenerCmd.Flags().StringP("channelid", "", "", "Channel id for notification system (Slack)")
	listenerCmd.MarkFlagRequired("addr")

	RootCmd.AddCommand(listenerCmd)
}
