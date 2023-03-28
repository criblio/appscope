package cmd

import (
	"fmt"
	"log"
	"net/http"

	"github.com/criblio/scope/promserver"
	"github.com/spf13/cobra"
)

/*
 * Args Matrix (X disallows)
 * port filesrc
 * port           -
 * filesrc        -
 *
*/

// promServerCmd represents a Prometheus target
var promServerCmd = &cobra.Command {
	Use:   "prom [flags]",
	Short: "Run the Prometheus Target",
	Long:  `Listen and respond to metrics requests from a Prometheus scraper.`,
	Example: `scope prom --port 9100`,
	Args: cobra.NoArgs,
	Run: func(cmd *cobra.Command, args []string) {
		port, _ := cmd.Flags().GetString("port")
		filesrc, _ := cmd.Flags().GetString("filesrc")

		laddr := fmt.Sprint("0.0.0.0:", port)
		fmt.Printf("file source %v port %v connection %v\n", filesrc, port, laddr)

		http.HandleFunc("/metrics", promserver.Handler)
		err := http.ListenAndServe(laddr, nil)
		if err != nil {
			log.Fatal("ListenAndServe: ", err)
		}
	},
}

func init() {
	promServerCmd.Flags().StringP("port", "p", "9100", "Port number to listen on for HTTP metrics requests")
	promServerCmd.Flags().StringP("filesrc", "f", "", "Source directory for metrics data from scoped apps")
	RootCmd.AddCommand(promServerCmd)
}

