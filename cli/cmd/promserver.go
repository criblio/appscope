package cmd

import (
	"fmt"
	"net"
	"net/http"
	"os"

	"github.com/criblio/scope/promserver"
	"github.com/criblio/scope/util"
	"github.com/spf13/cobra"
)

/*
 * Args Matrix (X disallows)
 * port filesrc
 * port           -
 * filesrc        -
 *
 */

func getMetrics(laddr string) {
	listen, err := net.Listen("tcp", laddr)
	if err != nil {
		util.ErrAndExit("Listen: %s", err)
		os.Exit(1)
	}

	defer listen.Close()

	for {
		conn, err := listen.Accept()
		if err != nil {
			util.ErrAndExit("Accept %s", err)
		}
		go promserver.Metrics(conn)
	}
}

// promServerCmd represents a Prometheus target
var promServerCmd = &cobra.Command{
	Use:     "prom [flags]",
	Short:   "Run the Prometheus Target",
	Long:    `Listen and respond to metrics requests from a Prometheus scraper.`,
	Example: `scope prom --port 9100`,
	Args:    cobra.NoArgs,
	Run: func(cmd *cobra.Command, args []string) {
		sport, _ := cmd.Flags().GetString("sport")
		mport, _ := cmd.Flags().GetString("mport")

		promserver.Pmsg("scraper port", sport, "metric port", mport)

		mfiles := promserver.GetMfiles()
		for _, mpath := range mfiles {
			promserver.Pmsg("Remove ", mpath)
			if err := os.Remove(mpath); err != nil {
				fmt.Println(err)
				return
			}
		}

		laddr := fmt.Sprint("0.0.0.0:", mport)
		go getMetrics(laddr)

		laddr = fmt.Sprint("0.0.0.0:", sport)
		http.HandleFunc("/metrics", promserver.Handler)
		err := http.ListenAndServe(laddr, nil)
		if err != nil {
			util.ErrAndExit("ListenAndServe: %s", err)
		}
	},
}

func init() {
	promServerCmd.Flags().StringP("sport", "s", "9090", "Port number to listen on for HTTP metrics requests")
	promServerCmd.Flags().StringP("mport", "m", "9109", "Port number to listen on for metrics from libscope")
	RootCmd.AddCommand(promServerCmd)
}
