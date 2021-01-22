package cmd

import (
	"fmt"
	"os"

	"github.com/criblio/scope/bolton"
	"github.com/spf13/cobra"
)

// boltonCmd represents the bolton command
var boltonCmd = &cobra.Command{
	Use:     "bolton",
	Short:   "In loving memory of bolton",
	Long:    `In loving memory of bolton`,
	Example: `scope bolton`,
	Args:    cobra.NoArgs,
	Hidden:  true,
	Run: func(cmd *cobra.Command, args []string) {
		logoBytes := bolton.MustAsset("logo.txt")
		os.Stdout.Write(logoBytes)
		fmt.Printf("\n      In Loving Meory of bolton    \n")
	},
}

func init() {
	RootCmd.AddCommand(boltonCmd)
}
