package cmd

import (
	"encoding/json"
	"io"
	"net"
	"os"
	"path/filepath"
	"time"

	linq "github.com/ahmetb/go-linq/v3"
	"github.com/criblio/scope/flows"
	"github.com/criblio/scope/util"
	"github.com/spf13/cobra"
	"golang.org/x/crypto/ssh/terminal"
)

// flowsCmd represents the flows command
var flowsCmd = &cobra.Command{
	Use:   "flows [flags] <ID>",
	Short: "Observed flows from the session, potentially including payloads",
	Long: `Displays observed flows from the given session. If ran with payload capture on,
can output full payloads from the flow.`,
	Example: `scope flows                # Displays all flows
scope flows 124x3c         # Displays more info about the flow
scope flows --out 124x3c   # Displays the outbound payload of that flow
`,
	Args: cobra.MaximumNArgs(1),
	Run: func(cmd *cobra.Command, args []string) {
		id, _ := cmd.Flags().GetInt("id")
		inFile, _ := cmd.Flags().GetBool("in")
		outFile, _ := cmd.Flags().GetBool("out")
		lastN, _ := cmd.Flags().GetInt("last")
		allFlows, _ := cmd.Flags().GetBool("all")
		jsonOut, _ := cmd.Flags().GetBool("json")
		sortField, _ := cmd.Flags().GetString("sort")
		sortReverse, _ := cmd.Flags().GetBool("reverse")
		peerNet, _ := cmd.Flags().GetIPNet("peer")
		// eval, _ := cmd.Flags().GetString("eval")
		enc := json.NewEncoder(os.Stdout)

		// var vm *goja.Runtime
		// var prog *goja.Program
		// var err error
		// if eval != "" {
		// 	vm = goja.New()
		// 	prog, err = goja.Compile("expr", eval, false)
		// 	util.CheckErrSprintf(err, "error compiling JavaScript expression: %v", err)
		// }

		termWidth, _, err := terminal.GetSize(0)
		if err != nil {
			// If we cannot get the terminal size, we are dealing with redirected stdin
			// as opposed to an actual terminal, so we will assume terminal width is
			// 160, to show all columns.
			termWidth = 160
		}

		sessions := sessionByID(id)

		// Open events.json file
		file, err := os.Open(sessions[0].EventsPath)
		util.CheckErrSprintf(err, "error opening events file: %v", err)
		fm, err := flows.GetFlows(sessions[0].PayloadsPath, file)
		util.CheckErrSprintf(err, "error getting flows: %v", err)
		flowsList := fm.List()
		from := linq.From(flowsList)
		if lastN > 0 && !allFlows {
			skip := from.Count() - lastN
			if skip < 0 {
				skip = 0
			}
			from = from.Skip(skip)
		}

		// Filters
		// if eval != "" {
		// 	from = from.Where(func(f interface{}) bool {
		// 		rf := reflect.ValueOf(f)
		// 		for i := 0; i < rf.NumField(); i++ {
		// 			vm.Set(rf.Type().Field(i).Name, rf.Field(i))
		// 		}
		// 		v, err := vm.RunProgram(prog)
		// 		util.CheckErrSprintf(err, "error evaluating JavaScript expression: %v", err)
		// 		res := v.Export()
		// 		switch res.(type) {
		// 		case bool:
		// 			return res.(bool)
		// 		default:
		// 			util.ErrAndExit("error: JavaScript return value is not boolean")
		// 		}
		// 		return false
		// 	})
		// }
		if peerNet.String() != "<nil>" {
			from = from.Where(func(f interface{}) bool {
				fff := f.(flows.Flow)
				peerIP := net.ParseIP(fff.PeerIP)
				return peerNet.Contains(peerIP)
			})
		}

		if sortField != "" {
			sortFunc := func(i, j interface{}) bool {
				f1 := util.GetJSONField(i, sortField)
				if f1 == nil {
					util.ErrAndExit("field %s does not exist in object %s", sortField, util.JSONBytes(i))
				}
				f2 := util.GetJSONField(j, sortField)
				if f2 == nil {
					util.ErrAndExit("field %s does not exist in object %s", sortField, util.JSONBytes(j))
				}
				var ret bool
				switch f1.Value().(type) {
				case int:
					ret = f1.Value().(int) > f2.Value().(int)
				case time.Time:
					ret = f1.Value().(time.Time).After(f2.Value().(time.Time))
				default:
					util.ErrAndExit("unknown type for field %s", sortField)
				}
				if sortReverse {
					return !ret
				}
				return ret
			}
			from = from.Sort(sortFunc)
		}
		from.ToSlice(&flowsList)
		if len(args) > 0 {
			hash, err := util.DecodeOffset(args[0])
			util.CheckErrSprintf(err, "could not decode flow id %s: %v", args[0], err)
			if f, found := fm[hash]; found {
				if inFile || outFile {
					var path string
					if inFile {
						path = filepath.Join(f.BaseDir, f.InFile)
					} else {
						path = filepath.Join(f.BaseDir, f.OutFile)
					}
					file, err := os.Open(path)
					util.CheckErrSprintf(err, "error opening in file %s: %v", path, err)
					io.Copy(os.Stdout, file)
					return
				}
				fields := []util.ObjField{
					{Name: "ID", Field: "id"},
					{Name: "Host IP", Field: "net_host_ip"},
					{Name: "Host Port", Field: "net_host_port"},
					{Name: "Peer IP", Field: "net_peer_ip"},
					{Name: "Peer Port", Field: "net_peer_port"},
					{Name: "Proc", Field: "proc"},
					{Name: "PID", Field: "pid"},
					{Name: "Last Sent", Field: "last_sent_time"},
					{Name: "Duration", Field: "duration"},
					{Name: "Bytes Recv", Field: "net_bytes_recv"},
					{Name: "Bytes Sent", Field: "net_bytes_sent"},
					{Name: "Protocol", Field: "net_protocol"},
					{Name: "Close Reason", Field: "net_close_reason"},
					{Name: "Base Dir", Field: "basedir"},
					{Name: "In File", Field: "in_file"},
					{Name: "Out File", Field: "out_file"},
				}
				if jsonOut {
					enc.Encode(f)
				} else {
					util.PrintObj(fields, f)
				}
				return
			} else {
				util.ErrAndExit("could not find flow for id %s", args[0])
			}
		}
		fields := []util.ObjField{
			{Name: "ID", Field: "id"},
			{Name: "Host IP", Field: "net_host_ip"},
			{Name: "Host Port", Field: "net_host_port"},
			{Name: "Peer IP", Field: "net_peer_ip"},
			{Name: "Peer Port", Field: "net_peer_port"},
			{Name: "Last Sent", Field: "last_sent_time", Transform: func(obj interface{}) string { return util.GetHumanDuration(time.Since(obj.(time.Time))) }},
			{Name: "Duration", Field: "duration"},
		}
		if termWidth > 145 {
			fields = append(fields, []util.ObjField{
				{Name: "Bytes Recv", Field: "net_bytes_recv"},
				{Name: "Bytes Sent", Field: "net_bytes_sent"},
				{Name: "Protocol", Field: "net_protocol"},
			}...)
		}
		if jsonOut {
			for _, f := range flowsList {
				enc.Encode(f)
			}
			return
		}
		util.PrintObj(fields, flowsList)
	},
}

func init() {
	flowsCmd.Flags().IntP("id", "i", -1, "Display flows from specific from session ID")
	flowsCmd.Flags().Bool("in", false, "Output contents of the inbound payload. Requires flow ID specified.")
	flowsCmd.Flags().Bool("out", false, "Output contents of the outbound payload. Requires flow ID specified.")
	flowsCmd.Flags().IntP("last", "n", 20, "Show last <n> flows")
	flowsCmd.Flags().BoolP("all", "a", false, "Show all flows")
	flowsCmd.Flags().BoolP("json", "j", false, "Output as newline delimited JSON")
	flowsCmd.Flags().StringP("sort", "s", "", "Sort descending by field (look at JSON output for field names)")
	flowsCmd.Flags().BoolP("reverse", "r", false, "Reverse sort to ascending")
	flowsCmd.Flags().IPNetP("peer", "p", net.IPNet{}, "Filter to peers in the given network")
	// flowsCmd.Flags().StringP("eval", "e", "", "Evaluate JavaScript expression against flow. Must return truthy to print flow.")
	RootCmd.AddCommand(flowsCmd)
}
