package cmd

import (
	"errors"
	"fmt"
	"io/ioutil"
	"os"
	"strings"

	"github.com/criblio/scope/events"
	"github.com/criblio/scope/libscope"
	"github.com/criblio/scope/util"
	"github.com/spf13/cobra"
	"golang.org/x/crypto/ssh/terminal"
)

/* Args Matrix (X disallows)
 *             id source sourcetype match fields allfields last json follow color all eval sort reverse
 * id          -
 * source         -                                                               X
 * sourcetype            -                                                        X
 * match                            -                                             X
 * fields                                 -      X
 * allfields                              X      -
 * last                                                    -                      X
 * json                                                         -           X
 * follow                                                  X         -                     X    X
 * color                                                                    -
 * all            X      X          X                      X                      -   X
 * eval                                                                           X   -
 * sort                                                              X                     -
 * reverse                                                           X                          -
 */

// eventsCmd represents the events command
var eventsCmd = &cobra.Command{
	Use:   "events [flags] ([eventId])",
	Short: "Outputs events for a session",
	Long: `Outputs events for a session. You can obtain detailed information about each event by inputting the Event ID as a positional parameter. (By default, the Event ID appears in blue in []'s at the left.) You can provide filters to narrow down by name (e.g., http, net, fs, console), or by field (e.g., fs.open, stdout, or net.open). You can use JavaScript expressions to further refine the query, and to express logic.`,
	Example: `scope events
scope events m61
scope events --sourcetype http
scope events --source stderr
scope events --match file
scope events --fields net_bytes_sent,net_bytes_recv --match net_bytes
scope events --follow
scope events --all
scope events --allfields
scope events --id 4
scope events --sort _time --reverse
scope events --eval 'sourcetype!="net"'
scope events -n 1000 -e 'sourcetype!="console" && source.indexOf("cribl.log") == -1 && (data["file.name"] || "").indexOf("/proc") == -1'`,
	Args: cobra.MaximumNArgs(1),
	Run: func(cmd *cobra.Command, args []string) {
		var em events.EventMatch
		em.Sources, _ = cmd.Flags().GetStringSlice("source")
		em.Sourcetypes, _ = cmd.Flags().GetStringSlice("sourcetype")
		em.Match, _ = cmd.Flags().GetString("match")
		em.LastN, _ = cmd.Flags().GetInt("last")
		em.AllEvents, _ = cmd.Flags().GetBool("all")
		jsonOut, _ := cmd.Flags().GetBool("json")
		allFields, _ := cmd.Flags().GetBool("allfields")
		fields, _ := cmd.Flags().GetStringSlice("fields")
		eval, _ := cmd.Flags().GetString("eval")
		sortField, _ := cmd.Flags().GetString("sort")
		sortReverse, _ := cmd.Flags().GetBool("reverse")
		id, _ := cmd.Flags().GetInt("id")
		follow, _ := cmd.Flags().GetBool("follow")
		forceColor, _ := cmd.Flags().GetBool("color")

		// Disallow bad argument combinations (see Arg Matrix at top of file)
		if len(em.Sources) > 0 && em.AllEvents {
			helpErrAndExit(cmd, "Cannot specify --source and --all")
		} else if len(em.Sourcetypes) > 0 && em.AllEvents {
			helpErrAndExit(cmd, "Cannot specify --sourcetype and --all")
		} else if em.Match != "" && em.AllEvents {
			helpErrAndExit(cmd, "Cannot specify --match and --all")
		} else if len(fields) > 0 && allFields {
			helpErrAndExit(cmd, "Cannot specify --fields and --allfields")
		} else if cmd.Flags().Lookup("last").Changed && em.AllEvents {
			helpErrAndExit(cmd, "Cannot specify --last and --all")
		} else if jsonOut && forceColor {
			helpErrAndExit(cmd, "Cannot specify --json and --color")
		} else if follow && sortField != "" {
			helpErrAndExit(cmd, "Cannot specify --follow and --sort")
		} else if follow && sortReverse {
			helpErrAndExit(cmd, "Cannot specify --follow and --reverse")
		} else if em.AllEvents && eval != "" {
			helpErrAndExit(cmd, "Cannot specify --all and --eval")
		}

		// Display all matches when matching
		if len(em.Sourcetypes) > 0 || len(em.Sources) > 0 || em.Match != "" || eval != "" {
			if !cmd.Flags().Lookup("last").Changed {
				em.AllEvents = true
			}
		}

		termWidth, _, err := terminal.GetSize(0)
		if err != nil {
			// If we cannot get the terminal size, we are dealing with redirected stdin
			// as opposed to an actual terminal, so we will assume terminal width is
			// 160, to show all columns.
			termWidth = 160
		}

		// Get one or many sessions from history
		sessions := sessionByID(id)

		// Open events.json file
		// Look in default session history path
		// Look in eventdest path
		var file util.ReadSeekCloser
		file, err = os.Open(sessions[0].EventsPath)
		defer file.Close()
		if err != nil && strings.Contains(err.Error(), "events.json: no such file or directory") {
			if util.CheckFileExists(sessions[0].EventsDestPath) {
				dest, _ := ioutil.ReadFile(sessions[0].EventsDestPath)
				fmt.Printf("Events were output to %s\n", dest)
				os.Exit(0)
			}
			promptClean(sessions[0:1])
		} else {
			util.CheckErrSprintf(err, "error opening events file: %v", err)
		}

		// If --follow is specified, tail the file
		if follow {
			file = util.NewTailReader(file)
		}

		// If an Id is specified, set the offset for the match function to retrieve it
		if len(args) > 0 {
			em.Offset, err = util.DecodeOffset(args[0])
			util.CheckErrSprintf(err, "Could not decode encoded offset: %v", err)
			em.AllEvents = false
		}

		// Filter and match events
		// File "file" -> EventMatch filter "em" -> All events (channel) "out"
		out := make(chan libscope.EventBody)
		go func() {
			err := em.Events(file, out)
			if err != nil && strings.Contains(err.Error(), "Error searching for Offset: EOF") {
				err = errors.New("Empty event file.")
			}
			util.CheckErrSprintf(err, "%v", err)
		}()

		// If we were passed an EventID, read that one event, print and exit
		// otherwise print multiple events
		if len(args) > 0 {
			events.PrintEvent(out, jsonOut)
		} else {
			events.PrintEvents(out, fields, sortField, eval, jsonOut, sortReverse, allFields, forceColor, termWidth)
		}
	},
}

func init() {
	eventsCmd.Flags().IntP("id", "i", -1, "Display info from specific from session ID")
	eventsCmd.Flags().StringSliceP("source", "s", []string{}, "Display events matching supplied sources")
	eventsCmd.Flags().StringSliceP("sourcetype", "t", []string{}, "Display events matching supplied sourcetypes")
	eventsCmd.Flags().StringP("match", "m", "", "Display events containing supplied string")
	eventsCmd.Flags().StringSliceP("fields", "", []string{}, "Display custom fields (look at JSON output for field names)")
	eventsCmd.Flags().Bool("allfields", false, "Displaying hidden fields")
	eventsCmd.Flags().IntP("last", "n", 20, "Show last <n> events")
	eventsCmd.Flags().BoolP("json", "j", false, "Output as newline delimited JSON")
	eventsCmd.Flags().BoolP("follow", "f", false, "Follow a file, like tail -f")
	eventsCmd.Flags().Bool("color", false, "Force color on (if tty detection fails or piping)")
	eventsCmd.Flags().BoolP("all", "a", false, "Show all events")
	eventsCmd.Flags().StringP("eval", "e", "", "Evaluate JavaScript expression against event. Must return truthy to print event.\nNote: Post-processes after matching, not guaranteed to return last <n> events.")
	eventsCmd.Flags().StringP("sort", "", "", "Sort descending by field (look at JSON output for field names)")
	eventsCmd.Flags().BoolP("reverse", "r", false, "Reverse sort to ascending. Must be combined with --sort")
	RootCmd.AddCommand(eventsCmd)
}
