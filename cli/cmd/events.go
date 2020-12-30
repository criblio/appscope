package cmd

import (
	"fmt"
	"math"
	"regexp"
	"sort"
	"strings"
	"time"

	"github.com/criblio/scope/events"
	"github.com/fatih/color"
	"github.com/spf13/cobra"
)

var nc = color.New

type colorOpts map[string]*color.Color

var darkColorOpts = colorOpts{
	"key":     nc(color.FgGreen),
	"val":     nc(color.FgHiWhite),
	"_time":   nc(color.FgWhite),
	"data":    nc(color.FgHiWhite),
	"proc":    nc(color.FgHiMagenta),
	"source":  nc(color.FgHiBlack),
	"default": nc(color.FgWhite),
}

var lightColorOpts = colorOpts{
	"key":     nc(color.FgGreen),
	"val":     nc(color.FgBlack),
	"_time":   nc(color.FgBlack),
	"data":    nc(color.FgHiBlack),
	"proc":    nc(color.FgHiMagenta),
	"source":  nc(color.FgHiWhite),
	"default": nc(color.FgBlack),
}

var sourcetypeColors = colorOpts{
	"console": nc(color.FgHiGreen),
	"http":    nc(color.FgHiBlue),
	"default": nc(color.FgYellow),
}

var sourcetypeFields = map[string][]string{
	"http": {"http.host", "http.method", "http.request_content_length", "http.scheme", "http.target", "http.response_content_length", "http.status", "duration", "req_per_sec"},
}

// color returns the matching color or the default
func (co colorOpts) color(color string) *color.Color {
	c := co[color]
	if c == nil {
		c = co["default"]
	}
	return c
}

var co colorOpts

// eventsCmd represents the events command
var eventsCmd = &cobra.Command{
	Use:     "events [flags]",
	Short:   "Outputs events for a session",
	Long:    `Outputs events for a session`,
	Example: `scope events`,
	Args:    cobra.NoArgs,
	Run: func(cmd *cobra.Command, args []string) {
		id, _ := cmd.Flags().GetInt("id")
		sources, _ := cmd.Flags().GetStringSlice("source")
		sourcetypes, _ := cmd.Flags().GetStringSlice("sourcetype")
		noFields, _ := cmd.Flags().GetBool("nofields")
		lastN, _ := cmd.Flags().GetInt("last")

		sessions := sessionByID(id)

		// width, _, err := term.GetSize(0)
		// util.CheckErrSprintf(err, "cannot get terminal width: %v", err)

		skipEvents := 0
		if lastN > -1 {
			count := events.Count(sessions[0].EventsPath)
			skipEvents = count - lastN + 1
		}

		in := make(chan map[string]interface{})
		idx := 0
		go events.Reader(sessions[0].EventsPath, func(line string) bool {
			idx++
			if skipEvents > 0 && idx < skipEvents {
				return false
			}
			all := []events.MatchFunc{}
			if len(sources) > 0 {
				matchsources := []events.MatchFunc{}
				for _, s := range sources {
					matchsources = append(matchsources, events.MatchField("source", s))
				}
				all = append(all, events.MatchAny(matchsources...))
			}
			if len(sourcetypes) > 0 {
				matchsourcetypes := []events.MatchFunc{}
				for _, t := range sourcetypes {
					matchsourcetypes = append(matchsourcetypes, events.MatchField("sourcetype", t))
				}
				all = append(all, events.MatchAny(matchsourcetypes...))
			}
			if len(all) == 0 {
				all = append(all, events.MatchAlways)
			}
			return events.MatchAll(all...)(line)
		}, in)
		for e := range in {
			if _, timeExists := e["_time"]; timeExists {
				timeFp := e["_time"].(float64)
				e["_time"] = time.Unix(int64(math.Floor(timeFp)), int64(math.Mod(timeFp, 1)*1000000)).Format(time.Stamp)
			}
			noop := func(orig interface{}) interface{} { return orig }
			out := colorToken(e, "_time", noop)
			out += colorToken(e, "proc", noop)
			out += sourcetypeColorToken(e)
			out += colorToken(e, "source", noop)
			out += colorToken(e, "data", func(orig interface{}) interface{} {
				ret := ""
				switch orig.(type) {
				case string:
					ret = ansiStrip(strings.TrimSpace(fmt.Sprintf("%s", orig)))
				case map[string]interface{}:
					ret = colorMap(orig.(map[string]interface{}), sourcetypeFields[e["sourcetype"].(string)])
				}
				return ret
			})

			// Delete uninteresting fields
			delete(e, "id")
			delete(e, "_channel")
			delete(e, "sourcetype")

			if !noFields {
				out += colorMap(e, []string{})
			}
			fmt.Printf("%s\n", out)
		}
	},
}

func colorToken(e map[string]interface{}, field string, transform func(interface{}) interface{}) string {
	val := e[field]
	delete(e, field)
	return co.color(field).Sprintf("%s ", transform(val))
}

func sourcetypeColorToken(e map[string]interface{}) string {
	st := e["sourcetype"].(string)
	return sourcetypeColors.color(st).Sprintf("%s ", st)
}

func colorMap(e map[string]interface{}, onlyFields []string) string {
	kv := ""
	keys := make([]string, 0, len(e))
	for k := range e {
		if len(onlyFields) > 0 {
			for _, field := range onlyFields {
				if k == field {
					keys = append(keys, k)
				}
			}
		} else {
			keys = append(keys, k)
		}

	}
	sort.Strings(keys)
	for _, k := range keys {
		kv += co.color("key").Sprintf("%s", k)
		kv += co.color("val").Sprintf(":%s ", interfaceToString(e[k]))
	}
	return kv
}

func interfaceToString(i interface{}) string {
	switch v := i.(type) {
	case float64:
		if v-math.Floor(v) < 0.000001 && v < 1e9 {
			return fmt.Sprintf("%d", int(v))
		}
		return fmt.Sprintf("%g", v)
	case string:
		stringFmt := "%s"
		if strings.Contains(v, " ") {
			stringFmt = "%q"
		}
		return fmt.Sprintf(stringFmt, v)
	default:
		return fmt.Sprintf("%v", v)
	}
}

const ansi = "[\u001B\u009B][[\\]()#;?]*(?:(?:(?:[a-zA-Z\\d]*(?:;[a-zA-Z\\d]*)*)?\u0007)|(?:(?:\\d{1,4}(?:;\\d{0,4})*)?[\\dA-PRZcf-ntqry=><~]))"

var ansire = regexp.MustCompile(ansi)

func ansiStrip(str string) string {
	return ansire.ReplaceAllString(str, "")
}

func init() {
	eventsCmd.Flags().Int("id", -1, "Display events from session ID")
	eventsCmd.Flags().StringSliceP("source", "s", []string{}, "Display events matching supplied sources")
	eventsCmd.Flags().StringSliceP("sourcetype", "t", []string{}, "Display events matching supplied sourcetypes")
	eventsCmd.Flags().Bool("nofields", false, "Disable displaying extra fields")
	eventsCmd.Flags().IntP("last", "n", -1, "Show n last events")
	RootCmd.AddCommand(eventsCmd)

	if co == nil {
		co = darkColorOpts // Hard coded for now
	}
}
