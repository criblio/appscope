package cmd

import (
	"context"
	"fmt"
	"math"
	"path/filepath"
	"regexp"
	"time"

	"github.com/criblio/scope/events"
	"github.com/criblio/scope/metrics"
	"github.com/criblio/scope/util"
	"github.com/mum4k/termdash"
	"github.com/mum4k/termdash/cell"
	"github.com/mum4k/termdash/container"
	"github.com/mum4k/termdash/container/grid"
	"github.com/mum4k/termdash/keyboard"
	"github.com/mum4k/termdash/linestyle"
	"github.com/mum4k/termdash/terminal/tcell"
	"github.com/mum4k/termdash/terminal/terminalapi"
	"github.com/mum4k/termdash/widgets/segmentdisplay"
	"github.com/mum4k/termdash/widgets/sparkline"
	"github.com/mum4k/termdash/widgets/text"
	"github.com/spf13/cobra"
	"golang.org/x/crypto/ssh/terminal"
)

// dashCmd represents the dash command
var dashCmd = &cobra.Command{
	Use:     "dash [flags]",
	Short:   "display scope dashboard",
	Long:    `Displays an interactive dashboard with an overview of whats happening with the selected session.`,
	Example: `scope dash`,
	Args:    cobra.NoArgs,
	Run: func(cmd *cobra.Command, args []string) {
		id, _ := cmd.Flags().GetInt("id")

		sessions := sessionByID(id)

		ctx, cancel := context.WithCancel(context.Background())

		w := newWidgets(ctx)

		go readMetrics(sessions[0].WorkDir, w)
		go readEvents(sessions[0].WorkDir, w)
		runDashboard(ctx, cancel, w)
	},
}

func init() {
	RootCmd.AddCommand(dashCmd)
	dashCmd.Flags().IntP("id", "i", -1, "Display info from specific from session ID")
}

func readMetrics(workDir string, w *widgets) {
	metricsPath := filepath.Join(workDir, "metrics.json")
	tr, err := util.NewTailReader(metricsPath)
	util.CheckErrSprintf(err, "%v", err)
	in := make(chan metrics.Metric)
	go metrics.Reader(tr, util.MatchAlways, in)

	lastT := time.Unix(0, 0)
	// Keep current values in a map
	var cur map[string]float64

	for m := range in {
		// Every 10 seconds, flush current values, setting 0 for those values which are not present
		if m.Time.Sub(lastT) > time.Duration(10*time.Second) {
			mnames := []string{"proc.cpu_perc",
				"proc.mem",
				"proc.fd",
				"proc.thread",
				"proc.child",
				"fs.open",
				"fs.close",
				"fs.read",
				"fs.write",
				"fs.error",
				"fs.duration",
				"net.rx",
				"net.tx",
				"net.tcp",
				"net.udp",
				"net.error",
				"net.conn_duration",
				"http.requests",
				"http.server.duration",
				"http.response.content_length",
				"net.dns"}
			for _, mname := range mnames {
				var val float64
				var ok bool
				if val, ok = cur[mname]; !ok {
					val = 0
				}
				switch mname {
				case "proc.cpu_perc":
					writeSingleValue(w.cpuPerc, val)
				case "proc.mem":
					writeSingleValue(w.mem, util.ByteCountSI(int64(val*1024)))
				case "proc.fd":
					writeSingleValue(w.fds, val)
				case "proc.thread":
					writeSingleValue(w.threads, val)
				case "proc.child":
					writeSingleValue(w.children, val)
				case "fs.open":
					writeSparklineFloat64(w.fsOpen, val)
				case "fs.close":
					writeSparklineFloat64(w.fsClose, val)
				case "fs.read":
					writeSparklineFloat64(w.fsRead, val)
				case "fs.write":
					writeSparklineFloat64(w.fsWrite, val)
				case "fs.error":
					writeSparklineFloat64(w.fsError, val)
				case "fs.duration":
					writeSparklineFloat64(w.fsDuration, val)
				case "net.rx":
					writeSparklineFloat64(w.netRx, val)
				case "net.tx":
					writeSparklineFloat64(w.netTx, val)
				case "net.tcp":
					writeSparklineFloat64(w.netTCP, val)
				case "net.udp":
					writeSparklineFloat64(w.netUDP, val)
				case "net.error":
					writeSparklineFloat64(w.netError, val)
				case "net.conn_duration":
					writeSparklineFloat64(w.netDuration, val)
				case "http.requests":
					writeSparklineFloat64(w.httpReq, val)
				case "http.server.duration":
					writeSparklineFloat64(w.httpDuration, val)
				case "http.response.content_length":
					writeSparklineFloat64(w.httpBytes, val)
				case "net.dns":
					writeSparklineFloat64(w.dnsReq, val)
				}
			}
			cur = make(map[string]float64, len(mnames))
			lastT = m.Time
		}
		_, ok := cur[m.Name]
		if ok {
			if m.Type == metrics.Gauge {
				cur[m.Name] = m.Value
			} else if m.Type == metrics.Count {
				cur[m.Name] = cur[m.Name] + m.Value
			} // If we don't know what type, don't do anything on second value
		} else {
			cur[m.Name] = m.Value
		}
	}
}

func readEvents(workDir string, w *widgets) {
	metricsPath := filepath.Join(workDir, "events.json")
	tr, err := util.NewTailReader(metricsPath)
	util.CheckErrSprintf(err, "%v", err)
	in := make(chan map[string]interface{})
	eventCount, _ := util.CountLines(metricsPath)
	termWidth, _, err := terminal.GetSize(0)
	skipEvents := eventCount - 20
	if skipEvents < 0 {
		skipEvents = 0
	}
	go events.Reader(tr, 0, util.MatchSkipN(skipEvents), in)

	for e := range in {
		eventText := getEventText(e, termWidth-4, false)
		results := ansiToTermDashColors(eventText)
		for _, r := range results {
			err = w.events.Write(r.text, r.options)
			util.CheckErrSprintf(err, "%v", err)
		}
	}
}

var colorTokenRegex = regexp.MustCompile(`([\x1b|\033]\[[0-9;]*[0-9]+m)([^\x1b|\033]+)[\x1b|\033]\[0m`)

type textWithWriteOption struct {
	text    string
	options text.WriteOption
}

func ansiToTermDashColors(input string) []textWithWriteOption {
	matchIndexes := colorTokenRegex.FindAllStringSubmatchIndex(input, -1)
	results := []textWithWriteOption{}

	for _, match := range matchIndexes {
		colorCode := input[match[2]:match[3]]
		matchText := input[match[4]:match[5]]

		var options text.WriteOption
		switch colorCode {
		case "\x1b[30m":
			options = text.WriteCellOpts(cell.FgColor(cell.ColorBlack))
		case "\x1b[31m":
			options = text.WriteCellOpts(cell.FgColor(cell.ColorRed))
		case "\x1b[32m":
			options = text.WriteCellOpts(cell.FgColor(cell.ColorGreen))
		case "\x1b[33m":
			options = text.WriteCellOpts(cell.FgColor(cell.ColorYellow))
		case "\x1b[34m":
			options = text.WriteCellOpts(cell.FgColor(cell.ColorBlue))
		case "\x1b[35m":
			options = text.WriteCellOpts(cell.FgColor(cell.ColorMagenta))
		case "\x1b[36m":
			options = text.WriteCellOpts(cell.FgColor(cell.ColorCyan))
		case "\x1b[40m":
			options = text.WriteCellOpts(cell.BgColor(cell.ColorBlack))
		case "\x1b[41m":
			options = text.WriteCellOpts(cell.BgColor(cell.ColorRed))
		case "\x1b[42m":
			options = text.WriteCellOpts(cell.BgColor(cell.ColorGreen))
		case "\x1b[43m":
			options = text.WriteCellOpts(cell.BgColor(cell.ColorYellow))
		case "\x1b[44m":
			options = text.WriteCellOpts(cell.BgColor(cell.ColorBlue))
		case "\x1b[45m":
			options = text.WriteCellOpts(cell.BgColor(cell.ColorMagenta))
		case "\x1b[47m":
			options = text.WriteCellOpts(cell.BgColor(cell.ColorWhite))
		case "\x1b[90m":
			options = text.WriteCellOpts(cell.FgColor(cell.ColorRGB24(85, 85, 85)))
		case "\x1b[91m":
			options = text.WriteCellOpts(cell.FgColor(cell.ColorRGB24(255, 85, 85)))
		case "\x1b[92m":
			options = text.WriteCellOpts(cell.FgColor(cell.ColorRGB24(85, 255, 85)))
		case "\x1b[93m":
			options = text.WriteCellOpts(cell.FgColor(cell.ColorRGB24(255, 255, 85)))
		case "\x1b[94m":
			options = text.WriteCellOpts(cell.FgColor(cell.ColorRGB24(85, 85, 255)))
		case "\x1b[95m":
			options = text.WriteCellOpts(cell.FgColor(cell.ColorRGB24(255, 85, 255)))
		case "\x1b[96m":
			options = text.WriteCellOpts(cell.FgColor(cell.ColorRGB24(85, 255, 255)))
		case "\x1b[97m":
			options = text.WriteCellOpts(cell.FgColor(cell.ColorRGB24(255, 255, 255)))
		case "\x1b[100m":
			options = text.WriteCellOpts(cell.BgColor(cell.ColorRGB24(85, 85, 85)))
		case "\x1b[101m":
			options = text.WriteCellOpts(cell.BgColor(cell.ColorRGB24(255, 85, 85)))
		case "\x1b[102m":
			options = text.WriteCellOpts(cell.BgColor(cell.ColorRGB24(85, 255, 85)))
		case "\x1b[103m":
			options = text.WriteCellOpts(cell.BgColor(cell.ColorRGB24(255, 255, 85)))
		case "\x1b[104m":
			options = text.WriteCellOpts(cell.BgColor(cell.ColorRGB24(85, 85, 255)))
		case "\x1b[105m":
			options = text.WriteCellOpts(cell.BgColor(cell.ColorRGB24(255, 85, 255)))
		case "\x1b[106m":
			options = text.WriteCellOpts(cell.BgColor(cell.ColorRGB24(85, 255, 255)))
		case "\x1b[107m":
			options = text.WriteCellOpts(cell.BgColor(cell.ColorWhite))
		default:
			options = text.WriteCellOpts(cell.FgColor(cell.ColorWhite))
		}

		results = append(results, textWithWriteOption{text: matchText, options: options})
	}
	results = append(results, textWithWriteOption{text: "\n", options: text.WriteCellOpts()})
	return results
}

func runDashboard(ctx context.Context, cancel context.CancelFunc, w *widgets) {
	t, err := tcell.New(tcell.ColorMode(terminalapi.ColorMode256))
	if err != nil {
		panic(err)
	}
	defer t.Close()

	c, err := container.New(t,
		container.ID(rootID),
		container.Border(linestyle.Light),
		container.BorderTitle("AppScope 0.5"))
	if err != nil {
		panic(err)
	}

	gridOpts, err := w.newGrid(ctx)

	if err != nil {
		panic(err)
	}

	if err := c.Update(rootID, gridOpts...); err != nil {
		panic(err)
	}

	quitter := func(k *terminalapi.Keyboard) {
		if k.Key == keyboard.KeyEsc || k.Key == keyboard.KeyCtrlC {
			cancel()
		}
	}
	if err := termdash.Run(ctx, t, c, termdash.KeyboardSubscriber(quitter), termdash.RedrawInterval(redrawInterval)); err != nil {
		panic(err)
	}
}

type widgets struct {
	cpuPerc      *segmentdisplay.SegmentDisplay
	mem          *segmentdisplay.SegmentDisplay
	threads      *segmentdisplay.SegmentDisplay
	children     *segmentdisplay.SegmentDisplay
	fds          *segmentdisplay.SegmentDisplay
	fsOpen       *sparkline.SparkLine
	fsClose      *sparkline.SparkLine
	fsRead       *sparkline.SparkLine
	fsWrite      *sparkline.SparkLine
	fsError      *sparkline.SparkLine
	fsDuration   *sparkline.SparkLine
	netRx        *sparkline.SparkLine
	netTx        *sparkline.SparkLine
	netTCP       *sparkline.SparkLine
	netUDP       *sparkline.SparkLine
	netError     *sparkline.SparkLine
	netDuration  *sparkline.SparkLine
	httpReq      *sparkline.SparkLine
	httpDuration *sparkline.SparkLine
	httpBytes    *sparkline.SparkLine
	dnsReq       *sparkline.SparkLine
	events       *text.Text
}

func newWidgets(ctx context.Context) *widgets {
	w := widgets{}
	w.cpuPerc = newSegment()
	w.mem = newSegment()
	w.threads = newSegment()
	w.fds = newSegment()
	w.children = newSegment()

	w.fsOpen = newSparkline(sparkline.Color(cell.ColorCyan))
	w.fsClose = newSparkline(sparkline.Color(cell.ColorCyan))
	w.fsRead = newSparkline(sparkline.Color(cell.ColorCyan))
	w.fsWrite = newSparkline(sparkline.Color(cell.ColorCyan))
	w.fsError = newSparkline(sparkline.Color(cell.ColorCyan))
	w.fsDuration = newSparkline(sparkline.Color(cell.ColorCyan))

	w.netRx = newSparkline(sparkline.Color(cell.ColorRed))
	w.netTx = newSparkline(sparkline.Color(cell.ColorRed))
	w.netTCP = newSparkline(sparkline.Color(cell.ColorRed))
	w.netUDP = newSparkline(sparkline.Color(cell.ColorRed))
	w.netError = newSparkline(sparkline.Color(cell.ColorRed))
	w.netDuration = newSparkline(sparkline.Color(cell.ColorRed))

	w.httpReq = newSparkline(sparkline.Color(cell.ColorBlue))
	w.httpBytes = newSparkline(sparkline.Color(cell.ColorBlue))
	w.httpDuration = newSparkline(sparkline.Color(cell.ColorBlue))
	w.dnsReq = newSparkline(sparkline.Color(cell.ColorGray))

	text, _ := text.New(text.RollContent())
	w.events = text
	return &w
}

func (w widgets) newGrid(ctx context.Context) ([]container.Option, error) {
	builder := grid.New()
	builder.Add(
		grid.RowHeightPerc(20,
			grid.ColWidthPerc(20,
				grid.Widget(w.cpuPerc,
					container.Border(linestyle.Light),
					container.BorderTitle("CPU Perc"),
				),
			),
			grid.ColWidthPerc(20,
				grid.Widget(w.mem,
					container.Border(linestyle.Light),
					container.BorderTitle("Memory"),
				),
			),
			grid.ColWidthPerc(20,
				grid.Widget(w.fds,
					container.Border(linestyle.Light),
					container.BorderTitle("FDs"),
				),
			),
			grid.ColWidthPerc(20,
				grid.Widget(w.threads,
					container.Border(linestyle.Light),
					container.BorderTitle("Threads"),
				),
			),
			grid.ColWidthPerc(20,
				grid.Widget(w.children,
					container.Border(linestyle.Light),
					container.BorderTitle("Children"),
				),
			),
		),
		grid.RowHeightPerc(50,
			grid.ColWidthPercWithOpts(33,
				[]container.Option{container.Border(linestyle.Light), container.BorderTitle("Filesystem")},
				grid.RowHeightPerc(15, grid.Widget(w.fsOpen, container.Border(linestyle.Light), container.BorderTitle("Open"))),
				grid.RowHeightPerc(15, grid.Widget(w.fsClose, container.Border(linestyle.Light), container.BorderTitle("Close"))),
				grid.RowHeightPerc(20, grid.Widget(w.fsRead, container.Border(linestyle.Light), container.BorderTitle("Read"))),
				grid.RowHeightPerc(20, grid.Widget(w.fsWrite, container.Border(linestyle.Light), container.BorderTitle("Write"))),
				grid.RowHeightPerc(15, grid.Widget(w.fsError, container.Border(linestyle.Light), container.BorderTitle("Error"))),
				grid.RowHeightPerc(15, grid.Widget(w.fsDuration, container.Border(linestyle.Light), container.BorderTitle("Duration"))),
			),
			grid.ColWidthPercWithOpts(33,
				[]container.Option{container.Border(linestyle.Light), container.BorderTitle("Network")},
				grid.RowHeightPerc(20, grid.Widget(w.netRx, container.Border(linestyle.Light), container.BorderTitle("Receive"))),
				grid.RowHeightPerc(20, grid.Widget(w.netTx, container.Border(linestyle.Light), container.BorderTitle("Transmit"))),
				grid.RowHeightPerc(15, grid.Widget(w.netTCP, container.Border(linestyle.Light), container.BorderTitle("TCP Conn"))),
				grid.RowHeightPerc(15, grid.Widget(w.netUDP, container.Border(linestyle.Light), container.BorderTitle("UDP Conn"))),
				grid.RowHeightPerc(15, grid.Widget(w.netError, container.Border(linestyle.Light), container.BorderTitle("Error"))),
				grid.RowHeightPerc(15, grid.Widget(w.netDuration, container.Border(linestyle.Light), container.BorderTitle("Duration"))),
			),
			grid.ColWidthPerc(33,
				grid.RowHeightPercWithOpts(66,
					[]container.Option{container.Border(linestyle.Light), container.BorderTitle("HTTP")},
					grid.RowHeightPerc(40, grid.Widget(w.httpReq, container.Border(linestyle.Light), container.BorderTitle("Requests"))),
					grid.RowHeightPerc(40, grid.Widget(w.httpDuration, container.Border(linestyle.Light), container.BorderTitle("Duration"))),
					grid.RowHeightPerc(20, grid.Widget(w.httpBytes, container.Border(linestyle.Light), container.BorderTitle("Response Bytes"))),
				),
				grid.RowHeightPerc(33, grid.Widget(w.dnsReq, container.Border(linestyle.Light), container.BorderTitle("DNS Requests"))),
			),
		),
		grid.RowHeightPerc(30,
			grid.Widget(w.events, container.Border(linestyle.Light), container.BorderTitle("Events")),
		),
	)

	return builder.Build()
}

func writeSingleValue(s *segmentdisplay.SegmentDisplay, val interface{}) {
	var chunk *segmentdisplay.TextChunk
	switch val.(type) {
	case int:
		chunk = segmentdisplay.NewChunk(fmt.Sprintf("%d", val))
	case float64:
		chunk = segmentdisplay.NewChunk(fmt.Sprintf("%.0f", val))
	case string:
		chunk = segmentdisplay.NewChunk(val.(string))
	}
	if err := s.Write([]*segmentdisplay.TextChunk{chunk}); err != nil {
		panic(err)
	}
}

func writeSparklineFloat64(w *sparkline.SparkLine, val float64) {
	w.Add([]int{int(math.Round(val))})
}

func newSegment() *segmentdisplay.SegmentDisplay {
	s, err := segmentdisplay.New()
	if err != nil {
		panic(err)
	}
	writeSingleValue(s, 0)
	return s
}

func newSparkline(opts ...sparkline.Option) *sparkline.SparkLine {
	sl, err := sparkline.New(opts...)
	if err != nil {
		panic(err)
	}
	return sl
}

// redrawInterval is how often termdash redraws the screen.
const redrawInterval = 1000 * time.Millisecond

// rootID is the ID assigned to the root container.
const rootID = "root"
