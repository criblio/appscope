package libscope

// Header represents the JSON object a client sends to
// LogStream when it connects.
type Header struct {
	Format    string     `json:"format"`    // `ndjson` for event/metric connections or `scope` for payloads
	AuthToken string     `json:"authToken"` // optional authentication token
	Info      HeaderInfo `json:"info"`      // the rest of the header information
}

// The `info` node in the JSON header object
type HeaderInfo struct {
	Configuration HeaderConf        `json:"configuration"` // AppScope configuration; i.e. `scope.yml`
	Environment   map[string]string `json:"environment"`   // Environment variables
	Process       HeaderProcess     `json:"process"`       // Process details
}

// HeaderConf is the `info.configuration` node in the JSON header object
type HeaderConf struct {
	Current HeaderConfCurrent `json:"current"`
}

// HeaderConfCurrent is the `info.configuration.current` node in the JSON header object
type HeaderConfCurrent struct {
	Event    HeaderConfEvent    `json:"event"`
	LibScope HeaderConfLibScope `json:"libscope"`
	Metric   HeaderConfMetric   `json:"metric"`
	Payload  HeaderConfPayload  `json:"payload"`
}

// HeaderConfEvent is the `info.configuration.current.event` node in the JSON header object
type HeaderConfEvent struct {
	Enable    string                 `json:"enable"`
	Format    HeaderConfEventFormat  `json:"format"`
	Transport HeaderTransport        `json:"transport"`
	Watch     []HeaderConfEventWatch `json:"watch"`
}

// HeaderConfEventFormat is the `info.configuration.current.event.format` node in the JSON header object
type HeaderConfEventFormat struct {
	Type            string `json:"type"`
	EnhanceFS       string `json:"enhancefs"`
	MaxEventsperSec int    `json:"maxeventspersec"`
}

// HeaderConfEventWatch is and element in the `info.configuration.current.event.watch` array in the JSON header object
type HeaderConfEventWatch struct {
	Type    string `json:"type"`
	Name    string `json:"name"`
	Field   string `json:"field"`
	Value   string `json:"value"`
	Headers string `json:"headers"`
}

// HeaderConfLibScope is the `info.configuration.current.libscope` node in the JSON header object
type HeaderConfLibScope struct {
	CommandDir    string                `json:"commanddir"`
	ConfigEvent   string                `json:"configevent"`
	SummaryPeriod int                   `json:"summaryperiod"`
	Log           HeaderConfLibScopeLog `json:"log"`
}

// HeaderConfLibScopeLog is the `info.configuration.current.libscope.log` node in the JSON header object
type HeaderConfLibScopeLog struct {
	Level     string          `json:"level"`
	Transport HeaderTransport `json:"transport"`
}

// HeaderConfMetric is the `info.configuration.current.metric` node in the JSON header object
type HeaderConfMetric struct {
	Enable    string                 `json:"enable"`
	Format    HeaderConfMetricFormat `json:"format"`
	Transport HeaderTransport        `json:"transport"`
}

// HeaderConfMetricFormat is the `info.configuration.current.metric.format` node in the JSON header object
type HeaderConfMetricFormat struct {
	Type         string            `json:"type"`
	StatsdPrefix string            `json:"statsdprefix"`
	StatsdMaxLen int               `json:"statsdmaxlen"`
	Verbosity    int               `json:"verbosity"`
	Tags         map[string]string `json:"tags"`
}

// HeaderConfPayload is the `info.configuration.current.payload` node in the JSON header object
type HeaderConfPayload struct {
	Enable string `json:"enable"`
	Dir    string `json:"dir"`
}

// HeaderProcess is the `info.process` node in the JSON header object
type HeaderProcess struct {
	Cmd         string `json:"cmd"`
	Hostname    string `json:"hostname"`
	Id          string `json:"id"`
	LibScopeVer string `json:"libscopever"`
	Pid         int    `json:"pid"`
	Ppid        int    `json:"ppid"`
	ProcName    string `json:"procname"`
}

// HeaderTransport is one of the `transport` nodes in the JSON header object
type HeaderTransport struct {
	Type      string             `json:"type"`
	Host      string             `json:"host"`
	Port      string             `json:"port"`
	Path      string             `json:"path"`
	Buffering string             `json:"buffering"`
	Tls       HeaderTransportTls `json:"tls"`
}

// HeaderTransportTls is one of the `transport.tls` nodes in the JSON header object
type HeaderTransportTls struct {
	Enable         string `json:"enable"`
	ValidateServer string `json:"validateserver"`
	CaCertPath     string `json:"cacertpath"`
}

// NewHeader creates a Header object
func NewHeader() Header {
	header := Header{}

	return header
}
