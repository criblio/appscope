package libscope

// Header represents the JSON object a client sends to
// LogStream when it connects.
type Header struct {
	Format    string     `json:"format,omitempty"`    // `ndjson` for event/metric connections or `scope` for payloads
	AuthToken string     `json:"authToken,omitempty"` // optional authentication token
	Info      HeaderInfo `json:"info,omitempty"`      // the rest of the header information
}

// The `info` node in the JSON header object
type HeaderInfo struct {
	Configuration HeaderConf        `json:"configuration,omitempty"` // AppScope configuration; i.e. `scope.yml`
	Environment   map[string]string `json:"environment,omitempty"`   // Environment variables
	Process       HeaderProcess     `json:"process,omitempty"`       // Process details
}

// HeaderConf is the `info.configuration` node in the JSON header object
type HeaderConf struct {
	Current HeaderConfCurrent `json:"current,omitempty"`
}

// HeaderConfCurrent is the `info.configuration.current` node in the JSON header object
type HeaderConfCurrent struct {
	Event    HeaderConfEvent    `json:"event,omitempty"`
	LibScope HeaderConfLibScope `json:"libscope,omitempty"`
	Metric   HeaderConfMetric   `json:"metric,omitempty"`
	Payload  HeaderConfPayload  `json:"payload,omitempty"`
}

// HeaderConfEvent is the `info.configuration.current.event` node in the JSON header object
type HeaderConfEvent struct {
	Enable    string                 `json:"enable,omitempty"`
	Format    HeaderConfEventFormat  `json:"format,omitempty"`
	Transport HeaderTransport        `json:"transport,omitempty"`
	Watch     []HeaderConfEventWatch `json:"watch,omitempty"`
}

// HeaderConfEventFormat is the `info.configuration.current.event.format` node in the JSON header object
type HeaderConfEventFormat struct {
	Type            string `json:"type,omitempty"`
	EnhanceFS       string `json:"enhancefs,omitempty"`
	MaxEventsperSec int    `json:"maxeventspersec,omitempty"`
}

// HeaderConfEventWatch is and element in the `info.configuration.current.event.watch` array in the JSON header object
type HeaderConfEventWatch struct {
	Type    string `json:"type,omitempty"`
	Name    string `json:"name,omitempty"`
	Field   string `json:"field,omitempty"`
	Value   string `json:"value,omitempty"`
	Headers string `json:"headers,omitempty"`
}

// HeaderConfLibScope is the `info.configuration.current.libscope` node in the JSON header object
type HeaderConfLibScope struct {
	CommandDir    string                `json:"commanddir,omitempty"`
	ConfigEvent   string                `json:"configevent,omitempty"`
	SummaryPeriod int                   `json:"summaryperiod,omitempty"`
	Log           HeaderConfLibScopeLog `json:"log,omitempty"`
}

// HeaderConfLibScopeLog is the `info.configuration.current.libscope.log` node in the JSON header object
type HeaderConfLibScopeLog struct {
	Level     string          `json:"level,omitempty"`
	Transport HeaderTransport `json:"transport,omitempty"`
}

// HeaderConfMetric is the `info.configuration.current.metric` node in the JSON header object
type HeaderConfMetric struct {
	Enable    string                 `json:"enable,omitempty"`
	Format    HeaderConfMetricFormat `json:"format,omitempty"`
	Transport HeaderTransport        `json:"transport,omitempty"`
}

// HeaderConfMetricFormat is the `info.configuration.current.metric.format` node in the JSON header object
type HeaderConfMetricFormat struct {
	Type         string            `json:"type,omitempty"`
	StatsdPrefix string            `json:"statsdprefix,omitempty"`
	StatsdMaxLen int               `json:"statsdmaxlen,omitempty"`
	Verbosity    int               `json:"verbosity,omitempty"`
	Tags         map[string]string `json:"tags,omitempty"`
}

// HeaderConfPayload is the `info.configuration.current.payload` node in the JSON header object
type HeaderConfPayload struct {
	Enable string `json:"enable,omitempty"`
	Dir    string `json:"dir,omitempty"`
}

// HeaderProcess is the `info.process` node in the JSON header object
type HeaderProcess struct {
	Cmd         string `json:"cmd,omitempty"`
	Hostname    string `json:"hostname,omitempty"`
	Id          string `json:"id,omitempty"`
	LibScopeVer string `json:"libscopever,omitempty"`
	Pid         int    `json:"pid,omitempty"`
	Ppid        int    `json:"ppid,omitempty"`
	ProcName    string `json:"procname"`
}

// HeaderTransport is one of the `transport` nodes in the JSON header object
type HeaderTransport struct {
	Type      string             `json:"type,omitempty"`
	Host      string             `json:"host,omitempty"`
	Port      string             `json:"port,omitempty"`
	Path      string             `json:"path,omitempty"`
	Buffering string             `json:"buffering,omitempty"`
	Tls       HeaderTransportTls `json:"tls,omitempty"`
}

// HeaderTransportTls is one of the `transport.tls` nodes in the JSON header object
type HeaderTransportTls struct {
	Enable         string `json:"enable,omitempty"`
	ValidateServer string `json:"validateserver,omitempty"`
	CaCertPath     string `json:"cacertpath,omitempty"`
}

// NewHeader creates a Header object
func NewHeader() Header {
	header := Header{}

	return header
}
