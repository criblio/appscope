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
	Current HeaderConfCurrent
}

// HeaderConfCurrent is the `info.configuration.current` node in the JSON header object
type HeaderConfCurrent struct {
	Event    HeaderConfEvent
	LibScope HeaderConfLibScope
	Metric   HeaderConfMetric
	Payload  HeaderConfPayload
}

// HeaderConfEvent is the `info.configuration.current.event` node in the JSON header object
type HeaderConfEvent struct {
	Enable string
	//Format    map[string]interface{}
	//Transport map[string]interface{}
	//Watch     map[string]interface{}
}

// HeaderConfLibScope is the `info.configuration.current.libscope` node in the JSON header object
type HeaderConfLibScope struct {
	CommandDir  string
	ConfigEvent string
	//Log           map[string]interface{}
	SummaryPeriod int
}

// HeaderConfMetric is the `info.configuration.current.metric` node in the JSON header object
type HeaderConfMetric struct {
	Enable string
	//Format    map[string]interface{}
	//Transport map[string]interface{}
}

// HeaderConfPayload is the `info.configuration.current.payload` node in the JSON header object
type HeaderConfPayload struct {
	Dir    string
	Enable string
}

// HeaderProcess is the `info.process` node in the JSON header object
type HeaderProcess struct {
	Cmd         string
	Hostname    string
	Id          string
	LibScopeVer string
	Pid         int
	Ppid        int
	ProcName    string
}

// NewHeader creates a Header object
func NewHeader() Header {
	header := Header{}

	return header
}
