package libscope

type Metric struct {
	Type    string    `json:"type"`
	Id      string    `json:"id"`
	Channel string    `json:"_channel"`
	Body    EventBody `json:"body"`
}

type MetricBody struct {
	Metric     string  `json:"_metric"`
	MetricType string  `json:"_metric_type"`
	Value      int64   `json:"_value"`
	Proc       string  `json:"proc"`
	Pid        int64   `json:"pid"`
	Op         string  `json:"op"`
	NumOps     int64   `json:"numops"`
	Host       string  `json:"host"`
	Unit       string  `json:"unit"`
	Time       float64 `json:"_time"`
	Domain     string  `json:"domain"`
	Duration   int64   `json:"duration"`
	Summary    string  `json:"summary"`
}
