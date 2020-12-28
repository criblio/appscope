package metrics

type Metric struct {
	Name  string     `json:"name"`
	Value float64    `json:"value"`
	Type  MetricType `json:"type"`
	Unit  string     `json:"unit"`
	// SampleRate float32     `json:"sample"`
	Tags []MetricTag `json:"tags"`
}

type MetricTag struct {
	Name  string `json:"name"`
	Value string `json:"value"`
}

type MetricType int

const (
	Count = iota
	Gauge
	Set
	Histogram
	Timer
	Distribution
)

func (mt MetricType) String() string {
	return [...]string{"Count", "Gauge", "Set", "Histogram", "Timer", "Distribution"}[mt]
}
