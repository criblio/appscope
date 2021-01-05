package metrics

import (
	"io"

	"github.com/criblio/scope/util"
)

// Metric represents a metric sample
type Metric struct {
	Name  string     `json:"name"`
	Value float64    `json:"value"`
	Type  MetricType `json:"type"`
	Unit  string     `json:"unit"`
	// SampleRate float32     `json:"sample"`
	Tags []MetricTag `json:"tags"`
}

// MetricTag is a key/value pair which describes an attribute of a metric
type MetricTag struct {
	Name  string `json:"name"`
	Value string `json:"value"`
}

// MetricType represents the type of a metric
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

// Reader reads dogstatsd metrics from a file
func Reader(r io.Reader, match func(string) bool, out chan Metric) (int, error) {
	br, err := util.NewlineReader(r, match, func(idx int, b []byte) error {
		m := parseDogStatsd(string(b))
		out <- m
		return nil
	})
	close(out)
	return br, err
}
