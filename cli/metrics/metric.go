package metrics

import (
	"encoding/json"
	"fmt"
	"io"
	"time"

	"github.com/criblio/scope/util"
)

// Metric represents a metric sample
type Metric struct {
	Name  string     `json:"name"`
	Value float64    `json:"value"`
	Type  MetricType `json:"type"`
	Unit  string     `json:"unit"`
	Pid   int        `json:"pid"`
	// SampleRate float32     `json:"sample"`
	Tags []MetricTag `json:"tags"`
	Time time.Time   `json:"_time"`
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

var metricTypesToCodes map[string]MetricType = map[string]MetricType{
	"counter": Count,
	"gauge":   Gauge,
}

// Reader reads dogstatsd metrics from a file
func Reader(r io.Reader, match func(string) bool, out chan Metric) (int, error) {
	br, err := util.NewlineReader(r, match, func(idx int, offset int, b []byte) error {
		m, err := parseJSONMetric(b)
		if err != nil {
			return err
		}
		out <- m
		return nil
	})
	close(out)
	return br, err
}

func parseJSONMetric(b []byte) (Metric, error) {
	m := Metric{}
	metricMap := map[string]interface{}{}
	err := json.Unmarshal(b, &metricMap)
	if err != nil {
		return m, err
	}
	m.Name = metricMap["_metric"].(string)
	delete(metricMap, "_metric")
	m.Value = metricMap["_value"].(float64)
	delete(metricMap, "_value")
	m.Type = metricTypesToCodes[metricMap["_metric_type"].(string)]
	delete(metricMap, "_metric_type")
	if _, ok := metricMap["unit"]; ok {
		m.Unit = metricMap["unit"].(string)
		delete(metricMap, "unit")
	}
	if _, ok := metricMap["pid"]; ok {
		m.Pid = int(metricMap["pid"].(float64))
		delete(metricMap, "pid")
	}
	m.Time = util.ParseEventTime(metricMap["_time"].(float64))
	delete(metricMap, "_time")
	for k, v := range metricMap {
		t := MetricTag{Name: k}
		switch v.(type) {
		case float64:
			t.Value = fmt.Sprintf("%.0f", v.(float64))
		case int:
			t.Value = fmt.Sprintf("%d", v.(int))
		case string:
			t.Value = v.(string)
		}
		m.Tags = append(m.Tags, t)
	}
	return m, nil
}
