package metrics

import (
	"sort"
	"testing"

	"github.com/criblio/scope/util"
	"github.com/stretchr/testify/assert"
)

func TestJsonMetric(t *testing.T) {
	test1 := `{"_metric":"fs.error","_metric_type":"counter","_value":3,"pid":86707,"op":"summary","file":"summary","class":"open_close","unit":"operation","_time":1610682430.52}`
	m, err := parseJSONMetric([]byte(test1))

	assert.NoError(t, err)
	sort.Slice(m.Tags, func(i, j int) bool { return m.Tags[i].Name < m.Tags[j].Name })

	assert.Equal(t, "fs.error", m.Name)
	assert.Equal(t, float64(3), m.Value)
	assert.Equal(t, MetricType(Count), m.Type)
	assert.Equal(t, "operation", m.Unit)
	assert.Equal(t, 86707, m.Pid)
	assert.Equal(t, "op", m.Tags[2].Name)
	assert.Equal(t, "summary", m.Tags[2].Value)
	assert.Equal(t, "file", m.Tags[1].Name)
	assert.Equal(t, "summary", m.Tags[1].Value)
	assert.Equal(t, "class", m.Tags[0].Name)
	assert.Equal(t, "open_close", m.Tags[0].Value)
	assert.Equal(t, util.ParseEventTime(1610682430.52), m.Time)

	test2 := `{"type":"metric","body":{"_metric":"fs.error","_metric_type":"counter","_value":3,"pid":86707,"op":"summary","file":"summary","class":"open_close","unit":"operation","_time":1610682430.52}}`
	m, err = parseJSONMetric([]byte(test2))
	assert.NoError(t, err)

	sort.Slice(m.Tags, func(i, j int) bool { return m.Tags[i].Name < m.Tags[j].Name })

	assert.Equal(t, "fs.error", m.Name)
	assert.Equal(t, float64(3), m.Value)
	assert.Equal(t, MetricType(Count), m.Type)
	assert.Equal(t, "operation", m.Unit)
	assert.Equal(t, 86707, m.Pid)
	assert.Equal(t, "op", m.Tags[2].Name)
	assert.Equal(t, "summary", m.Tags[2].Value)
	assert.Equal(t, "file", m.Tags[1].Name)
	assert.Equal(t, "summary", m.Tags[1].Value)
	assert.Equal(t, "class", m.Tags[0].Name)
	assert.Equal(t, "open_close", m.Tags[0].Value)
	assert.Equal(t, util.ParseEventTime(1610682430.52), m.Time)

	test3 := `{"foo":"bar"}`
	m, err = parseJSONMetric([]byte(test3))

	assert.Error(t, err)
}
