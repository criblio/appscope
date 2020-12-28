package metrics

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestParseDogStatsd(t *testing.T) {
	test1 := `proc.cpu:4339|c|#proc:sleep,pid:78871,host:8197a8bdc115,unit:microsecond`
	m := parseDogStatsd(test1)

	assert.Equal(t, "proc.cpu", m.Name)
	assert.Equal(t, float64(4339), m.Value)
	assert.Equal(t, MetricType(Count), m.Type)
	assert.Equal(t, "microsecond", m.Unit)
	assert.Equal(t, "proc", m.Tags[0].Name)
	assert.Equal(t, "sleep", m.Tags[0].Value)
	assert.Equal(t, "pid", m.Tags[1].Name)
	assert.Equal(t, "78871", m.Tags[1].Value)
	assert.Equal(t, "host", m.Tags[2].Name)
	assert.Equal(t, "8197a8bdc115", m.Tags[2].Value)

}
