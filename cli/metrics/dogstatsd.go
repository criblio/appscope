package metrics

import (
	"bufio"
	"os"
	"strconv"
	"strings"

	"github.com/criblio/scope/util"
)

// ReadMetricsFromFile reads dogstatsd metrics from a file
func ReadMetricsFromFile(path string) (ret []Metric) {
	file, err := os.Open(path)
	util.CheckErrSprintf(err, "Invalid session. Command likely exited without capturing metric data.\nerror reading metrics from file: %v", err)
	defer file.Close()

	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		m := parseDogStatsd((scanner.Text()))
		ret = append(ret, m)
	}

	if err := scanner.Err(); err != nil {
		util.ErrAndExit("error scanning metrics file %s: %v", path, err)
	}
	return ret
}

// parseDogStatsd parses Scope's dogstatsd output
// Not general purpose. Ignores sampling which is allowed in the format.
// Assumes tags come after metric type
func parseDogStatsd(line string) (ret Metric) {
	lastIdx := 0

	newIdx := strings.IndexRune(line[lastIdx:], ':')
	if lastIdx < 0 {
		return
	}
	ret.Name = line[0:newIdx]
	lastIdx += newIdx + 1

	newIdx = strings.IndexRune(line[lastIdx:], '|')
	if lastIdx < 0 {
		return
	}
	var err error
	ret.Value, err = strconv.ParseFloat(line[lastIdx:lastIdx+newIdx], 64)
	if err != nil {
		return
	}
	lastIdx += newIdx + 1

	metricCodes := map[string]MetricType{
		"c": Count,
		"g": Gauge,
	}
	newIdx = strings.IndexRune(line[lastIdx:], '|')
	if lastIdx < 0 {
		return
	}
	ret.Type = metricCodes[line[lastIdx:lastIdx+newIdx]]
	lastIdx += newIdx + 2 // Skip # tags char as well

	tags := strings.Split(line[lastIdx:], ",")
	for _, t := range tags {
		tagParts := strings.Split(t, ":")
		if tagParts[0] == "unit" {
			ret.Unit = tagParts[1]
		} else {
			ret.Tags = append(ret.Tags, MetricTag{Name: tagParts[0], Value: tagParts[1]})
		}
	}
	return ret
}
