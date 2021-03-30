package flows

import (
	"os"
	"sort"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestParseFlowFileName(t *testing.T) {
	files := []string{
		"asdf",
		"14942_65.8.158.81:443_172.17.0.2:42694.out",
		"14942_65.8.158.81:443_172.17.0.2:42694.in",
	}
	_, err := parseFlowFileName(files[0])
	assert.Error(t, err)
	flow, err := parseFlowFileName(files[1])
	assert.NoError(t, err)
	assert.Equal(t, Flow{
		Pid:      14942,
		HostIP:   "172.17.0.2",
		PeerIP:   "65.8.158.81",
		PeerPort: 443,
		HostPort: 42694,
		OutFile:  "14942_65.8.158.81:443_172.17.0.2:42694.out",
	}, flow)
	flow, err = parseFlowFileName(files[2])
	assert.NoError(t, err)
	assert.Equal(t, Flow{
		Pid:      14942,
		HostIP:   "172.17.0.2",
		PeerIP:   "65.8.158.81",
		PeerPort: 443,
		HostPort: 42694,
		InFile:   "14942_65.8.158.81:443_172.17.0.2:42694.in",
	}, flow)
}

func TestGetFlowFiles(t *testing.T) {
	os.Mkdir("testflows", 0755)
	flows, err := getFlowFiles("./testflows")
	assert.NoError(t, err)
	assert.Equal(t, []Flow{}, flows.List())

	// Will error if there's an improperly named file
	os.Create("testflows/foo")
	_, err = getFlowFiles("./testflows")
	assert.Error(t, err)
	os.Remove("testflows/foo")

	RestoreAssets("testflows", "")
	flows, err = getFlowFiles("./testflows/payloads")
	flowList := flows.List()
	sort.Slice(flowList, func(i, j int) bool { return flowList[i].BytesSent < flowList[j].BytesSent })
	assert.Equal(t, 548213, flowList[0].Pid)
	assert.Equal(t, "13.226.220.53", flowList[0].PeerIP)
	assert.Equal(t, 443, flowList[0].PeerPort)
	assert.Equal(t, "548213_13.226.220.53:443_172.17.0.2:50466.out", flowList[0].OutFile)
	os.RemoveAll("testflows")

}

func TestGetFlowEvents(t *testing.T) {
	os.Mkdir("testflows", 0755)
	RestoreAssets("testflows", "")
	file, err := os.Open("testflows/events.json")
	assert.NoError(t, err)
	flows, err := getFlowEvents(file)
	flowList := flows.List()
	sort.Slice(flowList, func(i, j int) bool { return flowList[i].BytesSent < flowList[j].BytesSent })
	f := flowList[0]
	assert.NoError(t, err)
	assert.Equal(t, "172.17.0.2", f.HostIP)
	assert.Equal(t, 50466, f.HostPort)
	assert.Equal(t, "13.226.220.53", f.PeerIP)
	assert.Equal(t, 443, f.PeerPort)
	assert.Equal(t, 55439, f.BytesReceived)
	os.RemoveAll("testflows")
}

func TestGetFlows(t *testing.T) {
	os.Mkdir("testflows", 0755)
	RestoreAssets("testflows", "")
	file, err := os.Open("testflows/events.json")
	assert.NoError(t, err)

	flows, err := GetFlows("./testflows/payloads", file)
	flowList := flows.List()
	sort.Slice(flowList, func(i, j int) bool { return flowList[i].BytesSent < flowList[j].BytesSent })
	f := flowList[1]
	assert.NoError(t, err)
	assert.Equal(t, 340, f.BytesReceived)
	assert.Equal(t, "", f.Protocol)
	os.RemoveAll("testflows")
}
