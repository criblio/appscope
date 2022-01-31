package flows

import (
	"errors"
	"os"
	"sort"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestParseFlowFileName(t *testing.T) {

	tests := []struct {
		name        string
		filename    string
		expErr      interface{}
		expPid      int
		expHostIP   string
		expPeerIP   string
		expPeerPort int
		expHostPort int
		expInFile   string
		expOutFile  string
	}{
		{
			name:     "random string",
			filename: "asdf",
			expErr:   errors.New("error parsing filename: asdf"),
		},
		{
			name:        "ip and port outfile",
			filename:    "14942_65.8.158.81:443_172.17.0.2:42694.out",
			expErr:      nil,
			expPid:      14942,
			expHostIP:   "172.17.0.2",
			expPeerIP:   "65.8.158.81",
			expPeerPort: 443,
			expHostPort: 42694,
			expInFile:   "",
			expOutFile:  "14942_65.8.158.81:443_172.17.0.2:42694.out",
		},
		{
			name:        "ip and port infile",
			filename:    "14942_65.8.158.81:443_172.17.0.2:42694.in",
			expErr:      nil,
			expPid:      14942,
			expHostIP:   "172.17.0.2",
			expPeerIP:   "65.8.158.81",
			expPeerPort: 443,
			expHostPort: 42694,
			expInFile:   "14942_65.8.158.81:443_172.17.0.2:42694.in",
			expOutFile:  "",
		},
		{
			name:        "af_unix outfile",
			filename:    "44603_af_unix:198372_af_unix:198371.out",
			expErr:      nil,
			expPid:      44603,
			expHostIP:   "af_unix",
			expPeerIP:   "af_unix",
			expPeerPort: 198372,
			expHostPort: 198371,
			expInFile:   "",
			expOutFile:  "44603_af_unix:198372_af_unix:198371.out",
		},
		{
			name:        "af_unix infile",
			filename:    "44603_af_unix:198372_af_unix:198371.in",
			expErr:      nil,
			expPid:      44603,
			expHostIP:   "af_unix",
			expPeerIP:   "af_unix",
			expPeerPort: 198372,
			expHostPort: 198371,
			expInFile:   "44603_af_unix:198372_af_unix:198371.in",
			expOutFile:  "",
		},
		{
			name:        "tlsrx infile",
			filename:    "44603_tlsrx:0_tlsrx:0.in",
			expErr:      nil,
			expPid:      44603,
			expHostIP:   "tlsrx",
			expPeerIP:   "tlsrx",
			expPeerPort: 0,
			expHostPort: 0,
			expInFile:   "44603_tlsrx:0_tlsrx:0.in",
			expOutFile:  "",
		},
		{
			name:        "tlstx outfile",
			filename:    "44603_tlstx:0_tlstx:0.out",
			expErr:      nil,
			expPid:      44603,
			expHostIP:   "tlstx",
			expPeerIP:   "tlstx",
			expPeerPort: 0,
			expHostPort: 0,
			expInFile:   "",
			expOutFile:  "44603_tlstx:0_tlstx:0.out",
		},
		{
			name:        "netrx infile",
			filename:    "44603_netrx:0_netrx:0.in",
			expErr:      nil,
			expPid:      44603,
			expHostIP:   "netrx",
			expPeerIP:   "netrx",
			expPeerPort: 0,
			expHostPort: 0,
			expInFile:   "44603_netrx:0_netrx:0.in",
			expOutFile:  "",
		},
		{
			name:        "nettx outfile",
			filename:    "44603_nettx:0_nettx:0.out",
			expErr:      nil,
			expPid:      44603,
			expHostIP:   "nettx",
			expPeerIP:   "nettx",
			expPeerPort: 0,
			expHostPort: 0,
			expInFile:   "",
			expOutFile:  "44603_nettx:0_nettx:0.out",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			res, err := parseFlowFileName(tc.filename)
			if err != nil {
				assert.Equal(t, tc.expErr, err)
				return
			}
			assert.Equal(t, tc.expPid, res.Pid)
			assert.Equal(t, tc.expHostIP, res.HostIP)
			assert.Equal(t, tc.expPeerIP, res.PeerIP)
			assert.Equal(t, tc.expPeerPort, res.PeerPort)
			assert.Equal(t, tc.expHostPort, res.HostPort)
			assert.Equal(t, tc.expInFile, res.InFile)
			assert.Equal(t, tc.expOutFile, res.OutFile)
		})
	}
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
	assert.NoError(t, err)
	flowList := flows.List()
	sort.Slice(flowList, func(i, j int) bool { return flowList[i].BytesSent < flowList[j].BytesSent })
	assert.Equal(t, 27752, flowList[1].Pid)
	assert.Equal(t, "99.84.74.114", flowList[1].PeerIP)
	assert.Equal(t, 443, flowList[1].PeerPort)
	assert.Equal(t, "27752_99.84.74.114:443_10.8.107.159:33716.out", flowList[1].OutFile)
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
	assert.Equal(t, "10.8.107.159", f.HostIP)
	assert.Equal(t, 33716, f.HostPort)
	assert.Equal(t, "99.84.74.114", f.PeerIP)
	assert.Equal(t, 443, f.PeerPort)
	assert.Equal(t, 65626, f.BytesReceived)
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
	assert.Equal(t, 59984, f.BytesReceived)
	assert.Equal(t, "", f.Protocol)
	os.RemoveAll("testflows")
}
