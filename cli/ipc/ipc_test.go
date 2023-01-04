package ipc

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestMsgSeparatorMissing(t *testing.T) {
	sep := ipcGetMsgSeparatorIndex([]byte("test"))
	assert.Equal(t, sep, -1)
}

func TestMsgSeparatorPresentSingleNul(t *testing.T) {
	testMsg := "test\x00"
	searchIndx := len(testMsg) - 1
	sep := ipcGetMsgSeparatorIndex([]byte(testMsg))
	assert.Equal(t, sep, searchIndx)
}

func TestMsgSeparatorPresentDoubleNul(t *testing.T) {
	testMsg := "test\x00test\x00"
	searchIndx := len("test\x00") - 1
	sep := ipcGetMsgSeparatorIndex([]byte(testMsg))
	assert.Equal(t, sep, searchIndx)
}

func TestParsingIpcFrameOnlyMeta(t *testing.T) {
	testMsg := "{\"status\":200,\"uniq\":1645,\"remain\":28}\x00"
	resp, _, err := parseIpcFrame([]byte(testMsg))
	assert.NoError(t, err)
	assert.EqualValues(t, *resp.Status, 200)
	assert.EqualValues(t, *resp.Uniq, 1645)
	assert.EqualValues(t, *resp.Remain, 28)
}

func TestParsingIpcFrameMissingUnique(t *testing.T) {
	testMsg := "{\"status\":200,\"remain\":28}\x00"
	resp, _, err := parseIpcFrame([]byte(testMsg))
	assert.ErrorIs(t, err, errMissingMandatoryField)
	assert.EqualValues(t, *resp.Status, 200)
	assert.EqualValues(t, *resp.Remain, 28)
}

func TestParsingIpcFrameScopeMsg(t *testing.T) {
	testMsg := "{\"status\":200,\"uniq\":1645,\"remain\":28}\x00{\"status\":200,\"scoped\":true}"
	resp, scopeMsg, err := parseIpcFrame([]byte(testMsg))
	assert.NoError(t, err)
	assert.EqualValues(t, *resp.Status, 200)
	assert.EqualValues(t, *resp.Uniq, 1645)
	assert.EqualValues(t, *resp.Remain, 28)
	cmd := CmdGetScopeStatus{}
	err = cmd.UnmarshalResp(scopeMsg)
	assert.NoError(t, err)
	assert.EqualValues(t, *cmd.Response.Status, 200)
	assert.EqualValues(t, cmd.Response.Scoped, true)
}

func TestParsingIpcFrameScopeMsgMissingStatus(t *testing.T) {
	testMsg := "{\"status\":200,\"uniq\":1645,\"remain\":28}\x00{\"scoped\":true}"
	resp, scopeMsg, err := parseIpcFrame([]byte(testMsg))
	assert.NoError(t, err)
	assert.EqualValues(t, *resp.Status, 200)
	assert.EqualValues(t, *resp.Uniq, 1645)
	assert.EqualValues(t, *resp.Remain, 28)
	cmd := CmdGetScopeStatus{}
	err = cmd.UnmarshalResp(scopeMsg)
	assert.ErrorIs(t, err, errMissingMandatoryField)
}
