package ipc

import (
	"net/http"
)

// This file describes the meta request and response structures used in IPC communcation

type metaReqCmd int

// Must be inline with server, see: meta_req_t
const (
	metaReqJson metaReqCmd = iota
	metaReqJsonPartial
)

// meta request - message which will be transferred in message queue the metadata part
// Must be inline with server, see: createMetaResp
type metaRequest struct {
	// Request command
	Req metaReqCmd `json:"req" jsonschema:"required,const"`
	// Request unique number
	Uniq int `json:"uniq" jsonschema:"required"`
	// Remain remain number of bytes of data in msg
	Remain int `json:"remain" jsonschema:"required"`
}

// respStatus describes the status of request operation (received from library)
// IMPORTANT NOTE:
// responseStatus must be inline with library: ipc_resp_status_t
type respStatus int

const (
	ResponseOK             respStatus = http.StatusOK
	ResponseOKwithContent             = http.StatusPartialContent
	ResponseBadRequest                = http.StatusBadRequest
	ResponseServerError               = http.StatusInternalServerError
	ResponseNotImplemented            = http.StatusNotImplemented
)

// ipcResponse describes message which will be transferred in response message queue (from library to CLI)
// Must be inline with server, see: createMetaResp
type ipcResponse struct {
	// Response status
	Status *respStatus `json:"status" jsonschema:"required"`
	// Request Unique Identifter
	Uniq *int `json:"uniq" jsonschema:"required"`
	// Remain data
	Remain *int `json:"remain" jsonschema:"required"`
}
