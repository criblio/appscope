package ipc

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"math/rand"
	"os"
	"syscall"
	"time"
)

// ipc structure representes Inter Process Communication object
type ipcObj struct {
	sender   *sendMessageQueue    // Message queue used to send messages
	receiver *receiveMessageQueue // Message queue used to receive messages
	ipcSame  bool                 // Indicator if IPC namespace switch occured
	mqUniqId int                  // Unique ID used in request/response
}

var (
	errCreateMsgQWriter          = errors.New("create writer message queue")
	errCreateMsgQReader          = errors.New("create reader message queue")
	errRequest                   = errors.New("error with sending request to PID")
	errFrameInconsistentUniqId   = errors.New("frame error inconsistent unique id during transmission")
	errFrameInconsistentDataSize = errors.New("frame error inconsistent data size during transmssion")
)

const ipcComTimeout time.Duration = 100 * time.Millisecond

// ipcDispatcher dispatches IPC communication to the process specified by the pid.
// Returns the bytes array
func ipcDispatcher(scopeReq []byte, pid int) (*IpcResponseCtx, error) {
	ipc, err := newIPC(pid)
	if err != nil {
		return nil, err
	}
	defer ipc.destroyIPC()

	err = ipc.sendIpcRequest(scopeReq)
	if err != nil {
		return nil, fmt.Errorf("%v %v", errRequest, pid)
	}
	return ipc.receiveIpcResponse()

	// resp, ipcStatus, err := ipc.receiveIpcResponse()
	// var prettyJSON bytes.Buffer
	// fmt.Println("Dispatcher ipc response status", ipcStatus)

	// if errDebug := json.Indent(&prettyJSON, []byte(resp), "", "    "); errDebug != nil {
	// 	fmt.Println("Error with indent data")
	// 	return resp, err
	// }
	// fmt.Println("Scope msg", prettyJSON.String())
	// Handle message queue response from application
}

// nsnewMsgQWriter creates an IPC writer structure with switching effective uid and gid
func nsnewMsgQWriter(name string, nsUid int, nsGid int, restoreUid int, restoreGid int) (*sendMessageQueue, error) {

	if err := syscall.Setegid(nsGid); err != nil {
		return nil, err
	}
	if err := syscall.Seteuid(nsUid); err != nil {
		return nil, err
	}

	sender, err := newMsgQWriter(name)
	if err != nil {
		return nil, err
	}

	if err := syscall.Seteuid(restoreUid); err != nil {
		return nil, err
	}

	if err := syscall.Setegid(restoreGid); err != nil {
		return nil, err
	}

	return sender, err
}

// nsnewMsgQReader creates an IPC reader structure with switching effective uid and gid
func nsnewMsgQReader(name string, nsUid int, nsGid int, restoreUid int, restoreGid int) (*receiveMessageQueue, error) {

	if err := syscall.Setegid(nsGid); err != nil {
		return nil, err
	}
	if err := syscall.Seteuid(nsUid); err != nil {
		return nil, err
	}

	receiver, err := newMsgQReader(name)
	if err != nil {
		return nil, err
	}

	if err := syscall.Seteuid(restoreUid); err != nil {
		return nil, err
	}

	if err := syscall.Setegid(restoreGid); err != nil {
		return nil, err
	}

	return receiver, err
}

// newIPC creates an IPC object designated for communication with specific PID
func newIPC(pid int) (*ipcObj, error) {
	restoreGid := os.Getegid()
	restoreUid := os.Geteuid()

	ipcSame, err := ipcNsIsSame(pid)
	if err != nil {
		return nil, err
	}

	// Retrieve information about user and group id
	nsUid, err := ipcNsTranslateUidFromPid(restoreUid, pid)
	if err != nil {
		return nil, err
	}
	nsGid, err := ipcNsTranslateGidFromPid(restoreGid, pid)
	if err != nil {
		return nil, err
	}

	// Retrieve information about process namespace PID
	_, ipcPid, err := ipcNsLastPidFromPId(pid)
	if err != nil {
		return nil, err
	}

	// Switch IPC namespace if needed
	if !ipcSame {
		if err := ipcNsSet(pid); err != nil {
			return nil, err
		}
	}

	//  Create message queue to Write into it
	sender, err := nsnewMsgQWriter(fmt.Sprintf("ScopeIPCIn.%d", ipcPid), nsUid, nsGid, restoreUid, restoreGid)
	if err != nil {
		if !ipcSame {
			ipcNsRestore()
		}
		return nil, errCreateMsgQWriter
	}

	// Create message queue to Read from it
	receiver, err := nsnewMsgQReader(fmt.Sprintf("ScopeIPCOut.%d", ipcPid), nsUid, nsGid, restoreUid, restoreGid)
	if err != nil {
		sender.destroy()
		if !ipcSame {
			ipcNsRestore()
		}
		return nil, errCreateMsgQReader
	}

	return &ipcObj{sender: sender, receiver: receiver, ipcSame: ipcSame, mqUniqId: rand.Intn(9999)}, nil
}

// destroyIPC destroys an IPC object
func (ipc *ipcObj) destroyIPC() {
	ipc.sender.destroy()
	ipc.receiver.destroy()
	if !ipc.ipcSame {
		ipcNsRestore()
	}
}

// receive receive the message from the process endpoint
func (ipc *ipcObj) receive() ([]byte, error) {
	return ipc.receiver.receive(ipcComTimeout)
}

// send sends the message to the process endpoint
func (ipc *ipcObj) send(msg []byte) error {
	return ipc.sender.send(msg, ipcComTimeout)
}

// metadata in ipc frame
func (ipc *ipcObj) frameMeta(remainBytes int) ([]byte, error) {
	metadata, err := json.Marshal(metaRequest{Req: metaReqJsonPartial, Uniq: ipc.mqUniqId, Remain: remainBytes})
	if err != nil {
		return metadata, err
	}
	// Ensure that metadata is nul terminated
	metadata = append(metadata, 0)
	return metadata, nil
}

// prepareFrame preparet the scope message
func (ipc *ipcObj) prepareFrame(remain int, scopeData []byte, offset int) ([]byte, int, error) {
	frame, _ := ipc.frameMeta(remain)
	maxPossibleDataLen := ipc.sender.cap - len(frame) - 1
	if maxPossibleDataLen < 0 {
		return frame, -1, errRequest
	}

	scopeFrameDataSize := maxPossibleDataLen
	if remain < maxPossibleDataLen {
		scopeFrameDataSize = remain
	}

	frameScopeData := scopeData[offset : offset+scopeFrameDataSize]

	frame = append(frame, frameScopeData...)
	frame = append(frame, 0)

	return frame, scopeFrameDataSize, nil
}

// sendIpcRequest puts the request in message queue (the message can be splited by the frames)
func (ipc *ipcObj) sendIpcRequest(scopeMsg []byte) error {
	scopeMsgDataRemain := len(scopeMsg)
	var scopeMsgOffset int
	for scopeMsgDataRemain != 0 {

		frame, dataSize, err := ipc.prepareFrame(scopeMsgDataRemain, scopeMsg, scopeMsgOffset)
		if err != nil {
			return err
		}

		scopeMsgOffset += dataSize
		scopeMsgDataRemain -= dataSize

		// Handling the last frame
		//  - change the request type metaReqJsonPartial -> metaReqJson
		if scopeMsgDataRemain == 0 {
			frame = bytes.Replace(frame, []byte("req\":1"), []byte("req\":0"), 1)
		}

		// Retry mechanism ?
		err = ipc.send(frame)
		if err != nil {
			return err
		}

	}
	return nil
}

// ipcGetMsgSeparatorIndex returns index of Meta and Scope message separator
func ipcGetMsgSeparatorIndex(msg []byte) int {
	return bytes.Index(msg, []byte("\x00"))
}

// parseIpcFrame parses single frame in IPC communication
// returns frame metadata, frame scopeData, error
func (ipc *ipcObj) parseIpcFrame(msg []byte) (ipcResponse, []byte, error) {
	var frameResp ipcResponse
	var err error

	// Only metadata case
	sepIndx := ipcGetMsgSeparatorIndex(msg)
	if sepIndx == -1 {
		err = json.Unmarshal(msg, &frameResp)
		if err != nil {
			return frameResp, nil, err
		}
		return frameResp, nil, err
	}

	metadata := msg[0:sepIndx]
	err = json.Unmarshal(metadata, &frameResp)
	if err != nil {
		return frameResp, nil, nil
	}

	// Skip the separator
	scopeData := msg[sepIndx+1:]

	return frameResp, scopeData, nil
}

// receiveIpcResponse receives the whole message from process endpoint
func (ipc *ipcObj) receiveIpcResponse() (*IpcResponseCtx, error) {
	listenForResponseTransmission := true
	var scopeMsg []byte
	var lastRespStatus respStatus
	firstFrame := true
	var previousFrameRemainBytes int

	for listenForResponseTransmission {

		msg, err := ipc.receive()
		if err != nil {
			return nil, err
		}

		ipcFrameMeta, scopeFrame, err := ipc.parseIpcFrame(msg)
		if err != nil {
			return nil, err
		}

		// Validate unique identifier for transmission
		if ipcFrameMeta.Uniq != ipc.mqUniqId {
			return nil, errFrameInconsistentUniqId
		}
		// Validate data remaining bytes in transmission
		if (!firstFrame) && ipcFrameMeta.Remain > previousFrameRemainBytes {
			return nil, errFrameInconsistentDataSize
		}

		previousFrameRemainBytes = ipcFrameMeta.Remain
		lastRespStatus = ipcFrameMeta.Status

		// Continue to listen if we still expect data
		if lastRespStatus != ResponseOKwithContent {
			listenForResponseTransmission = false
		}
		scopeMsg = append(scopeMsg, scopeFrame...)
		firstFrame = false
	}

	return &IpcResponseCtx{scopeMsg, lastRespStatus}, nil
}
