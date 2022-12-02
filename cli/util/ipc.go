package util

import (
	"errors"
	"fmt"
	"os"
	"syscall"
	"time"
)

// ipc structure representes Inter Process Communication object
type ipcObj struct {
	sender    *sendMessageQueue    // Message queue used to send messages
	receiver  *receiveMessageQueue // Message queue used to receive messages
	ipcSwitch bool                 // Indicator if IPC switch occured
}

var (
	errMissingProcMsgQueue = errors.New("missing message queue from PID")
	errMissingResponse     = errors.New("missing response from PID")
)

// ipcGetScopeStatus dispatches cmd to the process specified by the pid.
// Returns the byte answer from scoped process endpoint.
func ipcGetScopeStatus(pid int) ([]byte, error) {
	return ipcDispatcher(cmdGetScopeStatus, pid)
}

// ipcDispatcher dispatches cmd to the process specified by the pid.
// Returns the byte answer from scoped process endpoint.
func ipcDispatcher(cmd ipcCmd, pid int) ([]byte, error) {
	var answer []byte
	var responseReceived bool

	ipc, err := newIPC(pid)
	if err != nil {
		return answer, err
	}
	defer ipc.destroyIPC()

	if err := ipc.send(cmd.byte()); err != nil {
		return answer, err
	}

	// TODO: Ugly hack but we need to wait for answer from process
	for i := 0; i < 5000; i++ {
		if !ipc.empty() {
			responseReceived = true
			break
		}
		time.Sleep(time.Millisecond)
	}

	// Missing response
	// The message queue on the application side exists but we are unable to receive
	// an answer from it
	if !responseReceived {
		return answer, fmt.Errorf("%v %v", errMissingResponse, pid)
	}

	return ipc.receive()
}

// nsnewNonBlockMsgQReader creates an IPC structure with switching effective uid and gid
func nsnewNonBlockMsgQReader(name string, nsUid int, nsGid int, restoreUid int, restoreGid int) (*receiveMessageQueue, error) {

	if err := syscall.Setegid(nsGid); err != nil {
		return nil, err
	}
	if err := syscall.Seteuid(nsUid); err != nil {
		return nil, err
	}

	receiver, err := newNonBlockMsgQReader(name)
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

	ipcSame, err := namespaceSameIpc(pid)
	if err != nil {
		return nil, err
	}

	// Retrieve information about user nad group id
	nsUid, err := pidNsTranslateUid(restoreUid, pid)
	if err != nil {
		return nil, err
	}
	nsGid, err := pidNsTranslateGid(restoreGid, pid)
	if err != nil {
		return nil, err
	}

	// Retrieve information about process namespace PID
	_, ipcPid, err := pidLastNsPid(pid)
	if err != nil {
		return nil, err
	}

	// Switch IPC if neeeded
	if !ipcSame {
		if err := namespaceSwitchIPC(pid); err != nil {
			return nil, err
		}
	}

	// Try to open proc message queue
	sender, err := openMsgQWriter(fmt.Sprintf("ScopeIPCIn.%d", ipcPid))
	if err != nil {
		namespaceRestoreIPC()
		return nil, errMissingProcMsgQueue
	}

	// Try to create own message queue
	receiver, err := nsnewNonBlockMsgQReader(fmt.Sprintf("ScopeIPCOut.%d", ipcPid), nsUid, nsGid, restoreUid, restoreGid)
	if err != nil {
		sender.close()
		namespaceRestoreIPC()
		return nil, err
	}

	return &ipcObj{sender: sender, receiver: receiver, ipcSwitch: ipcSame}, nil
}

// destroyIPC destroys an IPC object
func (ipc *ipcObj) destroyIPC() {
	ipc.sender.close()
	ipc.receiver.close()
	ipc.receiver.unlink()
	if ipc.ipcSwitch {
		namespaceRestoreIPC()
	}
}

// receive receive the message from the process endpoint
func (ipc *ipcObj) receive() ([]byte, error) {
	return ipc.receiver.receive(0)
}

// empty checks if receiver message queque is empty
func (ipc *ipcObj) empty() bool {
	atr, err := ipc.receiver.getAttributes()
	if err != nil {
		return true
	}
	return atr.CurrentMessages == 0
}

// send sends the message to the process endpoint
func (ipc *ipcObj) send(msg []byte) error {
	return ipc.sender.send(msg, 0)
}
