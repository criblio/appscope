package util

import (
	"errors"
	"fmt"
	"syscall"
	"time"
	"unsafe"

	"golang.org/x/sys/unix"
)

var (
	errEmptyMsg    = errors.New("empty message not supported")
	errMsgQCreate  = errors.New("message queue error during create")
	errMsgQOpen    = errors.New("message queue error during open")
	errMsgQGetAttr = errors.New("message queue error during get attributes")
	errMsgQUnlink  = errors.New("message queue error during unlink")
	errMsgQSendMsg = errors.New("message queue error during sending msg")
	errMsgQRecvMsg = errors.New("message queue error during receiving msg")
)

// Default values of
// - maximum number of messsages in a queue
// - maximum message size in a queue
// Details: https://man7.org/linux/man-pages/man7/mq_overview.7.html
const mqMaxMsgMax int = 10
const mqMaxMsgSize int = 8192

// Message queque attribute structure
// Details: https://man7.org/linux/man-pages/man3/mq_getattr.3.html
type messageQueueAttributes struct {
	Flags           int // Message queue flags: 0 or O_NONBLCOK
	MaxQueueSize    int // Max # of messages in queue
	MaxMessageSize  int // Max message size in bytes
	CurrentMessages int // # of messages currently in queue
}

type messageQueue struct {
	fd   int    // Message queue file descriptor
	name string // Message queue name
	cap  int    // Message queue capacity
}

type sendMessageQueue struct {
	messageQueue
}

type receiveMessageQueue struct {
	messageQueue
}

// getMQAttributes retrieves Message queue attributes from file descriptor
func getMQAttributes(fd int) (*messageQueueAttributes, error) {

	attr := new(messageQueueAttributes)

	// int syscall(SYS_mq_getsetattr, mqd_t mqdes, const struct mq_attr *newattr, struct mq_attr *oldattr)
	// Details: https://man7.org/linux/man-pages/man2/mq_getsetattr.2.html

	_, _, errno := unix.Syscall(unix.SYS_MQ_GETSETATTR,
		uintptr(fd),                   // mqdes
		uintptr(0),                    // newattr
		uintptr(unsafe.Pointer(attr))) // oldattr

	if errno != 0 {
		return nil, fmt.Errorf("%v %v", errMsgQGetAttr, errno)
	}

	return attr, nil
}

// openMsgQWriter open existing message queue with write only permissions
func openMsgQWriter(name string) (*sendMessageQueue, error) {
	unixName, err := unix.BytePtrFromString(name)
	if err != nil {
		return nil, err
	}

	// mqd_t mq_open(const char *name, int oflag);
	// Details: https://man7.org/linux/man-pages/man3/mq_open.3.html
	mqfd, _, errno := unix.Syscall(
		unix.SYS_MQ_OPEN,
		uintptr(unsafe.Pointer(unixName)), // name
		uintptr(unix.O_WRONLY),            // oflag
		0,                                 // unused
	)

	if errno != 0 {
		return nil, fmt.Errorf("%v %v", errMsgQOpen, errno)
	}

	fd := int(mqfd)

	// Get Message Queue MaxMessageSize
	attr, err := getMQAttributes(fd)
	if err != nil {
		return nil, errno
	}

	return &sendMessageQueue{
		messageQueue: messageQueue{
			fd:   fd,
			name: name,
			cap:  attr.MaxMessageSize,
		},
	}, nil

}

// newNonBlockMsgQReader creates non-blocking message queue with read only permissions
func newNonBlockMsgQReader(name string) (*receiveMessageQueue, error) {
	unixName, err := unix.BytePtrFromString(name)
	if err != nil {
		return nil, err
	}

	oldmask := syscall.Umask(0)
	// mqd_t mq_open(const char *name, int oflag, mode_t mode,struct mq_attr *attr)
	// Details: https://man7.org/linux/man-pages/man3/mq_open.3.html
	mqfd, _, errno := unix.Syscall6(
		unix.SYS_MQ_OPEN,
		uintptr(unsafe.Pointer(unixName)),                   // name
		uintptr(unix.O_CREAT|unix.O_NONBLOCK|unix.O_RDONLY), // oflag
		uintptr(0666),                                       // mode
		uintptr(unsafe.Pointer(&messageQueueAttributes{
			MaxQueueSize:   mqMaxMsgMax,
			MaxMessageSize: mqMaxMsgSize,
		})), // attr
		0, // unused
		0, // unused
	)
	syscall.Umask(oldmask)

	if errno != 0 {
		return nil, fmt.Errorf("%v %v", errMsgQCreate, errno)
	}

	return &receiveMessageQueue{
		messageQueue: messageQueue{
			fd:   int(mqfd),
			name: name,
			cap:  mqMaxMsgSize,
		},
	}, nil
}

// getAttributes retrieves message queue attributes
func (mq *messageQueue) getAttributes() (*messageQueueAttributes, error) {
	return getMQAttributes(mq.fd)
}

// close close message queue
func (mq *messageQueue) close() error {
	return unix.Close(int(mq.fd))
}

// unlink unlinks message queue
func (mq *receiveMessageQueue) unlink() error {
	unixName, err := unix.BytePtrFromString(mq.name)
	if err != nil {
		return err
	}

	_, _, errno := unix.Syscall(
		unix.SYS_MQ_UNLINK,
		uintptr(unsafe.Pointer(unixName)),
		0, // unused
		0, // unused
	)

	if errno != 0 {
		return fmt.Errorf("%v %v", errMsgQUnlink, errno)
	}

	return nil
}

// getBlockAbsTime retrieve the time which the call will block
func getBlockAbsTime(timeout time.Duration) uintptr {
	if timeout == 0 {
		return 0
	}

	ts := unix.NsecToTimespec(time.Now().Add(timeout).UnixNano())
	return uintptr(unsafe.Pointer(&ts))
}

// send data to message queue
func (mq *sendMessageQueue) send(msg []byte, timeout time.Duration) error {
	msgLen := len(msg)
	if msgLen == 0 {
		return errEmptyMsg
	}

	// int mq_timedsend(mqd_t mqdes, const char *msg_ptr, size_t msg_len, unsigned int msg_prio, const struct timespec *abs_timeout)
	// Details: https:///man7.org/linux/man-pages/man3/mq_send.3.html

	_, _, errno := unix.Syscall6(
		unix.SYS_MQ_TIMEDSEND,
		uintptr(mq.fd),                   // mqdes
		uintptr(unsafe.Pointer(&msg[0])), // msg_ptr
		uintptr(msgLen),                  // msg_len
		uintptr(0),                       // msg_prio
		getBlockAbsTime(timeout),         // abs_timeout
		0,                                // unused
	)

	if errno != 0 {
		return fmt.Errorf("%v %v", errMsgQSendMsg, errno)
	}
	return nil
}

// receive data from message queque
func (mq *receiveMessageQueue) receive(timeout time.Duration) ([]byte, error) {

	recvBuf := make([]byte, mq.cap)

	// ssize_t mq_timedreceive(mqd_t mqdes, char *restrict msg_ptr, size_t msg_len, unsigned int *restrict msg_prio, const struct timespec *restrict abs_timeout)
	// Details: https://man7.org/linux/man-pages/man3/mq_receive.3.html

	size, _, errno := unix.Syscall6(
		unix.SYS_MQ_TIMEDRECEIVE,
		uintptr(mq.fd),                       // mqdes
		uintptr(unsafe.Pointer(&recvBuf[0])), // msg_ptr
		uintptr(mq.cap),                      // msg_len
		uintptr(0),                           // msg_prio
		getBlockAbsTime(timeout),             // abs_timeout
		0,                                    // unused
	)

	if errno != 0 {
		return nil, fmt.Errorf("%v %v", errMsgQRecvMsg, errno)
	}

	return recvBuf[0:int(size)], nil
}
