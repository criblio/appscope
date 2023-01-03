package ipc

import (
	"errors"
	"fmt"
	"syscall"
	"time"
	"unsafe"

	"golang.org/x/sys/unix"
)

var (
	errEmptyMsg           = errors.New("empty message not supported")
	errMsgQCreate         = errors.New("message queue error during create")
	errMsgQGetAttr        = errors.New("message queue error during get attributes")
	errMsgQUnlink         = errors.New("message queue error during unlink")
	errMsgQSendMsg        = errors.New("message queue error during sending msg")
	errMsgQReceiveMsg     = errors.New("message queue error during receive msg")
	ErrMsgQTimeoutReceive = errors.New("message queue timeout - receive")
	ErrMsgQTimeoutSend    = errors.New("message queue timeout - send")
)

// Default values of
// - maximum number of messsages in a queue
// - maximum message size in a queue
// Details: https://man7.org/linux/man-pages/man7/mq_overview.7.html
const mqMaxMsgMax int = 4

const mqMaxMsgSize int = 128

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

// msgQAttributes retrieves Message queue attributes from file descriptor
func msgQAttributes(fd int) (*messageQueueAttributes, error) {

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

// msgQOpen opens message queue with specific name and flags
func msgQOpen(name string, flags int) (int, error) {
	unixName, err := unix.BytePtrFromString(name)
	if err != nil {
		return -1, err
	}
	oldmask := syscall.Umask(0)

	// mqd_t mq_open(const char *name, int oflag, mode_t mode,struct mq_attr *attr)
	// Details: https://man7.org/linux/man-pages/man3/mq_open.3.html
	mqfd, _, errno := unix.Syscall6(
		unix.SYS_MQ_OPEN,
		uintptr(unsafe.Pointer(unixName)), // name
		uintptr(flags),                    // oflag
		uintptr(0666),                     // mode
		uintptr(unsafe.Pointer(&messageQueueAttributes{
			MaxQueueSize:   mqMaxMsgMax,
			MaxMessageSize: mqMaxMsgSize,
		})), // attr
		0, // unused
		0, // unused
	)
	syscall.Umask(oldmask)

	if errno != 0 {
		return -1, fmt.Errorf("%v %v", errMsgQCreate, errno)
	}
	return int(mqfd), nil
}

// msgQSend sends msg to specific msgQueue described by fd
func msgQSend(fd int, msg []byte, timeout time.Duration) error {
	msgLen := len(msg)
	if msgLen == 0 {
		return errEmptyMsg
	}

	// int mq_timedsend(mqd_t mqdes, const char *msg_ptr, size_t msg_len, unsigned int msg_prio, const struct timespec *abs_timeout)
	// Details: https:///man7.org/linux/man-pages/man3/mq_send.3.html

	_, _, errno := unix.Syscall6(
		unix.SYS_MQ_TIMEDSEND,
		uintptr(fd),                      // mqdes
		uintptr(unsafe.Pointer(&msg[0])), // msg_ptr
		uintptr(msgLen),                  // msg_len
		uintptr(0),                       // msg_prio
		getBlockAbsTime(timeout),         // abs_timeout
		0,                                // unused
	)

	switch errno {
	case 0:
		return nil
	case syscall.ETIMEDOUT:
		return ErrMsgQTimeoutSend
	default:
		return fmt.Errorf("%v %v", errMsgQSendMsg, errno)
	}
}

// msgQReceive receive msg from specific msgQueue described by fd
func msgQReceive(fd int, capacity int, timeout time.Duration) ([]byte, error) {

	recvBuf := make([]byte, capacity)

	// ssize_t mq_timedreceive(mqd_t mqdes, char *restrict msg_ptr, size_t msg_len, unsigned int *restrict msg_prio, const struct timespec *restrict abs_timeout)
	// Details: https://man7.org/linux/man-pages/man3/mq_receive.3.html

	size, _, errno := unix.Syscall6(
		unix.SYS_MQ_TIMEDRECEIVE,
		uintptr(fd),                          // mqdes
		uintptr(unsafe.Pointer(&recvBuf[0])), // msg_ptr
		uintptr(capacity),                    // msg_len
		uintptr(0),                           // msg_prio
		getBlockAbsTime(timeout),             // abs_timeout
		0,                                    // unused
	)

	switch errno {
	case 0:
		return recvBuf[0:int(size)], nil
	case syscall.ETIMEDOUT:
		return nil, ErrMsgQTimeoutReceive
	default:
		return nil, fmt.Errorf("%v %v", errMsgQReceiveMsg, errno)
	}
}

// msgQUnlink unlinks specific message queue
func msgQUnlink(name string) error {
	unixName, err := unix.BytePtrFromString(name)
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

// newMsgQReader creates message queue with write only permissions
func newMsgQWriter(name string) (*sendMessageQueue, error) {
	mqfd, err := msgQOpen(name, unix.O_CREAT|unix.O_WRONLY|unix.O_EXCL)
	if err != nil {
		return nil, err
	}
	return &sendMessageQueue{
		messageQueue: messageQueue{
			fd:   mqfd,
			name: name,
			cap:  mqMaxMsgSize,
		},
	}, nil
}

// newMsgQReader creates message queue with read only permissions
func newMsgQReader(name string) (*receiveMessageQueue, error) {
	mqfd, err := msgQOpen(name, unix.O_CREAT|unix.O_RDONLY|unix.O_EXCL)
	if err != nil {
		return nil, err
	}
	return &receiveMessageQueue{
		messageQueue: messageQueue{
			fd:   mqfd,
			name: name,
			cap:  mqMaxMsgSize,
		},
	}, nil
}

// getAttributes retrieves message queue attributes
func (mq *messageQueue) getAttributes() (*messageQueueAttributes, error) {
	return msgQAttributes(mq.fd)
}

// destroy destroys (closes + unlinks) message queue
func (mq *messageQueue) destroy() error {

	err := unix.Close(int(mq.fd))
	// This is not obvious sometimes unlink fails after first try
	for i := 1; i < 5; i++ {
		err = msgQUnlink(mq.name)
		if err != nil {
			time.Sleep(3 * time.Second)
		} else {
			return nil
		}
	}
	return err

}

// getBlockAbsTime retrieve the absolute time which the call (send/receive) will block
func getBlockAbsTime(timeout time.Duration) uintptr {
	if timeout == 0 {
		return 0
	}

	ts := unix.NsecToTimespec(time.Now().Add(timeout).UnixNano())
	return uintptr(unsafe.Pointer(&ts))
}

// send data to message queue
func (mq *sendMessageQueue) send(msg []byte, timeout time.Duration) error {
	return msgQSend(mq.fd, msg, timeout)
}

// receive data from message queque
func (mq *receiveMessageQueue) receive(timeout time.Duration) ([]byte, error) {
	return msgQReceive(mq.fd, mq.cap, timeout)
}
