package pidfd

import (
	"syscall"

	"golang.org/x/sys/unix"
)

// pidfd, process file descriptor
type PidFd int

// Obtain a file descriptor that refers to a process.
func Open(pid int) (PidFd, error) {
	pFd, err := unix.PidfdOpen(pid, 0)
	return PidFd(pFd), err
}

// Duplicate a file descriptor from proces file descriptor
func (pFd PidFd) GetFd(fd int) (int, error) {
	newFd, err := unix.PidfdGetfd(int(pFd), fd, 0)
	return newFd, err
}

// Send a signal to a process file descriptor
func (pFd PidFd) SendSignal(signo syscall.Signal) error {
	return unix.PidfdSendSignal(int(pFd), signo, nil, 0)
}

// Close a process file descriptor
func (pFd PidFd) Close() error {
	return unix.Close(int(pFd))
}
