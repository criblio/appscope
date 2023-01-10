package ipc

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"golang.org/x/sys/unix"
)

func TestNewMqReader(t *testing.T) {
	const mqName string = "goTestMq"

	msgQ, err := newMsgQReader(mqName)
	assert.NoError(t, err)
	assert.NotNil(t, msgQ)
	size, err := msgQ.size()
	assert.Zero(t, size)
	assert.NoError(t, err)
	err = msgQ.destroy()
	assert.NoError(t, err)
}

func TestNewMqWriter(t *testing.T) {
	const mqName string = "goTestMq"

	msgQ, err := newMsgQWriter(mqName)
	assert.NoError(t, err)
	assert.NotNil(t, msgQ)
	size, err := msgQ.size()
	assert.Zero(t, size)
	assert.NoError(t, err)
	err = msgQ.destroy()
	assert.NoError(t, err)
}

func TestNoDuplicate(t *testing.T) {
	const mqNameR string = "goTestMqR"
	const mqNameW string = "goTestMqW"

	msgQW, err := newMsgQWriter(mqNameW)
	assert.NoError(t, err)
	assert.NotNil(t, msgQW)
	msgQWNew, err := newMsgQWriter(mqNameW)
	assert.Error(t, err)
	assert.Nil(t, msgQWNew)

	msgQR, err := newMsgQReader(mqNameR)
	assert.NoError(t, err)
	assert.NotNil(t, msgQR)
	msgQRNew, err := newMsgQReader(mqNameR)
	assert.Error(t, err)
	assert.Nil(t, msgQRNew)
}

func TestCommunicationMqReader(t *testing.T) {
	const mqName string = "goTestMqReader"
	byteTestMsg := []byte("Lorem Ipsum")

	msgQReader, err := newMsgQReader(mqName)
	assert.NoError(t, err)
	assert.NotNil(t, msgQReader)

	msgQWriteFd, err := msgQOpen(mqName, unix.O_WRONLY)
	assert.NoError(t, err)
	assert.NotEqual(t, msgQWriteFd, -1)

	// Try to read from empty queue
	data, err := msgQReader.receive()
	assert.Nil(t, data)
	assert.Error(t, err)

	// Empty message send to w
	err = msgQSend(msgQWriteFd, []byte(""))
	assert.Error(t, err)
	size, err := msgQReader.size()
	assert.NoError(t, err)
	assert.Zero(t, size)
	size, err = msgQCurrentMessages(msgQWriteFd)
	assert.NoError(t, err)
	assert.Zero(t, size)

	// Normal message send
	err = msgQSend(msgQWriteFd, byteTestMsg)
	assert.NoError(t, err)
	size, err = msgQReader.size()
	assert.NoError(t, err)
	assert.Equal(t, size, 1)
	size, err = msgQCurrentMessages(msgQWriteFd)
	assert.NoError(t, err)
	assert.Equal(t, size, 1)

	data, err = msgQReader.receive()
	assert.Equal(t, data, byteTestMsg)
	assert.NotNil(t, data)
	assert.NoError(t, err)
	size, err = msgQReader.size()
	assert.NoError(t, err)
	assert.Zero(t, size)
	size, err = msgQCurrentMessages(msgQWriteFd)
	assert.NoError(t, err)
	assert.Zero(t, size)

	err = unix.Close(msgQWriteFd)
	assert.NoError(t, err)

	err = msgQReader.destroy()
	assert.NoError(t, err)
}

func TestCommunicationMqWriter(t *testing.T) {
	const mqName string = "goTestMqReader"
	byteTestMsg := []byte("Lorem Ipsum")

	msgQWriter, err := newMsgQWriter(mqName)
	assert.NoError(t, err)
	assert.NotNil(t, msgQWriter)

	msgQReaderFd, err := msgQOpen(mqName, unix.O_RDONLY)
	assert.NoError(t, err)
	assert.NotEqual(t, msgQReaderFd, -1)

	// Empty message send to writer message queue
	err = msgQWriter.send([]byte(""))
	assert.Error(t, err)
	size, err := msgQWriter.size()
	assert.NoError(t, err)
	assert.Zero(t, size)
	size, err = msgQCurrentMessages(msgQReaderFd)
	assert.NoError(t, err)
	assert.Zero(t, size)

	// Normal message send
	err = msgQWriter.send(byteTestMsg)
	assert.NoError(t, err)
	size, err = msgQWriter.size()
	assert.NoError(t, err)
	assert.Equal(t, size, 1)
	size, err = msgQCurrentMessages(msgQReaderFd)
	assert.NoError(t, err)
	assert.Equal(t, size, 1)

	data, err := msgQReceive(msgQReaderFd, 512)
	assert.Equal(t, data, byteTestMsg)
	assert.NotNil(t, data)
	assert.NoError(t, err)
	size, err = msgQWriter.size()
	assert.NoError(t, err)
	assert.Zero(t, size)
	size, err = msgQCurrentMessages(msgQReaderFd)
	assert.NoError(t, err)
	assert.Zero(t, size)

	err = unix.Close(msgQReaderFd)
	assert.NoError(t, err)

	err = msgQWriter.destroy()
	assert.NoError(t, err)
}
