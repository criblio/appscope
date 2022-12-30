package ipc

import (
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"golang.org/x/sys/unix"
)

func TestNewMqReader(t *testing.T) {
	const mqName string = "goTestMq"

	msgQ, err := newMsgQReader(mqName)
	assert.NoError(t, err)
	assert.NotNil(t, msgQ)
	atr, err := msgQ.getAttributes()
	assert.NotNil(t, atr)
	assert.NoError(t, err)
	err = msgQ.destroy()
	assert.NoError(t, err)
}

func TestNewMqWriter(t *testing.T) {
	const mqName string = "goTestMq"

	msgQ, err := newMsgQWriter(mqName)
	assert.NoError(t, err)
	assert.NotNil(t, msgQ)
	atr, err := msgQ.getAttributes()
	assert.NotNil(t, atr)
	assert.NoError(t, err)
	err = msgQ.destroy()
	assert.NoError(t, err)
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
	data, err := msgQReader.receive(time.Microsecond)
	assert.Nil(t, data)
	assert.Error(t, err)

	// Empty message send to w
	err = msgQSend(msgQWriteFd, []byte(""), time.Microsecond)
	assert.Error(t, err)
	atr, err := msgQReader.getAttributes()
	assert.NotNil(t, atr)
	assert.NoError(t, err)
	assert.Zero(t, atr.CurrentMessages)
	atr, err = msgQAttributes(msgQWriteFd)
	assert.NotNil(t, atr)
	assert.NoError(t, err)
	assert.Zero(t, atr.CurrentMessages)

	// Normal message send
	err = msgQSend(msgQWriteFd, byteTestMsg, time.Microsecond)
	assert.NoError(t, err)
	atr, err = msgQReader.getAttributes()
	assert.NotNil(t, atr)
	assert.NoError(t, err)
	assert.Equal(t, atr.CurrentMessages, 1)
	atr, err = msgQAttributes(msgQWriteFd)
	assert.NotNil(t, atr)
	assert.NoError(t, err)
	assert.Equal(t, atr.CurrentMessages, 1)

	data, err = msgQReader.receive(time.Microsecond)
	assert.Equal(t, data, byteTestMsg)
	assert.NotNil(t, data)
	assert.NoError(t, err)
	atr, err = msgQReader.getAttributes()
	assert.NotNil(t, atr)
	assert.NoError(t, err)
	assert.Zero(t, atr.CurrentMessages)
	atr, err = msgQAttributes(msgQWriteFd)
	assert.NotNil(t, atr)
	assert.NoError(t, err)
	assert.Zero(t, atr.CurrentMessages)

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
	err = msgQWriter.send([]byte(""), time.Microsecond)
	assert.Error(t, err)
	atr, err := msgQWriter.getAttributes()
	assert.NotNil(t, atr)
	assert.NoError(t, err)
	assert.Zero(t, atr.CurrentMessages)
	atr, err = msgQAttributes(msgQReaderFd)
	assert.NotNil(t, atr)
	assert.NoError(t, err)
	assert.Zero(t, atr.CurrentMessages)

	// Normal message send
	err = msgQWriter.send(byteTestMsg, time.Microsecond)
	assert.NoError(t, err)
	atr, err = msgQWriter.getAttributes()
	assert.NotNil(t, atr)
	assert.NoError(t, err)
	assert.Equal(t, atr.CurrentMessages, 1)
	atr, err = msgQAttributes(msgQReaderFd)
	assert.NotNil(t, atr)
	assert.NoError(t, err)
	assert.Equal(t, atr.CurrentMessages, 1)

	data, err := msgQReceive(msgQReaderFd, atr.MaxMessageSize, time.Microsecond)
	assert.Equal(t, data, byteTestMsg)
	assert.NotNil(t, data)
	assert.NoError(t, err)
	atr, err = msgQWriter.getAttributes()
	assert.NotNil(t, atr)
	assert.NoError(t, err)
	assert.Zero(t, atr.CurrentMessages)
	atr, err = msgQAttributes(msgQReaderFd)
	assert.NotNil(t, atr)
	assert.NoError(t, err)
	assert.Zero(t, atr.CurrentMessages)

	err = unix.Close(msgQReaderFd)
	assert.NoError(t, err)

	err = msgQWriter.destroy()
	assert.NoError(t, err)
}
