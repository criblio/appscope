package util

import (
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
)

func TestNewMq(t *testing.T) {
	const mqName string = "goTestMq"

	msgQ, err := newNonBlockMsgQReader(mqName)
	assert.NoError(t, err)
	assert.NotNil(t, msgQ)
	atr, err := msgQ.getAttributes()
	assert.NotNil(t, atr)
	assert.NoError(t, err)
	err = msgQ.close()
	assert.NoError(t, err)
	err = msgQ.unlink()
	assert.NoError(t, err)
}

func TestOpenMqNonExisting(t *testing.T) {
	msgQ, err := openMsgQWriter("not_existing")
	assert.Error(t, err)
	assert.Nil(t, msgQ)
}

func TestCommunicationMq(t *testing.T) {
	const mqName string = "goTestMq"
	byteTestMsg := []byte("Lorem Ipsum")

	msgQReader, err := newNonBlockMsgQReader(mqName)
	assert.NoError(t, err)
	assert.NotNil(t, msgQReader)
	atr, err := msgQReader.getAttributes()
	assert.NotNil(t, atr)
	assert.NoError(t, err)
	assert.Zero(t, atr.CurrentMessages)

	msgQWriter, err := openMsgQWriter(mqName)
	assert.NoError(t, err)
	assert.NotNil(t, msgQWriter)
	atr, err = msgQWriter.getAttributes()
	assert.NotNil(t, atr)
	assert.NoError(t, err)
	assert.Zero(t, atr.CurrentMessages)

	// Try to read from empty queue
	data, err := msgQReader.receive(time.Microsecond)
	assert.Nil(t, data)
	assert.Error(t, err)

	// Empty message send
	err = msgQWriter.send([]byte(""), time.Microsecond)
	assert.Error(t, err)
	atr, err = msgQReader.getAttributes()
	assert.NotNil(t, atr)
	assert.NoError(t, err)
	assert.Zero(t, atr.CurrentMessages)
	atr, err = msgQWriter.getAttributes()
	assert.NotNil(t, atr)
	assert.NoError(t, err)
	assert.Zero(t, atr.CurrentMessages)

	// Normal message send
	err = msgQWriter.send(byteTestMsg, time.Microsecond)
	assert.NoError(t, err)
	atr, err = msgQReader.getAttributes()
	assert.NotNil(t, atr)
	assert.NoError(t, err)
	assert.Equal(t, atr.CurrentMessages, 1)
	atr, err = msgQWriter.getAttributes()
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
	atr, err = msgQWriter.getAttributes()
	assert.NotNil(t, atr)
	assert.NoError(t, err)
	assert.Zero(t, atr.CurrentMessages)

	err = msgQWriter.close()
	assert.NoError(t, err)
	err = msgQReader.close()
	assert.NoError(t, err)
	err = msgQReader.unlink()
	assert.NoError(t, err)
}
