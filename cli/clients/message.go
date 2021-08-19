package clients

import (
	"bufio"
	"encoding/json"
	"io"
)

// Message represents a single message sent by AppScope.
// There are four types:
//   - Headers sent at the start of each connections
//   - Events
//   - Metrics
//   - Payloads which are followed by `len` raw bytes.
type Message struct {
	Raw     string
	Data    MessageData
	Payload MessagePayload
}

// MessageData is the map we extract the NDJSON messages into.
type MessageData map[string]interface{}

// MessagePayload is the byte array we store a payload in.
type MessagePayload []byte

// IsHeader returns TRUE if the Message is a Header.
func (message Message) IsHeader() bool {
	_, hasFormat := message.Data["format"]
	return hasFormat
}

// IsEvent returns TRUE if the Message is an Event.
func (message Message) IsEvent() bool {
	dataType, hasType := message.Data["type"]
	return hasType && "evt" == dataType
}

// IsMetric returns TRUE if the Message is a Metric
func (message Message) IsMetric() bool {
	dataType, hasType := message.Data["type"]
	return hasType && "metric" == dataType
}

// IsPayload returns TRUE if the Message is a Payload
func (message Message) IsPayload() bool {
	dataType, hasType := message.Data["type"]
	return hasType && "payload" == dataType
}

// ReadMessage reads a Message from a reader.
func ReadMessage(reader *bufio.Reader) (*Message, error) {
	// Read up to and including the first newline into a string
	data, err := reader.ReadString('\n')
	if err != nil {
		if err != io.EOF {
			return nil, err
		}
		return nil, err
	}

	// New Message pointer with `data` set
	message := &Message{Raw: data}

	// Parse the JSON into message.Data
	if err := json.Unmarshal([]byte(data), &message.Data); err != nil {
		return nil, err
	}

	// Read payload if it's expected
	dataLen, hasLen := message.Data["len"]
	if message.IsPayload() && hasLen {
		need := int(dataLen.(float64))
		message.Payload = make([]byte, 0)
		got := 0
		for got < need {
			chunk := make([]byte, need-got)
			n, err := reader.Read(chunk)
			if err != nil {
				return nil, err
			}
			got += n
			message.Payload = append(message.Payload, chunk...)
		}
	}

	return message, nil
}
