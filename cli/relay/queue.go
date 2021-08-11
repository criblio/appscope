package relay

type Message string

// Queue is a message queue
type Queue chan Message

// NewSenderQueue creates a queue
// Messages in this queue will be sent to the Cribl Destination
func NewSenderQueue() Queue {
	return make(Queue, Config.SenderQueueSize)
}
