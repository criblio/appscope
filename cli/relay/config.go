package relay

// Config is a global Configuration Instance
var Config Configuration

// Configuration is a Configuration Object Model
type Configuration struct {
	SenderQueueSize int
	CriblDest       string
	AuthToken       string
	Workers         int
}

// InitConfiguration initializes the global Configuration
func InitConfiguration() {
}
