package util

// ipcCmd represents the command structure
type ipcCmd int64

const (
	cmdGetScopeStatus ipcCmd = iota
)

func (cmd ipcCmd) string() string {
	switch cmd {
	case cmdGetScopeStatus:
		return "getScopeStatus"
	}
	return "unknown"
}

func (cmd ipcCmd) byte() []byte {
	return []byte(cmd.string())
}
