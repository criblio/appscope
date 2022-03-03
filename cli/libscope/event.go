package libscope

type Event struct {
	Type    string    `json:"type"`
	Id      string    `json:"id"`
	Channel string    `json:"_channel"`
	Body    EventBody `json:"body"`
}

type EventBody struct {
	Id         string
	SourceType string                 `json:"sourcetype"`
	Time       float64                `json:"_time"`
	Source     string                 `json:"source"`
	Host       string                 `json:"host"`
	Proc       string                 `json:"proc"`
	Cmd        string                 `json:"cmd"`
	Pid        int64                  `json:"pid"`
	Uid        int64                  `json:"uid"`
	Gid        int64                  `json:"gid"`
	Username   string                 `json:"username"`
	Groupname  string                 `json:"groupname"`
	Args       string                 `json:"args"`
	Data       map[string]interface{} `json:"data"`
}
