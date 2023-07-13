package notify

import "github.com/slack-go/slack"

type SlackSender struct {
	client    *slack.Client
	channelId string
}

func getChannelId() string {
	return "C05HBUJTC9W"
}

// NewSlackSender
func NewSlackSender(token string, channelName string) *SlackSender {
	ss := &SlackSender{
		client:    slack.New(token),
		channelId: getChannelId(),
	}

	return ss
}

// perform notification via post message to slack
func (s *SlackSender) Notify(msg string) {
	s.client.PostMessage(s.channelId,
		slack.MsgOptionText(msg, false),
		slack.MsgOptionAttachments(),
		slack.MsgOptionAsUser(false))
}
