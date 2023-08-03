package notify

import (
	"fmt"

	"github.com/slack-go/slack"
)

type Notifier interface {
	Notify(string)
}

type slackNotifier struct {
	client    *slack.Client
	channelId string
}

// NewSlackNotifier
func NewSlackNotifier(token string, channelId string) slackNotifier {
	return slackNotifier{
		client:    slack.New(token),
		channelId: channelId,
	}
}

// Post message (notify) via Slack
func (s slackNotifier) Notify(msg string) {
	_, _, err := s.client.PostMessage(s.channelId,
		slack.MsgOptionText(msg, false),
		slack.MsgOptionAttachments(),
		slack.MsgOptionAsUser(false))
	if err != nil {
		fmt.Println(err)
	}
}
